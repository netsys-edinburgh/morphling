package com.example.confidant.utils;

import static com.example.confidant.utils.TrainCentral.sendForwardIntermediate;

import android.util.Log;

import com.alibaba.fastjson.JSONObject;
import com.example.confidant.faultTolerance.ReplicationUtils;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;
import com.example.confidant.request.OnlineRequest;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.Base64;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class TrainWorker {
    private static final String tag = "utils.TrainWorker";

    public static native Object[] forwardIntermediate(int iterId, Object[] inputData);
    public static native Object[] trainIntermediateLast(int iterId, Object[] inputData, Object[] labelsData);
    public static native Object[] backwardIntermediateWorker(int iterId, Object[] outputGrad);

    /*
        Set the basic info sent by the central node
     */
    public static void setBasicInfoHandler(JSONObject data) {
        List<Integer> point = General.convertJSONToIntegerList(data.getJSONArray("point"));
        String modelName = data.getString("modelName");

        JSONObject mapObject = data.getJSONObject("modelArgs");
        Map<String, Double> modelArgs = new HashMap<>();

        // The deserialized data type of JSON is BigDecimal, which needs to be converted
        mapObject.forEach((key, value) -> {
            modelArgs.put(key, ((BigDecimal) value).doubleValue());
        });
        int aggrInterval = data.getIntValue("aggrInterval");

        Common.setModelName(modelName);
        Common.setModelArgs(modelArgs);
        Training.setPartitionPoint(point);
        Training.setAggregateInterval(aggrInterval);

        Common.printLog("Receiving partition point: " + point.toString());

        boolean enableMPS = Training.isEnableMPS();
        modelArgs.put("for_parallel", enableMPS ? 1.0 : 0.0);

        if (enableMPS) {
            Common.printLog("Enable Multi-Processor Scheduler ...");
            Model.initMultiProcessorScheduler(modelName, modelArgs);
        }

        Common.printLog("Create sub-model and initializing optimizers ...");
        Model.createSubModel(point);

        if (Common.getWeightsPath().length() > 0) {
            Common.printLog("Loading pre-trained weights from " + Common.getWeightsPath());
            Model.loadPretrainedSubWeights(point);
        }

        Optimizer.createSubOptimizer();

        // Initialize the executor in the cpp
        Common.printLog("Initializing training ...");
        Model.initTrain(Common.getBatchSize(), Common.getWorkerNum(), Common.getDeviceIdx());

        Common.printLog("Setup basic info finished ...");
    }

    /*
        Set the basic info sent by the central node
     */
    public static void setBasicInfoHandler(Map<String, Object> data) {
        List<Integer> point = (List<Integer>) data.get("point");
        String modelName = (String) data.get("modelName");

        // use new HashMap<>() to convert the map into modifiable one
        Map<String, Double> modelArgs = new HashMap<>((Map<String, Double>) data.get("modelArgs"));

        int aggrInterval = (int) data.get("aggrInterval");

        Common.setModelName(modelName);
        Common.setModelArgs(modelArgs);
        Training.setPartitionPoint(point);
        Training.setAggregateInterval(aggrInterval);

        Common.printLog("Receiving partition point: " + point.toString());

        boolean enableMPS = Training.isEnableMPS();
        modelArgs.put("for_parallel", enableMPS ? 1.0 : 0.0);
        if (enableMPS) {
            Common.printLog("Enable Multi-Processor Scheduler ...");
            Model.initMultiProcessorScheduler(modelName, modelArgs);
        }

        Common.printLog("Create sub-model and initializing optimizers ...");
        Model.createSubModel(point);

        if (Common.getWeightsPath().length() > 0) {
            Common.printLog("Loading pre-trained weights from " + Common.getWeightsPath());
            Model.loadPretrainedSubWeights(point);
        }


        Optimizer.createSubOptimizer();

        // Initialize the executor in the cpp
        Common.printLog("Initializing training ...");
        Model.initTrain(Common.getBatchSize(), Common.getWorkerNum(), Common.getDeviceIdx());

        Common.printLog("Setup basic info finished ...");
    }

    /**
     * Init the commit variable for the current training
     */
    public static void initEpoch(int epoch, double lr, int dataLen) {
        Log.i(tag, "Initializing epoch");
        Optimizer.setLearningRate(lr);
        Optimizer.resetOptimizer();

        // TODO: fault tolerance not yet implemented
        Training.initCommit();
        Training.setCommitEpoch(epoch);
        Training.setCommitLen(dataLen);

        // Init the train for the current epoch in cpp
        Model.initTrainEpoch();
    }

    /**
     * Handle the forwarding of the intermediate data
     */
    public static void handleForward(Map<String, Object> data) {
        Object[] inputData = (Object[]) data.get("data");
        int curIdx = (int) data.get("modelIdx");
        assert (curIdx == Common.getDeviceIdx());

        int iterId = (int) data.get("iterId");
        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        // Performing weights replication
        if (FaultTolerance.getLocalReplicationInterval() > 0 && iterId > 0 && iterId % FaultTolerance.getLocalReplicationInterval() == 0) {
            ReplicationUtils.replicateWeights(FaultTolerance.ReplicationType.LOCAL_REPLICATION);
        }

        if (FaultTolerance.getGlobalReplicationInterval() > 0 && iterId > 0 && iterId % FaultTolerance.getGlobalReplicationInterval() == 0) {
            ReplicationUtils.replicateWeights(FaultTolerance.ReplicationType.GLOBAL_REPLICATION);
        }

        commitLock.lock();

        // Same as the central node, the while condition guarantees the 1F1B rule
        int batchDiff = Common.getWorkerNum() - curIdx;
        while (!((int) commit.get("forwardId") == iterId - 1 && (iterId - FaultTolerance.getStartIterId() - batchDiff < 0 || (int) commit.get("backwardId") == iterId - batchDiff))) {
            Log.i(tag, String.format("Forward need to wait: forward id: %d, backward id: %d, iter id: %d", (int) commit.get("forwardId"), (int) commit.get("backwardId"), iterId));
            try {
                commitCondition.await();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        if (curIdx != Common.getWorkerNum() - 1) {
            // Not the last worker, send the intermediate data to the next worker

            Long startTime = System.currentTimeMillis();

            Object[] output = forwardIntermediate(iterId, inputData);

            Training.updateForwardTime(System.currentTimeMillis() - startTime);

            if (iterId % Training.getLogInterval() == 0) {
                int epoch = (int) commit.get("epoch");
                Common.printLog(String.format("Batch %d of epoch %d forward finish, time:%f s", iterId, epoch,
                        (System.currentTimeMillis() - startTime)/1000.0));
            }
            try {
                int forwardId = (int) commit.get("forwardId");
                commit.put("forwardId", forwardId + 1);
                commitCondition.signalAll();
            } finally {
                commitLock.unlock();
            }

            new Thread(new Runnable() {
                @Override
                public void run() {
                    sendForwardIntermediate(curIdx, iterId, output);
                }
            }).start();
        } else {
            // the last worker, start the backward process
            Common.printLog("Start computing the batch " + iterId);
            Object[] labels = Training.getLabels(iterId);
            if (labels == null) {
                Common.printLog("No labels for the batch " + iterId);
                return ;
            }

            Long startTime = System.currentTimeMillis();
            Object[] output = trainIntermediateLast(iterId, inputData, labels);

            Object[] lossVar = (Object[]) output[0];
             float loss = ((float[]) lossVar[0])[0];
//            float loss = ((List<Float>) lossVar[0]).get(0);

            if (iterId % Training.getLogInterval() == 0) {
                int epoch = (int) commit.get("epoch");
                Common.printLog(String.format("Batch %d of epoch %d farward and backward finish, loss %f, Time:%f s",
                        iterId, epoch, loss, (System.currentTimeMillis() - startTime)/1000.0));
            }

            try {
                int forwardId = (int) commit.get("forwardId");
                int backwardId = (int) commit.get("backwardId");
                commit.put("forwardId", forwardId + 1);
                commit.put("backwardId", backwardId + 1);
                commitCondition.signalAll();
            } finally {
                commitLock.unlock();
            }

            new Thread(new Runnable() {
                @Override
                public void run() {
                    sendBackward(curIdx - 1, iterId, (Object[]) output[1]);
                }
            }).start();
        }
    }

    /*
       Use the gradients received from other workers to compute backward
    */
    public static void handleBackward(Map<String, Object> data) {
        int curIdx = (int) data.get("modelIdx");
        assert (curIdx == Common.getDeviceIdx());
        int iterId = (int) data.get("iterId");

        Object[] grad = (Object[]) data.get("data");

        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        commitLock.lock();

        int batchDiff = Common.getWorkerNum() - curIdx;

        while (!((int) commit.get("backwardId") == iterId - 1 && (int) commit.get("forwardId") == iterId + batchDiff - 1)) {
            Log.i(tag, String.format("Backward need to wait: forward id: %d, backward id: %d, iter id: %d", (int) commit.get("forwardId"), (int) commit.get("backwardId"), iterId));
            if ((int) commit.get("backwardId") == iterId - 1 && (int) commit.get("forwardId") == (int) commit.get("dataLen") - 1) {
                Log.i(tag, "Current epoch finish, no more wait...");
                break;
            }

            try {
                commitCondition.await();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        Date startDate = new Date();
        long startTime = startDate.getTime();
        Object[] inputGrad = backwardIntermediateWorker(iterId, grad);

        Date endDate = new Date();
        long endTime = endDate.getTime();

        Training.updateBackwardTime(endTime - startTime);

        int epoch = (int) commit.get("epoch");
        Common.printLog(String.format("Batch %d of epoch %d backward finish, start time: %s, end time: %s, time:%f s", iterId, epoch, startTime, endTime,
                (endTime - startTime)/1000.0));

        Log.i(tag, iterId + " Backward");

        try {
            int backwardId = (int) commit.get("backwardId");
            commit.put("backwardId", backwardId + 1);
            commitCondition.signalAll();
        } finally {
            commitLock.unlock();
        }

        new Thread(new Runnable() {
            @Override
            public void run() {
                sendBackward(curIdx - 1, iterId, inputGrad);
            }
        }).start();
    }

    public static void sendForwardIntermediate(int targetIdx, int iterId, Object[] output) {
        String nextUrl = Common.getUrlFromWorker(Common.getDeviceIdx() + 1);
        int modelVersion = 1;
        double lr = Optimizer.getLearningRate();
        Date startSendDate = new Date();

        // OnlineRequest.sendTrainForward(nextUrl, iterId,targetIdx + 1, modelVersion, lr, output);
        OnlineGRPCRequest.sendTrainForward(nextUrl, iterId, targetIdx + 1, modelVersion, lr, output);

        Date endSendDate = new Date();
        Common.printLog(String.format("Batch %d forward send finish, start time: %s, end time: %s, time:%f s", iterId, startSendDate.getTime(), endSendDate.getTime(),
                (endSendDate.getTime() - startSendDate.getTime())/1000.0));
    }

    public static void sendBackward(int targetIdx, int iterId, Object[] output) {
        String backwardUrl = Common.getUrlFromWorker(targetIdx);

        Date startSendDate = new Date();

        // OnlineRequest.sendTrainBackward(backwardUrl, output, targetIdx, iterId);
        if (targetIdx > 0) {
            OnlineGRPCRequest.sendWorkerTrainBackward(backwardUrl, output, targetIdx, iterId);
        } else {
            OnlineGRPCRequest.sendCentralTrainBackward(backwardUrl, output, targetIdx, iterId);
        }

        Date endSendDate = new Date();
        Common.printLog(String.format("Batch %d backward send finish, start time: %s, end time: %s, time:%f s", iterId, startSendDate.getTime(), endSendDate.getTime(),
                (endSendDate.getTime() - startSendDate.getTime())/1000.0));
    }
}
