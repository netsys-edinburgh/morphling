package com.example.confidant.utils;

import android.util.Log;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.example.confidant.faultTolerance.PassiveFTHandler;
import com.example.confidant.faultTolerance.ReplicationUtils;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;
import com.example.confidant.request.OnlineRequest;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class TrainCentral {
    private static final String tag = "utils.TrainCentral";

    public static native Object[] forwardOneBatch(int iterId);
    public static native void skipOneBatch();
    public static native void backwardIntermediateCentral(int iterId, Object[] outputGrad);
    /*
        Create the sub-model and dataset, and init the variables that are needed for training
     */
    public static void startTrain() {
        Common.printLog("Collaborative train start");
        List<Integer> point = Training.getPartitionPoint();

        String modelName = Common.getModelName();
        Map<String, Double> modelArgs = Common.getModelArgs();
        boolean enableMPS = Training.isEnableMPS();
        modelArgs.put("for_parallel", enableMPS ? 1.0 : 0.0);

        if (enableMPS) {
            Common.printLog("Enable Multi-Processor Scheduler ...");
            Model.initMultiProcessorScheduler(modelName, modelArgs);
        }

        Common.printLog("Creating sub model ...");
        Common.setDeviceIdx(0);
        Model.createSubModel(point);

        if (Common.getWeightsPath().length() > 0) {
            Common.printLog("Loading pre-trained weights from " + Common.getWeightsPath());
            Model.loadPretrainedSubWeights(point);
        }

        Common.printLog("Setting up dataset: " + Common.getDatasetName() + " with batch size " + Common.getBatchSize());
        Dataset.createDataset(Common.getDatasetName(), Common.getDatasetPath(), Common.getBatchSize());

        Common.printLog("Initializing optimizer ...");
        Optimizer.createSubOptimizer();

        Common.initSemaphore();

        Common.printLog("Initializing executor and batch size in jni ...");
        Model.initTrain(Common.getBatchSize(), Common.getWorkerNum(), Common.getDeviceIdx());

        TrainDistribute();
    }

    /*
        Main logic of the distributed training
     */
    public static void TrainDistribute() {
        Training.initCommit();

        Common.printLog("Start formal distributed training");
        int dataLen = Dataset.getDataLen();
        Training.setCommitLen(dataLen);

        int epochs = Training.getEpochs();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double lr = Optimizer.getLearningRate();

            Map<String, String> workers = Common.getWorkers();
            for (Map.Entry<String, String> entry : workers.entrySet()) {
                String idx = entry.getKey();
                if (!idx.equals("0")) {
                    String url = entry.getValue();
                    // OnlineRequest.sendStartEpoch(url, epoch, lr, dataLen);
                    OnlineGRPCRequest.sendStartEpoch(url, epoch, lr, dataLen);
                }
            }

            Optimizer.resetOptimizer(); // weight pool and batch counter reset here
            // TODO: Update profiling interval and reset time

            // Fault tolerance related
            if (epoch > 0) {
                FaultTolerance.setStartIterId(0);
            }

            // TODO: newly added for fault tolerance
            FaultTolerance.clearReceivedIterIds();

            Training.setStartTime(System.currentTimeMillis());
            trainEpochDistribute(epoch);
        }
    }

    /*
        Main logic of distributively training one epoch
    */
    public static void trainEpochDistribute(int epoch) {
        Model.initTrainEpoch();
        Map<String, Object> commit = Training.getCommit();
        int iterations = (int) commit.get("dataLen");

        for (int iterId = 0; iterId < iterations; iterId++) {
            // for checkpointing-based training
//            if (iterId < FaultTolerance.getStartIterId()) {
//                if (iterId == FaultTolerance.getStartIterId() - 1) {
//                    commit.put("forwardId", iterId);
//                    commit.put("backwardId", iterId);
//                }
//                Common.printLog("Skipping iterId: " + iterId);
//                skipOneBatch();
//                continue;
//            }

            Semaphore sem = Common.getSemaphore();
            try {
                sem.acquire();

                while (FaultTolerance.getSystemStatus() != FaultTolerance.SystemStatus.NORMAL) {
//                    Common.printLog("Try to input iterId " + iterId + ", but system status is not normal, system status: " + FaultTolerance.getSystemStatus());
                    Thread.sleep(1000);
                }

                trainOneBatchDistribute(epoch, iterId);

            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /*
        Main logic of distributively training one batch, with sending the intermediate batch to the next worker
     */
    public static void trainOneBatchDistribute(int epoch, int iterId) {
        // Weights Replication, no global replication on the central node
        if (FaultTolerance.getLocalReplicationInterval() > 0 && iterId > 0 && iterId % FaultTolerance.getLocalReplicationInterval() == 0) {
            ReplicationUtils.replicateWeights(FaultTolerance.ReplicationType.LOCAL_REPLICATION);
        }

        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        commitLock.lock();

        int batchDiff = Common.getWorkerNum();
        // Key condition check to guarantee the 1F1B rule
        // If the iterId does not satisfy the condition, the related thread is blocked
        while (!((int) commit.get("forwardId") == iterId - 1 && (iterId - FaultTolerance.getStartIterId() - batchDiff < 0 || (int) commit.get("backwardId") == iterId - batchDiff))) {
            Log.i(tag, String.format("Forward need to wait: forward id: %d, backward id: %d, iter id: %d", (int) commit.get("forwardId"), (int) commit.get("backwardId"), iterId));
            try {
                commitCondition.await();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        // for record
        Date startDate = new Date();
        long startTime = startDate.getTime();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String startDateStr = sdf.format(startDate);

        // long startTime = System.currentTimeMillis();

        // forward one data
        Object[] output = forwardOneBatch(iterId);

        if (iterId % Training.getLogInterval() == 0) {
            Date endDate = new Date();
            long endTime = endDate.getTime();
            String endDateStr = sdf.format(endDate);
            Common.printLog(String.format("Batch %d of epoch %d compute finish, start time: %s, end time: %s, time:%f s", iterId, epoch, startTime, endTime,
                    (endTime - startTime)/1000.0));
        }

        // Update the commit variable and wake up all the blocked threads
        try {
            int forwardId = (int) commit.get("forwardId");
            commit.put("forwardId", forwardId + 1);
            commitCondition.signalAll();
        } finally {
            commitLock.unlock();
        }

        // Common.printLog(iterId + " Forward: " + ((float[])((Object[])output[0])[0])[0]);

        // Send the intermediate output using another thread without blocking the computation of the next batch
        // Set the passive timeout for passive fault tolerance
        new Thread(new Runnable() {
            @Override
            public void run() {
                sendForwardIntermediate(epoch, iterId, output);
                setPassiveTimeout(iterId);
            }
        }).start();
    }

    public static void sendForwardIntermediate(int epoch, int iterId, Object[] output) {
        String nextUrl = Common.getUrlFromWorker(Common.getDeviceIdx() + 1);
        String lossUrl = Common.getUrlFromWorker(Common.getWorkerNum() - 1);

        int modelVersion = 1;
        double lr = Optimizer.getLearningRate();

        // The labels should be send to the last worker
        // OnlineRequest.sendLabels(lossUrl, iterId, (Object[]) output[1]);
        OnlineGRPCRequest.sendLabels(lossUrl, iterId, (Object[]) output[1]);

        Date startSendDate = new Date();

        // OnlineRequest.sendTrainForward(nextUrl, iterId,1, modelVersion, lr, (Object[]) output[0]);
        OnlineGRPCRequest.sendTrainForward(nextUrl, iterId, 1, modelVersion, lr, (Object[]) output[0]);

//        Date endSendDate = new Date();
//        Common.printLog(String.format("Batch %d of epoch %d send finish, start time: %s, end time: %s, time:%f s", iterId, epoch, startSendDate.getTime(), endSendDate.getTime(),
//                (endSendDate.getTime() - startSendDate.getTime())/1000.0));
        Common.printLog(String.format("Forward batch %d send finish, start time: %s", iterId, startSendDate.getTime()));
    }

    /*
        Called by the sparkAPI, handling the backward of the received gradients
     */
    public static void handleBackwardIntermediate(JSONObject data) {
        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        commitLock.lock();

        int batchDiff = Common.getWorkerNum();
        int iterId = data.getIntValue("iterId");

        // Key condition check to guarantee the 1F1B rule
        // If the iterId does not satisfy the condition, the related thread is blocked
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

        Object[] outputGrad = new Object[0];
        if (Objects.equals(data.getString("framework"), "Pytorch")) {
            JSONArray inputJsonArr = data.getJSONArray("data");
            outputGrad = General.parseJSONObject(inputJsonArr);
        } else {
            String gradDataArr = data.getString("data");
            byte[] gradDataBytes = Base64.getDecoder().decode(gradDataArr);
            // outputGrad = new Object[0];
            try {
                outputGrad = (Object[]) General.convertToObjectArray(gradDataBytes);
            } catch (IOException e) {
                // throw new RuntimeException(e);
                Common.printLog("IOException in convertToObjectArray: " + e.getMessage());
            } catch (ClassNotFoundException e) {
                // throw new RuntimeException(e);
                Common.printLog("ClassNotFoundException in convertToObjectArray: " + e.getMessage());
            }
        }

        FaultTolerance.addReceivedIterId(iterId);

        // for record
        Date startDate = new Date();
        long startTime = startDate.getTime();

        backwardIntermediateCentral(iterId, outputGrad);

        Date endDate = new Date();
        long endTime = endDate.getTime();

//        int epoch = (int) commit.get("epoch");
//        Common.printLog(String.format("Batch %d of epoch %d backward finish, start time: %s, end time: %s, time:%f s", iterId, epoch, startTime, endTime,
//                (endTime - startTime)/1000.0));

//        if(iterId == 9){
//            Common.printLog(String.format("Batch 10 Finished, time:%f s",
//                    (System.currentTimeMillis() - Training.getStartTime())/1000.0));}
//        else {
//            Common.printLog(String.format("Computing endless!"));
//        }

        try {
            int backwardId = (int) commit.get("backwardId");
            commit.put("backwardId", backwardId + 1);
            commitCondition.signalAll();
        } finally {
            commitLock.unlock();
        }

        Semaphore sem = Common.getSemaphore();
        sem.release();
    }

    public static void handleBackwardIntermediate(Map<String, Object> data) {
        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        commitLock.lock();

        int batchDiff = Common.getWorkerNum();
        int iterId = (int) data.get("iterId");
        Date endSendDate = new Date();
//        Common.printLog(String.format("Backward batch %d receive finished, end time: %s", iterId, endSendDate.getTime()));

        Common.printLog("Backward start iterId: " + iterId);
        // Key condition check to guarantee the 1F1B rule
        // If the iterId does not satisfy the condition, the related thread is blocked
        int curForwardId = (int) commit.get("forwardId");
        while (!((int) commit.get("backwardId") == iterId - 1 && (int) commit.get("forwardId") == iterId + batchDiff - 1)) {
            Log.i(tag, String.format("Backward need to wait: forward id: %d, backward id: %d, iter id: %d", (int) commit.get("forwardId"), (int) commit.get("backwardId"), iterId));
            if ((int) commit.get("backwardId") == iterId - 1 && (int) commit.get("forwardId") == (int) commit.get("dataLen") - 1) {
                Log.i(tag, "Current epoch finish, no more wait...");
                break;
            }

            try {
                commitCondition.await();

                if ((int) commit.get("forwardId") < curForwardId) {
                    Common.printLog("handleBackwardIntermediate(): Train process fault happens, backward teminates ...");
                    try {
                        commitCondition.signalAll();
                    } finally {
                        commitLock.unlock();
                    }
                    Semaphore sem = Common.getSemaphore();
                    sem.release();
                    return ;
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        Object[] outputGrad = (Object[]) data.get("data");

        FaultTolerance.addReceivedIterId(iterId);

        // for record
        Date startDate = new Date();
        long startTime = startDate.getTime();

        backwardIntermediateCentral(iterId, outputGrad);
        Date endDate = new Date();
        long endTime = endDate.getTime();

        int epoch = (int) commit.get("epoch");
        Common.printLog(String.format("Batch %d of epoch %d backward finish, start time: %s, end time: %s, time:%f s", iterId, epoch, startTime, endTime,
                (endTime - startTime)/1000.0));

//        if(iterId == 9){
//            Common.printLog(String.format("Batch 10 Finished, time:%f s",
//                    (System.currentTimeMillis() - Training.getStartTime())/1000.0));}
//        else {
//            Common.printLog(String.format("Computing endless!"));
//        }

        try {
            int backwardId = (int) commit.get("backwardId");
            commit.put("backwardId", backwardId + 1);
            commitCondition.signalAll();
        } finally {
            commitLock.unlock();
        }

        Semaphore sem = Common.getSemaphore();
        sem.release();
    }

    private static void setPassiveTimeout(int iterId) {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(FaultTolerance.getBackwardTimeout());
                    if (!FaultTolerance.isReceived(iterId) && FaultTolerance.getSystemStatus() == FaultTolerance.SystemStatus.NORMAL) {
                        Common.printLog("Backward data is not received in time, iter id: " + iterId + ", start fault tolerance");
                        PassiveFTHandler.backwardTimeoutHandler(iterId);
                    }
                } catch (InterruptedException e) {
                    Log.e(tag, "Passive FT thread sleep failed: " + e.getMessage());
                }

            }
        }, "WorkerFaultTolerance.weightRedistributeWorkerHandler");

        thread.start();
    }

}
