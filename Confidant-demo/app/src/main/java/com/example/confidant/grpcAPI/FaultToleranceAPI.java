package com.example.confidant.grpcAPI;

import static android.content.Context.BATTERY_SERVICE;

import android.content.Context;
import android.os.BatteryManager;
import android.util.Pair;

import com.alibaba.fastjson.JSONObject;
import com.example.confidant.faultTolerance.PassiveFTHandler;
import com.example.confidant.faultTolerance.ProactiveFTHandler;
import com.example.confidant.faultTolerance.ProactiveFTUtils;
import com.example.confidant.faultTolerance.RedistributionUtils;
import com.example.confidant.faultTolerance.ReplicationUtils;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.rpc.api.CommitFaultSyncRequest;
import com.example.confidant.rpc.api.CommonResponse;
import com.example.confidant.rpc.api.DcInfoResponse;
import com.example.confidant.rpc.api.EmptyRequest;
import com.example.confidant.rpc.api.FaultToleranceGreeterGrpc;
import com.example.confidant.rpc.api.NotifySubstituteDeviceRequest;
import com.example.confidant.rpc.api.ParamsFromRemoteRequest;
import com.example.confidant.rpc.api.PassiveRedistributionFinishRequest;
import com.example.confidant.rpc.api.ProactiveFTUpdateWorkersRequest;
import com.example.confidant.rpc.api.RestartSyncStateRequest;
import com.example.confidant.rpc.api.SendTensorTestRequest;
import com.example.confidant.rpc.api.TensorResponse;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.rpc.api.UnifiedFloatTensorList;
import com.example.confidant.rpc.api.UnifiedIntTensor;
import com.example.confidant.rpc.api.WeightsRedistributeRequest;
import com.example.confidant.rpc.api.WeightsReplicationRequest;
import com.example.confidant.rpc.api.WorkerExitRequest;
import com.example.confidant.utils.General;
import com.example.confidant.utils.GrpcUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

import io.grpc.stub.StreamObserver;

public class FaultToleranceAPI extends FaultToleranceGreeterGrpc.FaultToleranceGreeterImplBase {
    @Override
    public void weightsReplication(WeightsReplicationRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        FaultTolerance.ReplicationType replicationType = FaultTolerance.ReplicationType.values()[request.getReplicationType()];

        for (Map.Entry<String, UnifiedFloatTensorList> entry : request.getWeightsMap().entrySet()) {
            List<UnifiedFloatTensor> tensorList = entry.getValue().getWeightList();
            Object[] dataObjArr = GrpcUtils.convertUnifiedFloatTensorListToObjectArr(tensorList);
            FaultTolerance.addWeightsMap(entry.getKey(), dataObjArr, replicationType);
        }

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();

        Common.printLog("Receiving weights replication with status: " + commonResponse.getStatus());
    }

    @Override
    public void commitFaultSync(CommitFaultSyncRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        List<Integer> partitionPoint = request.getPointList();
        int iterId = request.getIterId();

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                RedistributionUtils.commitFaultSyncHandler(partitionPoint, iterId);
            }
        }, "WorkerFaultTolerance.commitWorker");

        thread.start();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();

        Common.printLog("Reply to client: " + commonResponse.getStatus());
    }

    @Override
    public void weightsRedistribute(WeightsRedistributeRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        List<String> failedIdx = request.getFailedSetList();
        List<Integer> points = request.getPointList();

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                RedistributionUtils.weightRedistributeWorkerHandler(failedIdx, points);
            }
        }, "WorkerFaultTolerance.weightRedistributeWorkerHandler");

        thread.start();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();

        Common.printLog("Reply to client: " + commonResponse.getStatus());
    }

    @Override
    public void restartSyncState(RestartSyncStateRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        Map<Integer, String> workers = request.getWorkersMap();
        Map<String, String> workersStr = new HashMap<>();
        for (Map.Entry<Integer, String> entry : workers.entrySet()) {
            workersStr.put(String.valueOf(entry.getKey()), entry.getValue());
        }

        List<Integer> points = request.getPointList();
        int idx = request.getIdx();

        PassiveFTHandler.restartSyncStates(idx, workersStr, points);

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();

        Common.printLog("Reply to client: " + commonResponse.getStatus());
    }

    @Override
    public void passiveRedistributionFinish(PassiveRedistributionFinishRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        int idx = request.getIdx();
        Common.printLog("Redistribution finish notified by " + idx);
        FaultTolerance.addRedistributedDevice(String.valueOf(idx));

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();

        Common.printLog("Reply to client: " + commonResponse.getStatus());
    }

    @Override
    public void paramsFromRemote(ParamsFromRemoteRequest request, StreamObserver<TensorResponse> responseObserver) {
        List<Integer> layers = request.getLayersList();
        Map<String, UnifiedFloatTensorList> ret = new HashMap<>();

        for (int i = 0; i < layers.size(); i++) {
            Object[] curWeights = FaultTolerance.getLocalWeightsByLayer(String.valueOf(layers.get(i)));
            List<UnifiedFloatTensor> tensorList = GrpcUtils.convertObjectArrToUnifiedFloatTensorList(curWeights);
            UnifiedFloatTensorList unifiedFloatTensorList = UnifiedFloatTensorList.newBuilder()
                    .addAllWeight(tensorList)
                    .build();
            ret.put(String.valueOf(layers.get(i)), unifiedFloatTensorList);
        }

        TensorResponse tensorResponse = TensorResponse.newBuilder()
                .putAllWeights(ret)
                .build();

        responseObserver.onNext(tensorResponse);
        responseObserver.onCompleted();

    }

    @Override
    public void passiveCommitFinish(EmptyRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        Common.printLog("Passive commit finish notified");
        FaultTolerance.addPassiveCommitDoneWorker(0);

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void workerExit(WorkerExitRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                ProactiveFTHandler.handlerHelper(String.valueOf(request.getDeviceIdx()));
            }
        }, "CentralFaultTolerance.workerExit");
        thread.start();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void dCInfo(EmptyRequest request, StreamObserver<DcInfoResponse> responseObserver) {
        // Profile computing capacity
        List<Float> computingCapacity = Common.getCurrentDeviceCapacity();
        if (computingCapacity.size() == 0) {
            computingCapacity = ProactiveFTUtils.profileComputingCapacityVector();
        }

        // Get battery level
        Context context = Common.getContext();
        BatteryManager bm = (BatteryManager) context.getSystemService(BATTERY_SERVICE);
        Integer chargeCounter = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);

        DcInfoResponse dcInfoResponse = DcInfoResponse.newBuilder().addAllCcv(computingCapacity).setBattery(chargeCounter).build();
        responseObserver.onNext(dcInfoResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void notifySubstituteDevice(NotifySubstituteDeviceRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        String quitUrl = request.getQuitUrl();
        String centralUrl = request.getCentralUrl();
        String quitIdx = String.valueOf(request.getQuitIdx());
        List<Integer> points = request.getPointsList();

        int epoch = request.getEpoch();
        float lr = request.getLr();
        int dataLen = request.getLen();

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                Training.initCommit();
                Training.setCommitEpoch(epoch);
                Training.setCommitLen(dataLen);

                ProactiveFTUtils.fetchWeightsFromQuitDeviceHandler(quitUrl, quitIdx, centralUrl, points);
            }
        }, "WorkerFaultTolerance.weightsFromQuitDevice");
        thread.start();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void notifyCentralFetchFinish(EmptyRequest request, StreamObserver<CommonResponse> responseObserver) {
        Common.printLog("Weights fetch finish notified from quitting device");
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        FaultTolerance.setProactiveFetchWeightsFinished(true);
        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void weightsFromQuitDevice(EmptyRequest request, StreamObserver<TensorResponse> responseObserver) {
        Common.printLog("Weights from the quit device requested");

        List<Integer> points = Training.getPartitionPoint();
        Pair<Integer, Integer> layersPair = General.getLayerFromPoint(points, Common.getDeviceIdx());
        int startLayer = layersPair.first;
        int endLayer = layersPair.second;
        if (endLayer == -1) {
            endLayer = Training.getTotalLayers() - 1;
        }

        Map<String, UnifiedFloatTensorList> ret = new HashMap<>();

        for (int i = startLayer; i <= endLayer; i++) {
            Object[] curWeights = ReplicationUtils.getSubModelWeightsByLayer(i, true);
            List<UnifiedFloatTensor> tensorList = GrpcUtils.convertObjectArrToUnifiedFloatTensorList(curWeights);
            UnifiedFloatTensorList unifiedFloatTensorList = UnifiedFloatTensorList.newBuilder()
                    .addAllWeight(tensorList)
                    .build();
            ret.put(String.valueOf(i), unifiedFloatTensorList);
        }

        TensorResponse tensorResponse = TensorResponse.newBuilder()
                .putAllWeights(ret)
                .build();

        responseObserver.onNext(tensorResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void proactiveFTUpdateWorkers(ProactiveFTUpdateWorkersRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        Common.setDeviceIdx(Integer.parseInt(String.valueOf(request.getIdx())));
        Map<Integer, String> workers = request.getWorkersMap();
        // convert all integer key into string
        Map<String, String> workersStr = new HashMap<>();
        for (Map.Entry<Integer, String> entry : workers.entrySet()) {
            workersStr.put(String.valueOf(entry.getKey()), entry.getValue());
        }

        int startId = request.getStartIterId();
        FaultTolerance.setStartIterId(startId);

        if (Training.getCommit() != null) {
            Common.printLog("Init commit with start iter id: " + startId);
            Map<String, Object> commit = Training.getCommit();
            Lock commitLock = (Lock) commit.get("lock");
            Condition commitCondition = (Condition) commit.get("lockCondition");

            assert commitLock != null;
            assert commitCondition != null;

            commitLock.lock();
            try {
                commit.put("forwardId", startId - 1);
                commit.put("backwardId", startId - 1);
                commitCondition.signalAll();
            } finally {
                commitLock.unlock();
            }
        }

        Common.printLog("proactiveFTUpdateWorkers(): Set as worker node with idx " + Common.getDeviceIdx());
        Common.setWorkers(workersStr);

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }
}
