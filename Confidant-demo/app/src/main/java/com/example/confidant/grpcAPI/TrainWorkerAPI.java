package com.example.confidant.grpcAPI;

import com.alibaba.fastjson.JSONObject;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.rpc.api.BandwidthResponse;
import com.example.confidant.rpc.api.CommonResponse;
import com.example.confidant.rpc.api.EmptyRequest;
import com.example.confidant.rpc.api.HandleForwardRequest;
import com.example.confidant.rpc.api.LabelsRequest;
import com.example.confidant.rpc.api.MemoryResponse;
import com.example.confidant.rpc.api.SendTensorTestRequest;
import com.example.confidant.rpc.api.SendTrainBackwardRequest;
import com.example.confidant.rpc.api.SetBasicInfoRequest;
import com.example.confidant.rpc.api.StartEpochRequest;
import com.example.confidant.rpc.api.TransformerBlockResponse;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.rpc.api.UnifiedIntTensor;
import com.example.confidant.rpc.api.UpdateWorkersRequest;
import com.example.confidant.rpc.api.WorkerGreeterGrpc;
import com.example.confidant.utils.General;
import com.example.confidant.utils.GrpcUtils;
import com.example.confidant.utils.TrainWorker;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.grpc.stub.StreamObserver;

public class TrainWorkerAPI extends WorkerGreeterGrpc.WorkerGreeterImplBase {
    @Override
    public void sendTensorTest(SendTensorTestRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("Success")
                .build();

        UnifiedFloatTensor tensor = request.getTensor();
        // Common.printLog("Received tensor: " + tensor.getDataList() + ", " + tensor.getDataType() + ", " + tensor.getDataShapeList());

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();

        Common.printLog("Reply to client: " + commonResponse.getStatus());
    }

    @Override
    public void handleForward(HandleForwardRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        Date previousEndTime = new Date();
        Common.printLog(String.format("Batch %d send finish, previous forward end time: %s",request.getIterId(),previousEndTime.getTime()));

        List<UnifiedFloatTensor> reqData = request.getDataList();
        Object[] dataObjArr = GrpcUtils.convertUnifiedFloatTensorListToObjectArr(reqData);

        Map<String, Object> data = new HashMap<>();
        data.put("modelIdx", request.getModelIdx());
        data.put("iterId", request.getIterId());
        data.put("version", request.getVersion());
        data.put("data", dataObjArr);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                TrainWorker.handleForward(data);
            }
        }, "TrainWorker.handleForward");
        thread.start();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void sendTrainBackward(SendTrainBackwardRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        Date previousEndTime = new Date();
        Common.printLog(String.format("Batch %d send finish, previous backward end time: %s",request.getIterId(),previousEndTime.getTime()));


        List<UnifiedFloatTensor> reqData = request.getDataList();
        Object[] dataObjArr = GrpcUtils.convertUnifiedFloatTensorListToObjectArr(reqData);

        Map<String, Object> data = new HashMap<>();
        data.put("modelIdx", request.getModelIdx());
        data.put("iterId", request.getIterId());
        data.put("data", dataObjArr);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                TrainWorker.handleBackward(data);
            }
        }, "TrainWorker.sendTrainBackward");
        thread.start();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void labels(LabelsRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        List<UnifiedIntTensor> labels = request.getLabelsList();
        int iterId = request.getIterId();
        // UnifiedIntTensor labels = data.get(0);
        assert labels != null;
        Object[] labelsObj = GrpcUtils.convertUnifiedIntTensorListToObjectArr(labels);
        // Object[] labelsObj = GrpcUtils.convertUnifiedIntTensorToObjectArr(labels);

        Training.updateLabelsPool(iterId, labelsObj);

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void isAvailable(EmptyRequest request, StreamObserver<CommonResponse> responseObserver) {
        String res = "ok";
        if (Common.getDeviceIdx() != -1) {
            res = "Occupied";
        }
        Common.printLog("GRPC Request: Found by a central node");
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus(res)
                .build();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void startEpoch(StartEpochRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        int epoch = request.getEpoch();
        double lr = request.getLr();
        int dataLen = request.getLen();
        TrainWorker.initEpoch(epoch, lr, dataLen);

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void updateWorkers(UpdateWorkersRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        // for passive ft
        if (Common.getDeviceIdx() != -1) {
            Common.setPrevDeviceIdx(Common.getDeviceIdx());
        }

        Common.setDeviceIdx(Integer.parseInt(String.valueOf(request.getIdx())));
        Map<Integer, String> workers = request.getWorkersMap();
        // convert all integer key into string
        Map<String, String> workersStr = new HashMap<>();
        for (Map.Entry<Integer, String> entry : workers.entrySet()) {
            workersStr.put(String.valueOf(entry.getKey()), entry.getValue());
        }

        Common.printLog("Set as worker node with idx " + Common.getDeviceIdx());
        Common.setWorkers(workersStr);

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void setBasicInfo(SetBasicInfoRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();

        Map<String, Object> data = new HashMap<>();
        data.put("point", request.getPointList());
        data.put("modelName", request.getModelName());
        data.put("modelArgs", request.getModelArgsMap());
        data.put("aggrInterval", request.getAggrInterval());
        TrainWorker.setBasicInfoHandler(data);

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void measureBandwidth(EmptyRequest request, StreamObserver<BandwidthResponse> responseObserver) {
        int idx = Common.getDeviceIdx();
        String ipAddr = Common.getUrlFromWorker(idx + 1);
        float bandwidth = (float) General.measureNeighborBandwidth(ipAddr);

        BandwidthResponse bandwidthResponse = BandwidthResponse.newBuilder()
                .setBandwidth(bandwidth)
                .build();

        responseObserver.onNext(bandwidthResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void memoryInfo(EmptyRequest request, StreamObserver<MemoryResponse> responseObserver) {
        List<Float> mem = General.getCurrentMemoryInfo();

        MemoryResponse memoryResponse = MemoryResponse.newBuilder()
                .setAvailMem(mem.get(0))
                .setTotalMem(mem.get(1))
                .build();

        responseObserver.onNext(memoryResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void profileTransformerBlock(EmptyRequest request, StreamObserver<TransformerBlockResponse> responseObserver) {
        float transformerBlockTime = General.profileTransformerBlockHelper(1);

        TransformerBlockResponse transformerBlockResponse = TransformerBlockResponse.newBuilder()
                .setTime(transformerBlockTime)
                .build();

        responseObserver.onNext(transformerBlockResponse);
        responseObserver.onCompleted();
    }
}
