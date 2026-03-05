package com.example.confidant.grpcAPI;

import com.example.confidant.globalStates.Common;
import com.example.confidant.rpc.api.CommonResponse;
import com.example.confidant.rpc.api.HandleForwardRequest;
import com.example.confidant.rpc.api.SendTensorTestRequest;
import com.example.confidant.rpc.api.CentralGreeterGrpc;
import com.example.confidant.rpc.api.SendTrainBackwardRequest;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.utils.GrpcUtils;
import com.example.confidant.utils.TrainCentral;
import com.example.confidant.utils.TrainWorker;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.grpc.stub.StreamObserver;

public class TrainCentralAPI extends CentralGreeterGrpc.CentralGreeterImplBase {
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
    public void sendTrainBackward(SendTrainBackwardRequest request, StreamObserver<CommonResponse> responseObserver) {
        CommonResponse commonResponse = CommonResponse.newBuilder()
                .setStatus("ok")
                .build();
        Common.printLog("Received sendTrainBackward request: " + request.getIterId());
//        List<UnifiedFloatTensor> reqData = request.getDataList();
//        Object[] dataObjArr = GrpcUtils.convertUnifiedFloatTensorListToObjectArr(reqData);
//        Map<String, Object> data = new HashMap<>();
//        data.put("iterId", request.getIterId());
//        data.put("data", dataObjArr);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                List<UnifiedFloatTensor> reqData = request.getDataList();
                Object[] dataObjArr = GrpcUtils.convertUnifiedFloatTensorListToObjectArr(reqData);
                Map<String, Object> data = new HashMap<>();
                data.put("iterId", request.getIterId());
                data.put("data", dataObjArr);
                TrainCentral.handleBackwardIntermediate(data);
            }
        }, "TrainCentral.sendTrainBackward");
        thread.start();

        responseObserver.onNext(commonResponse);
        responseObserver.onCompleted();
    }
}
