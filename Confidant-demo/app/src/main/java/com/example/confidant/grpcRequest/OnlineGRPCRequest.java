package com.example.confidant.grpcRequest;

import com.example.confidant.globalStates.Common;
import com.example.confidant.rpc.api.CommonResponse;
import com.example.confidant.rpc.api.HandleForwardRequest;
import com.example.confidant.rpc.api.LabelsRequest;
import com.example.confidant.rpc.api.SendTensorTestRequest;
import com.example.confidant.rpc.api.CentralGreeterGrpc;
import com.example.confidant.rpc.api.SendTrainBackwardRequest;
import com.example.confidant.rpc.api.SetBasicInfoRequest;
import com.example.confidant.rpc.api.StartEpochRequest;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.rpc.api.UnifiedIntTensor;
import com.example.confidant.rpc.api.UpdateWorkersRequest;
import com.example.confidant.rpc.api.WorkerGreeterGrpc;
import com.example.confidant.utils.GrpcUtils;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;

public class OnlineGRPCRequest {
    public static String sendTensorTest(String url, List<Float> data, List<Integer> dataShape) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);

            UnifiedFloatTensor tensor = UnifiedFloatTensor.newBuilder()
                    .addAllData(data)  // Adding float data
                    .setDataType(1)  // Setting dataType
                    .addAllDataShape(dataShape)  // Setting dataShape
                    .build();

            SendTensorTestRequest request = SendTensorTestRequest.newBuilder().setTensor(tensor).build();

            if (Common.getDeviceIdx() == 0) {
                WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);
                CommonResponse res = stub.sendTensorTest(request);
                return res.getStatus();
            } else {
                CentralGreeterGrpc.CentralGreeterBlockingStub stub = CentralGreeterGrpc.newBlockingStub(curChannel);
                CommonResponse res = stub.sendTensorTest(request);
                return res.getStatus();
            }
            // CommonResponse res = stub.sendTensorTest(request);

        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendLabels(String url, int iterId, Object[] data) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);

            List<UnifiedIntTensor> tensorList = GrpcUtils.convertObjectArrToUnifiedIntTensorList(data);

            LabelsRequest request = LabelsRequest.newBuilder().addAllLabels(tensorList).setIterId(iterId).build();
            CommonResponse res = stub.labels(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendTrainForward(String url, int iterId, int idx, int version, double lr, Object[] data) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            // WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);
            WorkerGreeterGrpc.WorkerGreeterStub stub = WorkerGreeterGrpc.newStub(curChannel);

            List<UnifiedFloatTensor> tensorList = GrpcUtils.convertObjectArrToUnifiedFloatTensorList(data);

            HandleForwardRequest request = HandleForwardRequest.newBuilder()
                                            .addAllData(tensorList)
                                            .setIterId(iterId)
                                            .setModelIdx(idx)
                                            .setVersion(version)
                                            .setLr(lr)
                                            .build();
            // Common.printLog(String.format("sendTrainForward of iterId %d with IP %s ", iterId, url));

            stub.handleForward(request, new StreamObserver<CommonResponse>() {
                @Override
                public void onNext(CommonResponse value) {
//                    Common.printLog("sendTrainForward receives res of iterId: " + iterId);
                }

                @Override
                public void onError(Throwable t) {
//                    Common.printLog("sendTrainForward receives error of iterId: " + iterId + " " + t.getMessage());
                }

                @Override
                public void onCompleted() {
//                    Common.printLog("sendTrainForward receives complete of iterId: " + iterId);
                }
            });

            // Common.printLog("sendTrainForward receives res of iterId: " + iterId);
            return "ok";
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendWorkerTrainBackward(String url, Object[] grad, int modelIdx, int iterId) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            // WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);
            WorkerGreeterGrpc.WorkerGreeterStub stub = WorkerGreeterGrpc.newStub(curChannel);

            List<UnifiedFloatTensor> tensorList = GrpcUtils.convertObjectArrToUnifiedFloatTensorList(grad);

            SendTrainBackwardRequest request = SendTrainBackwardRequest.newBuilder()
                                                .addAllData(tensorList)
                                                .setIterId(iterId)
                                                .setModelIdx(modelIdx)
                                                .build();
            Common.printLog(String.format("sendTrainBackward of iterId %d with IP %s ", iterId, url));
            stub.sendTrainBackward(request, new StreamObserver<CommonResponse>() {
                @Override
                public void onNext(CommonResponse value) {
                    Common.printLog("sendWorkerTrainBackward receives res of iterId: " + iterId);
                }

                @Override
                public void onError(Throwable t) {
                    Common.printLog("sendWorkerTrainBackward receives error of iterId: " + iterId + " " + t.getMessage());
                }

                @Override
                public void onCompleted() {
                    Common.printLog("sendWorkerTrainBackward receives complete of iterId: " + iterId);
                }
            });
            // Common.printLog("sendWorkerTrainBackward receives res of iterId: " + iterId);
            return "ok";
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendCentralTrainBackward(String url, Object[] grad, int modelIdx, int iterId) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            // CentralGreeterGrpc.CentralGreeterBlockingStub stub = CentralGreeterGrpc.newBlockingStub(curChannel);
            CentralGreeterGrpc.CentralGreeterStub stub = CentralGreeterGrpc.newStub(curChannel);
            List<UnifiedFloatTensor> tensorList = GrpcUtils.convertObjectArrToUnifiedFloatTensorList(grad);

            SendTrainBackwardRequest request = SendTrainBackwardRequest.newBuilder()
                    .addAllData(tensorList)
                    .setIterId(iterId)
                    .setModelIdx(modelIdx)
                    .build();
            Common.printLog(String.format("sendTrainBackward of iterId %d with IP %s ", iterId, url));
            stub.sendTrainBackward(request, new StreamObserver<CommonResponse>() {
                @Override
                public void onNext(CommonResponse value) {
                    Common.printLog("sendCentralTrainBackward receives res of iterId: " + iterId);
                }

                @Override
                public void onError(Throwable t) {
                    Common.printLog("sendCentralTrainBackward receives error of iterId: " + iterId + " " + t.getMessage());
                    Common.printLog(t.getMessage());
                }

                @Override
                public void onCompleted() {
                    Common.printLog("sendCentralTrainBackward receives complete of iterId: " + iterId);
                }
            });
            // Common.printLog("sendCentralTrainBackward receives res of iterId: " + iterId);
            return "ok";
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendStartEpoch(String url, int epoch, double lr, int dataLen) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);

            StartEpochRequest request = StartEpochRequest.newBuilder()
                    .setEpoch(epoch)
                    .setLr(lr)
                    .setLen(dataLen)
                    .build();

            CommonResponse res = stub.startEpoch(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendWorkers(String url, String idx, Map<String, String> workers) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);

            // convert string key into int key
            Map<Integer, String> workersMap = new HashMap<>();
            for (Map.Entry<String, String> entry : workers.entrySet()) {
                workersMap.put(Integer.parseInt(entry.getKey()), entry.getValue());
            }

            UpdateWorkersRequest request = UpdateWorkersRequest.newBuilder()
                                                .putAllWorkers(workersMap)
                                                .setIdx(Integer.parseInt(idx))
                                                .build();

            CommonResponse res = stub.updateWorkers(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendBasicInfo(String url, List<Integer> point, String modelName, Map<String, Double> modelArgs, int aggregateInterval) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);


            SetBasicInfoRequest request = SetBasicInfoRequest.newBuilder()
                    .addAllPoint(point)
                    .setModelName(modelName)
                    .putAllModelArgs(modelArgs)
                    .setAggrInterval(aggregateInterval)
                    .build();

            CommonResponse res = stub.setBasicInfo(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }
}
