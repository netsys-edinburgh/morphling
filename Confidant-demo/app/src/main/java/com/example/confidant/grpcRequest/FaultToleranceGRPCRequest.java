package com.example.confidant.grpcRequest;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.example.confidant.faultTolerance.DeviceCompatibilityInfo;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
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
import com.example.confidant.rpc.api.TensorResponse;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.rpc.api.UnifiedFloatTensorList;
import com.example.confidant.rpc.api.UpdateWorkersRequest;
import com.example.confidant.rpc.api.WeightsRedistributeRequest;
import com.example.confidant.rpc.api.WeightsReplicationRequest;
import com.example.confidant.rpc.api.WorkerExitRequest;
import com.example.confidant.rpc.api.WorkerGreeterGrpc;
import com.example.confidant.utils.GrpcUtils;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class FaultToleranceGRPCRequest {
    public static String sendRestartSyncState(String url, String idx, Map<String, String> workers, List<Integer> prevPoint) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            Map<Integer, String> workersMap = new HashMap<>();
            for (Map.Entry<String, String> entry : workers.entrySet()) {
                workersMap.put(Integer.parseInt(entry.getKey()), entry.getValue());
            }

            RestartSyncStateRequest request = RestartSyncStateRequest.newBuilder()
                    .setIdx(Integer.parseInt(idx))
                    .addAllPoint(prevPoint)
                    .putAllWorkers(workersMap)
                    .build();

            CommonResponse res = stub.restartSyncState(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendWeightsRedistribute(String url, List<String> failedIdx, List<Integer> points) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            WeightsRedistributeRequest request = WeightsRedistributeRequest.newBuilder()
                    .addAllFailedSet(failedIdx)
                    .addAllPoint(points)
                    .build();

            CommonResponse res = stub.weightsRedistribute(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String commitFaultSync(String url, List<Integer> partitionPoint, int iterId) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            CommitFaultSyncRequest request = CommitFaultSyncRequest.newBuilder()
                    .setIterId(iterId)
                    .addAllPoint(partitionPoint)
                    .build();

            CommonResponse res = stub.commitFaultSync(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String sendWeightsReplication(String url, Map<String, List<UnifiedFloatTensor>> weightsMap, FaultTolerance.ReplicationType replicationType) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            Map<String, UnifiedFloatTensorList> weights = new HashMap<>();
            for (Map.Entry<String, List<UnifiedFloatTensor>> entry : weightsMap.entrySet()) {
                List<UnifiedFloatTensor> curWeight = entry.getValue();
                UnifiedFloatTensorList tensorList = UnifiedFloatTensorList.newBuilder()
                        .addAllWeight(curWeight)
                        .build();

                weights.put(entry.getKey(), tensorList);
            }

            WeightsReplicationRequest request = WeightsReplicationRequest.newBuilder()
                    .setReplicationType(replicationType.ordinal())
                    .putAllWeights(weights)
                    .build();

            CommonResponse res = stub.weightsReplication(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String notifyPassiveCommitFinish(String url) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();

            CommonResponse res = stub.passiveCommitFinish(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static Map<String, Object[]> getParamsFromRemote(String url, List<Integer> layers) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            ParamsFromRemoteRequest request = ParamsFromRemoteRequest.newBuilder()
                    .addAllLayers(layers)
                    .build();

            TensorResponse res = stub.paramsFromRemote(request);

            Map<String, Object[]> ret = new HashMap<>();
            for (Map.Entry<String, UnifiedFloatTensorList> entry : res.getWeightsMap().entrySet()) {
                List<UnifiedFloatTensor> tensorList = entry.getValue().getWeightList();
                Object[] dataObjArr = GrpcUtils.convertUnifiedFloatTensorListToObjectArr(tensorList);
                ret.put(entry.getKey(), dataObjArr);
            }

            return ret;
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return new HashMap<>();
        }
    }


    public static String notifyRedistributionFinish(String url, int idx) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            PassiveRedistributionFinishRequest request = PassiveRedistributionFinishRequest.newBuilder()
                    .setIdx(idx)
                    .build();

            CommonResponse res = stub.passiveRedistributionFinish(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    public static String notifyWorkerExit(String url, int deviceIdx) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            WorkerExitRequest request = WorkerExitRequest.newBuilder().setDeviceIdx(deviceIdx).build();

            CommonResponse res = stub.workerExit(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    /*
        Get the device compatibility information (computing capacity vector and battery) of device
     */
    public static DeviceCompatibilityInfo getDeviceCompatibilityInfo(String url) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();

            DcInfoResponse res = stub.dCInfo(request);

            DeviceCompatibilityInfo dc = new DeviceCompatibilityInfo(res.getCcvList(), res.getBattery());
            return dc;
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return new DeviceCompatibilityInfo(new java.util.ArrayList<>(), -1);
        }
    }

    /*
        Ask the substitute device to fetch all weights from the quit device
     */
    public static String notifySubstituteDevice(String targetDeviceUrl, String quitUrl, String centralUrl, String quitIdx, List<Integer> curPoints) {
        try {
            ManagedChannel curChannel = Common.getChannel(targetDeviceUrl);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            NotifySubstituteDeviceRequest request = NotifySubstituteDeviceRequest.newBuilder()
                    .setQuitUrl(quitUrl)
                    .setCentralUrl(centralUrl)
                    .setQuitIdx(Integer.parseInt(quitIdx))
                    .addAllPoints(curPoints)
                    .build();

            CommonResponse res = stub.notifySubstituteDevice(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    /**
     * Fetch all weights from the quitting device
     */
    public static Map<String, Object[]> fetchAllWeightsFromQuittingDevice(String url) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();

            TensorResponse res = stub.weightsFromQuitDevice(request);

            Map<String, Object[]> ret = new HashMap<>();
            for (Map.Entry<String, UnifiedFloatTensorList> entry : res.getWeightsMap().entrySet()) {
                List<UnifiedFloatTensor> tensorList = entry.getValue().getWeightList();
                Object[] dataObjArr = GrpcUtils.convertUnifiedFloatTensorListToObjectArr(tensorList);
                ret.put(entry.getKey(), dataObjArr);
            }

            return ret;
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return new HashMap<>();
        }
    }

    /**
     * Notify the central node that the substitute device has fetched all weights from the quitting device
     */
    public static String notifyCentralFetchFinish(String targetDeviceUrl) {
        try {
            ManagedChannel curChannel = Common.getChannel(targetDeviceUrl);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();

            CommonResponse res = stub.notifyCentralFetchFinish(request);

            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return String.format("Failed... : %n%s", sw.toString());
        }
    }

    /**
     * Send the new workers to all worker nodes including startIterId
     */
    public static String proactiveFTSendWorkers(String url, String idx, Map<String, String> workers, int startIterId) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            FaultToleranceGreeterGrpc.FaultToleranceGreeterBlockingStub stub = FaultToleranceGreeterGrpc.newBlockingStub(curChannel);

            // convert string key into int key
            Map<Integer, String> workersMap = new HashMap<>();
            for (Map.Entry<String, String> entry : workers.entrySet()) {
                workersMap.put(Integer.parseInt(entry.getKey()), entry.getValue());
            }

            ProactiveFTUpdateWorkersRequest request = ProactiveFTUpdateWorkersRequest.newBuilder()
                    .putAllWorkers(workersMap)
                    .setIdx(Integer.parseInt(idx))
                    .setStartIterId(startIterId)
                    .build();

            CommonResponse res = stub.proactiveFTUpdateWorkers(request);

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
