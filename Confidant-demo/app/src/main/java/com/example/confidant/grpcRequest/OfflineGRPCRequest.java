package com.example.confidant.grpcRequest;

import com.example.confidant.globalStates.Common;
import com.example.confidant.rpc.api.BandwidthResponse;
import com.example.confidant.rpc.api.CommonResponse;
import com.example.confidant.rpc.api.EmptyRequest;
import com.example.confidant.rpc.api.MemoryResponse;
import com.example.confidant.rpc.api.SendTrainBackwardRequest;
import com.example.confidant.rpc.api.TransformerBlockResponse;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.rpc.api.WorkerGreeterGrpc;
import com.example.confidant.utils.GrpcUtils;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

import io.grpc.ManagedChannel;

public class OfflineGRPCRequest {
    public static String checkAvailable(String url) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();
            CommonResponse res = stub.isAvailable(request);
            return res.getStatus();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return "Check available network fail";
        }
    }

    public static float sendMeasureBandwidth(String url) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();
            BandwidthResponse res = stub.measureBandwidth(request);
            return res.getBandwidth();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return -1.0f;
        }
    }

    public static List<Float> getMemoryInfo(String url) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();
            MemoryResponse res = stub.memoryInfo(request);
            List<Float> mem = new ArrayList<>();
            mem.add(res.getAvailMem());
            mem.add(res.getTotalMem());

            return mem;
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();

            List<Float> mem = new ArrayList<>();
            mem.add(-1.0f);
            mem.add(-1.0f);
            return mem;
        }
    }

    public static float sendProfileTransformerBlock(String url) {
        try {
            ManagedChannel curChannel = Common.getChannel(url);
            WorkerGreeterGrpc.WorkerGreeterBlockingStub stub = WorkerGreeterGrpc.newBlockingStub(curChannel);

            EmptyRequest request = EmptyRequest.newBuilder().build();
            TransformerBlockResponse res = stub.profileTransformerBlock(request);
            return res.getTime();
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.flush();
            return -1.0f;
        }
    }
}
