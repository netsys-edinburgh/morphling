package com.example.confidant.grpcAPI;

import com.example.confidant.globalStates.Common;

import java.io.IOException;
import java.util.concurrent.Executors;

import io.grpc.InsecureServerCredentials;
import io.grpc.Server;
import io.grpc.okhttp.OkHttpServerBuilder;

public class GrpcServer {
    private static Server grpcServer;
    private static TrainCentralAPI trainCentralAPI = new TrainCentralAPI();
    private static TrainWorkerAPI trainWorkerAPI = new TrainWorkerAPI();
    private static FaultToleranceAPI faultToleranceAPI = new FaultToleranceAPI();

    public static void createGrpcServer(int role, int port) {
        try {
            if (role == 0) {
                // central node
                Common.printLog(String.format("Initializing the GRPC Service as a central node on port %d", port));
                grpcServer = OkHttpServerBuilder
                        .forPort(port, InsecureServerCredentials.create())
                        .addService(trainCentralAPI)
                        .addService(faultToleranceAPI)
//                        .maxInboundMessageSize(2000 * 1024 * 1024)
                        .maxInboundMessageSize(2000 * 1024 * 1024)
//                        .maxInboundMessageSize(300 * 1024 * 1024)
                        .executor(Executors.newCachedThreadPool())
                        .build();
            } else if (role == 1) {
                // worker node
                Common.printLog(String.format("Initializing the GRPC Service as a worker node on port %d", port));
                grpcServer = OkHttpServerBuilder
                        .forPort(port, InsecureServerCredentials.create())
                        .addService(trainWorkerAPI)
                        .addService(faultToleranceAPI)
//                        .maxInboundMessageSize(300 * 1024 * 1024)
                        .maxInboundMessageSize(2000 * 1024 * 1024)
//                        .maxInboundMessageSize(2000 * 1024 * 1024)
                        .executor(Executors.newCachedThreadPool())
                        .build();
            }

            grpcServer.start();
            grpcServer.awaitTermination();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
