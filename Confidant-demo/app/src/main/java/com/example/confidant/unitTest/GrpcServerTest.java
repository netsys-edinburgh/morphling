package com.example.confidant.unitTest;

import com.example.confidant.globalStates.Common;
import com.example.confidant.grpcAPI.GrpcServer;

import java.io.IOException;

public class GrpcServerTest {
    public static void GrpcServerTestEntry(int role) {
        int port = 50000;
        GrpcServer.createGrpcServer(role, port);
    }
}
