package com.example.confidant.unitTest;

import com.example.confidant.globalStates.Common;
import com.example.confidant.grpcRequest.HelloworldRequest;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;

public class GrpcClientTest {
    private static String ipAddr = "192.168.8.100";
    private static int port = 6800;

    public static void GrpcClientTestEntry() {
        // sayHelloTest();
        sendUnifiedTensorTest();
    }

    public static void sayHelloTest() {

        Common.printLog(String.format("Sending request to %s with port %d ...", ipAddr, port));

        String res = HelloworldRequest.sayHello(ipAddr, port);
        Common.printLog("Reply from server: " + res);
    }

    public static void sendUnifiedTensorTest() {
        Common.printLog(String.format("Sending request to %s with port %d ...", ipAddr, port));

        // String res = OnlineGRPCRequest.sendTensorTest(ipAddr);
        // Common.printLog("Reply from server: " + res);
    }
}
