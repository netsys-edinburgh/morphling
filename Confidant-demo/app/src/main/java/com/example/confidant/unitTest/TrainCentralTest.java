package com.example.confidant.unitTest;

import com.example.confidant.globalStates.Common;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;
import com.example.confidant.request.OnlineRequest;
import com.example.confidant.utils.TrainCentral;

public class TrainCentralTest {
    public static void TrainCentralTestEntry() {
        // GRPCSendLabelTest();
        // HTTPSendLabelTest();
        GRPCSendIntermediateTest();
    }

    public static void GRPCSendIntermediateTest() {
        Common.printLog("GRPCSendIntermediateTest starts ...");

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                Object[] output = TrainCentral.forwardOneBatch(0);
                String nextUrl = Common.getUrlFromWorker(Common.getDeviceIdx() + 1);
                Common.printLog("Sending data to " + nextUrl);
                String res = OnlineGRPCRequest.sendTrainForward(nextUrl, 0, 1, 0, 0.0, (Object[]) output[0]);
                Common.printLog("Received msg " + res);
            }
        }, "TrainCentralTest.GRPCSendIntermediateTest");
        thread.start();

        Common.printLog("GRPCSendIntermediateTest finished");
    }

    public static void GRPCSendLabelTest() {
        Common.printLog("GRPCSendLabelTest starts ...");

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                Object[] output = TrainCentral.forwardOneBatch(0);
                String nextUrl = Common.getUrlFromWorker(Common.getDeviceIdx() + 1);
                Common.printLog("Sending data to " + nextUrl);
                String res = OnlineGRPCRequest.sendLabels(nextUrl, 0, (Object[]) output[1]);
                Common.printLog("Received msg " + res);
            }
        }, "TrainCentralTest.GRPCSendLabelTest");
        thread.start();

        Common.printLog("GRPCSendLabelTest finished");
    }

    public static void HTTPSendLabelTest() {
        Common.printLog("HTTPSendLabelTest starts ...");

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                Object[] output = TrainCentral.forwardOneBatch(0);
                String nextUrl = Common.getUrlFromWorker(Common.getDeviceIdx() + 1);
                Common.printLog("Sending data to " + nextUrl);
                String res = OnlineRequest.sendLabels(nextUrl, 0, (Object[]) output[1]);
                Common.printLog("Received msg " + res);
            }
        }, "TrainCentralTest.HTTPSendLabelTest");
        thread.start();

        Common.printLog("HTTPSendLabelTest finished");
    }
}
