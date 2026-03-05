package com.example.confidant.unitTest;

import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.utils.DynamicScheduler;
import com.example.confidant.utils.Model;
import com.example.confidant.utils.General;
import com.example.confidant.utils.OfflineProfiler;

import java.util.List;
import java.util.Map;

public class OfflineProfilerTest {
    static {
        System.loadLibrary("confidant");
    }

    public static void offlineProfilerTestEntry() {
        Common.printLog("Offline Profiler Test Entry");

        String modelName = Common.getModelName();
        Map<String, Double> modelArgs = Common.getModelArgs();
        Model.initModel(modelName, modelArgs);
        Training.setTotalLayers(modelArgs.get("total_layer").intValue());

        // computeTimeProfilingTest();
        // outputDataSizeProfilingTest();
        calculatePartitionPointTest();
        // pingTest();
    }

    public static void computeTimeProfilingTest() {
        int totalLayer = Training.getTotalLayers();
        int deviceNum = 3;
        Pair<List<Float>, List<Float>> computeTime = Model.profileModelTimeHelper();
        for (int i = 0; i < totalLayer; i++) {
            Common.printLog(String.format("Forward time of layer %d: %f", i, computeTime.first.get(i)));
        }
        Common.printLog("computeTimeProfilingTest finished!");
        return ;
    }

    public static void outputDataSizeProfilingTest() {
        List<Float> dataSize = Model.profileDataSizeHelper();
        for (int i = 0; i < dataSize.size(); i++) {
            Common.printLog(String.format("Data size of layer %d: %f", i, dataSize.get(i)));
        }
        Common.printLog("outputDataSizeProfilingTest finished!");
    }

    public static void pingTest() {
        String ipAddr = "192.168.8.100";
        General.PingThread pingThread = new General.PingThread(ipAddr);
        pingThread.start();

        try {
            pingThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        double bandwidth = General.calculateBandwidth(Common.getPingResult());
        Common.printLog(String.format("Ping test finished, ping result %f", bandwidth));
    }

    public static void calculatePartitionPointTest() {
        OfflineProfiler profiler = new OfflineProfiler(Training.getTotalLayers(), Common.getWorkerNum());
        Common.setOfflineProfiler(profiler);
        profiler.profileBandwidth();
        profiler.staticProfiling();

        List<Integer> partitionPoint = DynamicScheduler.calculatePartitionPoint(true);
        Common.printLog("Partition point: " + partitionPoint);
    }
}
