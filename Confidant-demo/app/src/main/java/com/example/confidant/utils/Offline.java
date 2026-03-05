package com.example.confidant.utils;

import android.util.Log;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.OfflineGRPCRequest;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;
import com.example.confidant.request.OfflineRequest;
import com.example.confidant.request.OnlineRequest;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class Offline {
    private static final String tag = "utils.Offline";

    /*
        Find the available workers nodes listed in the urls
     */
    public static int electWorkers() {
        Log.i(tag, "Start electing worker ...");
        Map<String, String> workers = Common.getWorkers();
        List<String> urls = Common.getUrls();

        // Check the availability of each worker
        // TODO: multi-threading
        int count = 0;
        for (int i = 0; i < urls.size(); i++) {
            // We use HTTP to check the availability of the devices for simplicity
            String res = OfflineRequest.checkAvailable(urls.get(i));
//            String res = OfflineGRPCRequest.checkAvailable(urls.get(i));
            if (res.equals("ok")) {
                workers.put(String.valueOf(count + 1), urls.get(i));
                count++;
            }
        }
        Log.i(tag, "Current election ends ...");

        Common.printLog("Current worker lists: ");
        for (Map.Entry<String, String> entry : workers.entrySet()) {
            Common.printLog(entry.getKey() + ": " + entry.getValue());
        }

        // Common.getWorkers() returns the reference, hence no need to update worker
        return workers.size();
    }

    /**
     * Send the available worker set to all worker nodes
     */
    public static void distributeWorkers() {
        Map<String, String> workers = Common.getWorkers();
        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            if (!idx.equals("0")) {
                String url = entry.getValue();
                // String res = OnlineRequest.sendWorkers(url, idx, workers);
                String res = OnlineGRPCRequest.sendWorkers(url, idx, workers);
                if (!res.equals("ok")) {
                    Log.e(tag, "Distribute workers to " + url + " fails");
                }
            }
        }
        Log.i(tag, "Distribute workers ends ...");
    }

    /**
     * Perform the offline profiling
     */
    public static void offlineProfiling(Map<String, Object> config) {
        Date startDate = new Date();
        long startTime = startDate.getTime();

        Common.printLog("Start offline profiling of model " + Common.getModelName() + "...");
        String modelName = Common.getModelName();
        Map<String, Double> modelArgs = Common.getModelArgs();

//        Model.initModel(modelName, modelArgs);
//
//        Training.setTotalLayers(modelArgs.get("total_layer").intValue());
//        OfflineProfiler profiler = new OfflineProfiler(Training.getTotalLayers(), Common.getWorkerNum());
//        Common.setOfflineProfiler(profiler);
//        profiler.getMemoryInfo();
//        profiler.profileBandwidth();
//        profiler.staticProfiling();
//        profiler.blockProfiling();
//        profiler.updateComputingCapacityByBlockProfiling();
//
//        List<Integer> partitionPoint = DynamicScheduler.calculatePartitionPoint(false);
//        Common.printLog("Initial Partition Point: " + partitionPoint);
//        List<Integer> partitionPoint = DynamicScheduler.calculatePartitionPointMemory(false, config);
        List<Integer> partitionPoint = new ArrayList<Integer>(){{
            add(5);
            add(10);

        }};
        Common.printLog("Actual Partition Point: " + partitionPoint);

        Training.setPartitionPoint(partitionPoint);

        Date endDate = new Date();
        long endTime = endDate.getTime();
        Common.printLog("Offline profiling ends, elapsed time: " + (endTime - startTime) + "ms");
    }

    /**
     * Send the basic information to the workers, including partition point, model name, model args...
     */
    public static void distributeBasicInfo() {
        Common.printLog("Sending partition point and basic info to workers ...");
        Map<String, String> workers = Common.getWorkers();
        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            if (!idx.equals("0")) {
                String url = entry.getValue();
                // String res = OnlineRequest.sendBasicInfo(url, Training.getPartitionPoint(), Common.getModelName(), Common.getModelArgs(), Training.getAggregateInterval());
                String res = OnlineGRPCRequest.sendBasicInfo(url, Training.getPartitionPoint(), Common.getModelName(), Common.getModelArgs(), Training.getAggregateInterval());
                if (!res.equals("ok")) {
                    Common.printLog("Send basic info to " + url + " fails");
                }
            }
        }
    }
}
