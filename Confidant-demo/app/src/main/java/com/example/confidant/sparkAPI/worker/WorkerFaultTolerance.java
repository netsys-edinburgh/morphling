package com.example.confidant.sparkAPI.worker;

import com.alibaba.fastjson.JSONObject;
//import com.example.ftpipehd_mnn.faultTolerance.Redistribution;
//import com.example.ftpipehd_mnn.globalStates.Common;
//import com.example.ftpipehd_mnn.globalStates.FaultTolerance;
//import com.example.ftpipehd_mnn.utils.General;

import java.io.IOException;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import spark.Spark;

public class WorkerFaultTolerance {
    private final static String routePrefix = "/ft";

    public static void defineRoutes() {
//        Spark.post(routePrefix + "/backup_weight", (req, res) -> {
//            JSONObject data = JSONObject.parseObject(req.body());
//            Thread thread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    byte[] weightsMap = data.getBytes("weight");
//                    try {
//                        Map<String, Object[]> weights = General.deserializeMap(weightsMap);
//                        // iterate over the map
//                        for (Map.Entry<String, Object[]> entry : weights.entrySet()) {
//                            String key = entry.getKey();
//                            Object[] value = entry.getValue();
//                            FaultTolerance.addLocalWeightsMap(key, value);
//                        }
//                        Common.printLog("Backup local weights finished");
//                    } catch (IOException e) {
//                        throw new RuntimeException(e);
//                    } catch (ClassNotFoundException e) {
//                        throw new RuntimeException(e);
//                    }
//                }
//            }, "WorkerFaultTolerance.backupLocalWeight");
//            thread.start();
//
//            return "ok";
//        });
//
//        Spark.get(routePrefix + "/fetch_desired_weight", (req, res) -> {
//            String layersStr = req.queryParams("layers");
//            Map<String, Object[]> ret = new HashMap<>();
//
//            String[] layersStrArr = layersStr.substring(1, layersStr.length() - 1).split(", ");
//            Common.printLog("Fetching weights of layer: " + layersStr);
//
//            for (int i = 0; i < layersStrArr.length; i++) {
//                ret.put(layersStrArr[i], FaultTolerance.getLocalWeightsByLayer(layersStrArr[i]));
//            }
//
//            byte[] serializedData = new byte[0];
//            try {
//                serializedData = General.serializeMap(ret);
//            } catch (IOException e) {
//                throw new RuntimeException(e);
//            }
//
//            Long startTime = System.currentTimeMillis();
//            String encodedString = Base64.getEncoder().encodeToString(serializedData);
//            Long endTime = System.currentTimeMillis();
//            Common.printLog("fetch_desired_weight: Base64 encoding time: " + (endTime - startTime) + "ms");
//
//            return encodedString;
//        });
//
//        Spark.post(routePrefix + "/weight_redistribute", (req, res) -> {
//            JSONObject data = JSONObject.parseObject(req.body());
//            List<String> failedIdx = data.getJSONArray("failed_set").toJavaList(String.class);
//            List<Integer> points = data.getJSONArray("points").toJavaList(Integer.class);
//
//            Thread thread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    Redistribution.weightRedistributeWorkerHandler(failedIdx, points);
//                }
//            }, "WorkerFaultTolerance.weightRedistributeWorkerHandler");
//
//            thread.start();
//
//            return "ok";
//        });
//
//        Spark.post(routePrefix + "/commit_fault_sync", (req, res) -> {
//            JSONObject data = JSONObject.parseObject(req.body());
//            List<Integer> partitionPoint = data.getJSONArray("partition_point").toJavaList(Integer.class);
//            int iterId = data.getInteger("iter_id");
//
//            Thread thread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    Redistribution.commitFaultSyncHandler(partitionPoint, iterId);
//                }
//            }, "WorkerFaultTolerance.commitWorker");
//
//            thread.start();
//
//            return "ok";
//        });
    }
}
