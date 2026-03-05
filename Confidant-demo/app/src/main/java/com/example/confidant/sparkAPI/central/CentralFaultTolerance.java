package com.example.confidant.sparkAPI.central;

import com.alibaba.fastjson.JSONObject;
//import com.example.confidant.faultTolerance.ProactiveFTHandler;
//import com.example.ftpipehd_mnn.globalStates.FaultTolerance;
//import com.example.ftpipehd_mnn.utils.General;

import java.io.IOException;
import java.util.Map;

import spark.Spark;

public class CentralFaultTolerance {
    private final static String routePrefix = "/ft";
    public static void defineRoutes() {
//        Spark.get(routePrefix + "/redistribution_finish", (req, res) -> {
//            String idx = req.queryParams("idx");
//            FaultTolerance.addRedistributedDevice(idx);
//            return "ok";
//        });
//
//        Spark.post(routePrefix + "/backup_weight", (req, res) -> {
//            Thread thread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    JSONObject data = JSONObject.parseObject(req.body());
//                    byte[] weightsMap = data.getBytes("weight");
//                    try {
//                        Map<String, Object[]> weights = General.deserializeMap(weightsMap);
//                        // iterate over the map
//                        for (Map.Entry<String, Object[]> entry : weights.entrySet()) {
//                            String key = entry.getKey();
//                            Object[] value = entry.getValue();
//                            FaultTolerance.addLocalWeightsMap(key, value);
//                        }
//                    } catch (IOException e) {
//                        throw new RuntimeException(e);
//                    } catch (ClassNotFoundException e) {
//                        throw new RuntimeException(e);
//                    }
//                }
//            }, "CentralFaultTolerance.backupLocalWeight");
//            thread.start();
//
//            return "ok";
//        });
//
//        Spark.post(routePrefix + "/global_backup_weight", (req, res) -> {
//            Thread thread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    JSONObject data = JSONObject.parseObject(req.body());
//                    byte[] weightsMap = data.getBytes("weight");
//                    try {
//                        Map<String, Object[]> weights = General.deserializeMap(weightsMap);
//                        // iterate over the map
//                        for (Map.Entry<String, Object[]> entry : weights.entrySet()) {
//                            String key = entry.getKey();
//                            Object[] value = entry.getValue();
//                            FaultTolerance.addGlobalWeightsMap(key, value);
//                        }
//                    } catch (IOException e) {
//                        throw new RuntimeException(e);
//                    } catch (ClassNotFoundException e) {
//                        throw new RuntimeException(e);
//                    }
//                }
//            }, "CentralFaultTolerance.backupGlobalWeight");
//            thread.start();
//
//            return "ok";
//        });
//
//        Spark.get(routePrefix + "/fetch_desired_weight", (req, res) -> {
//            String layers = req.queryParams("layers");
//            return "ok";
//        });
//
//        Spark.get(routePrefix + "/notify_exit", (req, res) -> {
//            String quitIdx = req.queryParams("idx");
//            int iterId = Integer.parseInt(req.queryParams("iter_id"));
//            Thread thread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    ProactiveFTHandler.handlerHelper(quitIdx, iterId);
//                }
//            }, "CentralFaultTolerance.nofityExit");
//            thread.start();
//            return "ok";
//        });
    }
}
