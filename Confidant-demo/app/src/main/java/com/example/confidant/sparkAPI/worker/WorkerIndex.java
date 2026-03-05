package com.example.confidant.sparkAPI.worker;

import com.alibaba.fastjson.JSONObject;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.utils.General;
import com.example.confidant.utils.TrainWorker;

import java.io.IOException;
import java.util.Base64;
import java.util.Map;

import spark.Spark;

public class WorkerIndex {
    private static final String tag = "WorkerIndex";
    public static void defineRoutes() {
        Spark.get("/isAvailable", (req, res) -> {
            if (Common.getDeviceIdx() != -1) {
                return "Occupied";
            }
            Common.printLog("Found by a central node");
            return "ok";
        });

        Spark.post("/updateWorkers", (req, res) -> {
            JSONObject data = JSONObject.parseObject(req.body());
            Common.setDeviceIdx(Integer.parseInt(data.getString("idx")));
            JSONObject mapObject = data.getJSONObject("workers");
            Map<String, String> workers = mapObject.toJavaObject(Map.class);

            Common.printLog("Set as worker node with idx " + Common.getDeviceIdx());
            Common.setWorkers(workers);
            return "ok";
        });

        Spark.get("/measureBandwidth", (req, res) -> {
            int idx = Common.getDeviceIdx();
            String ipAddr = Common.getUrlFromWorker(idx + 1);
            double bandwidth = General.measureNeighborBandwidth(ipAddr);
            return String.valueOf(bandwidth);
        });

        Spark.post("/setBasicInfo", (req, res) -> {
            JSONObject data = JSONObject.parseObject(req.body());
            TrainWorker.setBasicInfoHandler(data);
            return "ok";
        });

        Spark.post("/startEpoch", (req, res) -> {
            JSONObject data = JSONObject.parseObject(req.body());
            int epoch = data.getIntValue("epoch");
            double lr = data.getDouble("lr");
            int dataLen = data.getIntValue("len");
            TrainWorker.initEpoch(epoch, lr, dataLen);
            return "ok";
        });

        Spark.post("/handleForward", (req, res) -> {
            JSONObject data = JSONObject.parseObject(req.body());
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    TrainWorker.handleForward(data);
                }
            }, "TrainWorker.handleForward");
            thread.start();
            return "ok";
        });

        Spark.post("/sendTrainBackward", (req, res) -> {
            JSONObject data = JSONObject.parseObject(req.body());

            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    TrainWorker.handleBackward(data);
                }
            }, "TrainWorker.sendTrainBackward");
            thread.start();
            return "ok";
        });

        Spark.post("/labels", (req, res) -> {
            JSONObject data = JSONObject.parseObject(req.body());
            int iterId = data.getIntValue("iterId");

            // base64 impl
            String labelDataArr = data.getString("data");
            byte[] labelDataBytes = Base64.getDecoder().decode(labelDataArr);
            Object[] labels = new Object[0];
            try {
                labels = (Object[]) General.convertToObjectArray(labelDataBytes);
            } catch (IOException e) {
                // throw new RuntimeException(e);
                Common.printLog("IOException in convertToObjectArray: " + e.getMessage());
            } catch (ClassNotFoundException e) {
                // throw new RuntimeException(e);
                Common.printLog("ClassNotFoundException in convertToObjectArray: " + e.getMessage());
            }
            Training.updateLabelsPool(iterId, labels);

            return "ok";
        });

    }
}
