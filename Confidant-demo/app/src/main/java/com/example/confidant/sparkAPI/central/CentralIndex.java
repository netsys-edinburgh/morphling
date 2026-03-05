package com.example.confidant.sparkAPI.central;

import com.alibaba.fastjson.JSONObject;
import com.example.confidant.utils.TrainCentral;

import spark.Spark;

public class CentralIndex {
    public static void defineRoutes() {
        Spark.post("/sendTrainBackward", (req, res) -> {
            JSONObject data = JSONObject.parseObject(req.body());
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    TrainCentral.handleBackwardIntermediate(data);
                }
            }, "TrainCentral.sendTrainBackward");
            thread.start();
            return "ok";
        });
    }
}
