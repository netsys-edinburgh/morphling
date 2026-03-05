package com.example.confidant.unitTest;

import com.alibaba.fastjson.JSONObject;
import com.example.confidant.globalStates.Common;
import com.example.confidant.utils.TrainWorker;

import spark.Spark;

public class HttpServerTest {
    public static void HttpServerTestEntry() {
        int port = 50000;
        Spark.port(port);
        Common.printLog("Initializing the Spark Service listening on port 50000");

        Spark.post("/sendTensorTest", (req, res) -> {
            Common.printLog("Received a tensor from a client");
            JSONObject data = JSONObject.parseObject(req.body());
            return "ok";
        });
    }
}
