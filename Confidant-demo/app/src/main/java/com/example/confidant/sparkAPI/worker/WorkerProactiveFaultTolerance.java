package com.example.confidant.sparkAPI.worker;

import static android.content.Context.BATTERY_SERVICE;

import android.content.Context;
import android.os.BatteryManager;

import com.alibaba.fastjson.JSONObject;

import java.util.List;

import spark.Spark;

public class WorkerProactiveFaultTolerance {
    private final static String routePrefix = "/pft";

    public static void defineRoutes() {
//        Spark.get(routePrefix + "/get_computing_capacity", (req, res) -> {
//            List<Float> computingCapacity = Common.getCurrentDeviceCapacity();
//            if (computingCapacity.size() == 0) {
//                computingCapacity = DynamicScheduler.profileComputingCapacityVector();
//            }
//
//            // get battery level
//            Context context = Common.getContext();
//            BatteryManager bm = (BatteryManager) context.getSystemService(BATTERY_SERVICE);
//            Integer chargeCounter = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
//
//            return computingCapacity.toString() + "#" + chargeCounter.toString();
//        });
//
//        Spark.post(routePrefix + "/fetch_weights_all_from_quit_device", (req, res) -> {
//            JSONObject data = JSONObject.parseObject(req.body());
//            String quitUrl = data.getString("quit_url");
//            String quitIdx = data.getString("quit_idx");
//            List<Integer> points = data.getJSONArray("points").toJavaList(Integer.class);
//            Thread thread = new Thread(new Runnable() {
//                @Override
//                public void run() {
//                    ProactiveFTRequestHandler.fetchWeightsAllFromQuitDeviceHandler(quitUrl, quitIdx, points);
//                }
//            }, "WorkerFaultTolerance.backupLocalWeight");
//            thread.start();
//
//            return "ok";
//        });
    }
}
