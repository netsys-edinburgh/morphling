package com.example.confidant.request;

import android.util.Log;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.android.volley.Request;
import com.android.volley.toolbox.RequestFuture;
import com.android.volley.toolbox.StringRequest;
import com.example.confidant.globalStates.Common;
import com.example.confidant.utils.General;

import java.io.IOException;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class OnlineRequest {
    private static final String tag = "OnlineRequest";

    public static String sendWorkers(String url, String idx, Map<String, String> workers) {
        String targetUrl = url + "/updateWorkers";

        JSONObject payload = new JSONObject();
        payload.put("idx", idx);
        payload.put("workers", workers);

        String payloadStr = JSON.toJSONString(payload);

        RequestFuture future = RequestFuture.newFuture();
        StringRequest stringRequest = new StringRequest(Request.Method.POST, targetUrl, future, future) {
            @Override
            public byte[] getBody() {
                return payloadStr.getBytes();
            }

            @Override
            public String getBodyContentType() {
                return "text/plain; charset=utf-8";
            }
        };
        RequestHelper.mRequestQueue.add(stringRequest);
        try {
            String res = (String) future.get(10, TimeUnit.SECONDS);
            return res;
        } catch (ExecutionException | InterruptedException e) {
            Log.e(tag, e.getMessage());
            return e.getMessage();
        } catch (TimeoutException e) {
            Log.i(tag, "updateWorkers timeout");
            Log.i(tag, e.getMessage());
            return "updateWorkers timeout";
        }
    }

    public static String sendBasicInfo(String url, List<Integer> point, String modelName, Map<String, Double> modelArgs, int aggregateInterval) {
        String targetUrl = url + "/setBasicInfo";

        JSONObject payload = new JSONObject();
        payload.put("point", point);
        payload.put("modelName", modelName);
        payload.put("modelArgs", modelArgs);
        payload.put("aggrInterval", aggregateInterval);

        String payloadStr = JSON.toJSONString(payload);
        RequestFuture future = RequestFuture.newFuture();
        StringRequest stringRequest = new StringRequest(Request.Method.POST, targetUrl, future, future) {
            @Override
            public byte[] getBody() {
                return payloadStr.getBytes();
            }

            @Override
            public String getBodyContentType() {
                return "text/plain; charset=utf-8";
            }
        };
        RequestHelper.mRequestQueue.add(stringRequest);
        try {
            String res = (String) future.get(20, TimeUnit.SECONDS);
            return res;
        } catch (ExecutionException | InterruptedException e) {
            Log.e(tag, e.getMessage());
            // throw new RuntimeException(e);
            return e.getMessage();
        } catch (TimeoutException e) {
            Log.i(tag, "updateWorkers timeout");
            Log.i(tag, e.getMessage());
            return "updateWorkers timeout";
        }
    }

    public static String sendStartEpoch(String url, int epoch, double lr, int dataLen) {
        String targetUrl = url + "/startEpoch";

        JSONObject payload = new JSONObject();
        payload.put("epoch", epoch);
        payload.put("lr", lr);
        payload.put("len", dataLen);

        String payloadStr = JSON.toJSONString(payload);

        return RequestHelper.stringPostRequestHelper(payloadStr, targetUrl, "sendStartEpoch");
    }

    public static String sendTrainForward(String url, int iterId, int idx, int version, double lr, Object[] data) {
        String targetUrl = url + "/handleForward";

        byte[] dataArrBytes = new byte[0];
        try {
            dataArrBytes = General.convertToByteArray(data);

        } catch (IOException e) {
            // throw new RuntimeException(e);
            Common.printLog("OnlineRequest.sendTrainForward: " + e.getMessage());
            return e.getMessage();
        }
        String dataStr = Base64.getEncoder().encodeToString(dataArrBytes);
        JSONObject payload = new JSONObject();
        payload.put("iterId", iterId);
        payload.put("data", dataStr);
        payload.put("modelIdx", idx);
        payload.put("version", version);
        payload.put("framework", "MNN");

        //TODO: payload.put("term", term);

        if (lr != 0.0) {
            payload.put("lr", lr);
        }

        String payloadStr = JSON.toJSONString(payload);
        return RequestHelper.stringPostRequestHelper(payloadStr, targetUrl, "sendTrainForward");
    }

    public static String sendLabels(String url, int iterId, Object[] data) {
        String targetUrl = url + "/labels";

        byte[] dataArrBytes = new byte[0];
        try {
            dataArrBytes = General.convertToByteArray(data);

        } catch (IOException e) {
            // throw new RuntimeException(e);
            Common.printLog("OnlineRequest.sendTrainForward: " + e.getMessage());
            return e.getMessage();
        }
        String dataStr = Base64.getEncoder().encodeToString(dataArrBytes);

        JSONObject payload = new JSONObject();
        payload.put("iterId", iterId);
        payload.put("data", dataStr);
        payload.put("framework", "MNN");

        String payloadStr = JSON.toJSONString(payload);
        return RequestHelper.stringPostRequestHelper(payloadStr, targetUrl, "sendLabels");
    }

    public static String sendTrainBackward(String url, Object[] grad, int modelIdx, int iterId) {
        String targetUrl = url + "/sendTrainBackward";

        byte[] dataArrBytes = new byte[0];
        try {
            dataArrBytes = General.convertToByteArray(grad);

        } catch (IOException e) {
            // throw new RuntimeException(e);
            Common.printLog("OnlineRequest.sendTrainForward: " + e.getMessage());
            return e.getMessage();
        }
        String gradStr = Base64.getEncoder().encodeToString(dataArrBytes);

        JSONObject payload = new JSONObject();
        payload.put("data", gradStr);

        payload.put("modelIdx", modelIdx);
        payload.put("iterId", iterId);

        String payloadStr = JSON.toJSONString(payload);
        return RequestHelper.stringPostRequestHelper(payloadStr, targetUrl, "sendTrainBackward");
    }
}
