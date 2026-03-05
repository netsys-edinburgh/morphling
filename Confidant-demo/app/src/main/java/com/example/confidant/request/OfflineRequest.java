package com.example.confidant.request;

import android.util.Log;

import com.android.volley.Request;
import com.android.volley.toolbox.RequestFuture;
import com.android.volley.toolbox.StringRequest;
import com.example.confidant.globalStates.Common;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class OfflineRequest {
    private static final String tag = "OfflineRequest";

    /**
     * Check whether the device with this url is available
     * @param url
     */
    public static String checkAvailable(String url) {
        String targetUrl = String.format("http://%s:%d/isAvailable", url, Common.getHttpPort());

        RequestFuture future = RequestFuture.newFuture();
        StringRequest stringRequest = new StringRequest(Request.Method.GET, targetUrl, future, future);

        RequestHelper.mRequestQueue.add(stringRequest);
        try {
            String res = (String) future.get(10, TimeUnit.SECONDS);
            return res;
        } catch (ExecutionException e) {
            Log.e(tag, e.getMessage());
            return "Check available network fail";
        } catch (InterruptedException e) {
            Log.e(tag, e.getMessage());
            return "Check available network fail";
        } catch (TimeoutException e) {
            Log.i(tag, "checkAvailable timeout");
            Log.i(tag, e.getMessage());
            return "Check available network fail";
        }
    }

    public static float sendMeasureBandwidth(String url) {
        String targetUrl = url + "/measureBandwidth";

        RequestFuture future = RequestFuture.newFuture();
        StringRequest stringRequest = new StringRequest(Request.Method.GET, targetUrl, future, future);

        RequestHelper.mRequestQueue.add(stringRequest);
        try {
            String res = (String) future.get(30, TimeUnit.SECONDS);
            return Float.parseFloat(res);
        } catch (ExecutionException e) {
            Log.e(tag, e.getMessage());
            return -1.0f;
        } catch (InterruptedException e) {
            Log.e(tag, e.getMessage());
            return -1.0f;
        } catch (TimeoutException e) {
            Log.i(tag, "measureBandwidth timeout");
            Log.i(tag, e.getMessage());
            return -1.0f;
        }
    }
}
