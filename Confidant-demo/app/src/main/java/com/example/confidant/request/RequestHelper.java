package com.example.confidant.request;

import android.content.Context;
import android.util.Log;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.RequestFuture;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.confidant.globalStates.Common;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class RequestHelper {
    // Use Volley to send HTTP requests
    public static RequestQueue mRequestQueue;
    public static final String tag = "RequestHelper";
    public static void initRequestHelper(Context context) {
        mRequestQueue = Volley.newRequestQueue(context);
    }

    public static String stringGetRequestHelper(String targetUrl, String functionName, int timeout) {
        RequestFuture future = RequestFuture.newFuture();
        StringRequest stringRequest = new StringRequest(Request.Method.GET, targetUrl, future, future);
        RequestHelper.mRequestQueue.add(stringRequest);
        try {
            String res = (String) future.get(timeout, TimeUnit.SECONDS);
            return res;
        } catch (ExecutionException | InterruptedException e) {
            Log.e(tag, e.getMessage());
            // throw new RuntimeException(e);
            return e.getMessage();
        } catch (TimeoutException e) {
            Log.i(tag, functionName + " timeout");
            Log.i(tag, e.getMessage());
            return functionName + " timeout";
        }
    }

    /*
        * Send a GET request to the target URL and return the response as a byte array
     */
    public static byte[] byteArrGetRequestHelper(String targetUrl, String functionName) {
        RequestFuture future = RequestFuture.newFuture();
        StringRequest stringRequest = new StringRequest(Request.Method.GET, targetUrl, future, future);
        RequestHelper.mRequestQueue.add(stringRequest);
        try {
            byte[] res = (byte[]) future.get(20, TimeUnit.SECONDS);
            return res;
        } catch (ExecutionException | InterruptedException e) {
            Log.e(tag, e.getMessage());
            Common.printLog("RequestHelper.byteArrGetRequestHelper: " + e.getMessage());
            // throw new RuntimeException(e);
            return new byte[0];
        } catch (TimeoutException e) {
            Log.i(tag, functionName + " timeout");
            Log.i(tag, e.getMessage());
            Common.printLog("RequestHelper.byteArrGetRequestHelper: " + e.getMessage());
            return new byte[0];
        }
    }

    /*
        * Send a POST request to the target URL and return the response as a string
     */
    public static String stringPostRequestHelper(String payloadStr, String targetUrl, String functionName) {
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

        stringRequest.setRetryPolicy(new DefaultRetryPolicy(
                20000,
                DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));

        RequestHelper.mRequestQueue.add(stringRequest);
        try {
            String res = (String) future.get(30, TimeUnit.SECONDS);
            return res;
        } catch (ExecutionException | InterruptedException e) {
            Log.e(tag, e.getMessage());
            // throw new RuntimeException(e);
            return e.getMessage();
        } catch (TimeoutException e) {
            Log.i(tag, functionName + " timeout");
            Log.i(tag, e.getMessage());
            return functionName + " timeout";
        } catch (Exception e) {
            Log.e(tag, e.getMessage());
            return e.getMessage();
        }
    }
}
