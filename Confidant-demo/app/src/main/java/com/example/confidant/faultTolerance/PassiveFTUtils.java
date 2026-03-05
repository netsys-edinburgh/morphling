package com.example.confidant.faultTolerance;

import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.grpcRequest.OfflineGRPCRequest;
import com.example.confidant.request.OfflineRequest;

import java.util.List;
import java.util.Map;

public class PassiveFTUtils {
    private static final String tag = "PassiveFTUtils";

    public static Pair<List<String>, List<String>> findFailedDevice() {
        Common.printLog("Finding failed device ...");
        Map<String, String> workers = Common.getWorkers();

        List<String> failedIdx = new java.util.ArrayList<>();
        List<String> restartIdx = new java.util.ArrayList<>();

        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            if (!idx.equals("0")) {
                String url = entry.getValue();
                String res = OfflineRequest.checkAvailable(url);
                if (res.equals("no") || res.equals("Check available network fail")) {
                    failedIdx.add(idx);
                } else if (!res.equals("Occupied")) {
                    restartIdx.add(idx);
                }
            }
        }

        return new Pair<>(failedIdx, restartIdx);
    }

    public static void updateWorkersByFailedIdx(List<String> failedIdx) {
        Map<String, String> workers = Common.getWorkers();
        Map<String, String> newWorkers = new java.util.HashMap<>();
        newWorkers.put("0", workers.get("0"));

        if (failedIdx.size() == 1) {
            for (Map.Entry<String, String> entry : workers.entrySet()) {
                String idx = entry.getKey();
                String url = entry.getValue();
                if (!idx.equals("0")) {
                    if (Integer.parseInt(idx) > Integer.parseInt(failedIdx.get(0))) {
                        newWorkers.put(String.valueOf(Integer.parseInt(idx) - 1), url);
                    } else if (Integer.parseInt(idx) < Integer.parseInt(failedIdx.get(0))) {
                        newWorkers.put(idx, url);
                    }
                }
            }
        } else {
            int cnt = 1;
            for (Map.Entry<String, String> entry : workers.entrySet()) {
                String idx = entry.getKey();
                String url = entry.getValue();
                if (!idx.equals("0")) {
                    if (!failedIdx.contains(idx)) {
                        newWorkers.put(String.valueOf(cnt), url);
                        cnt += 1;
                    }
                }
            }
        }

        Common.updateWorkers(newWorkers);
        // print the new workers
        Common.printLog("New workers: ");
        for (Map.Entry<String, String> entry : newWorkers.entrySet()) {
            Common.printLog(entry.getKey() + ": " + entry.getValue());
        }
    }

    /*
        Load the backup parameters after partitioning new sub model
    */
    public static void loadBackupParams(Map<String, Object[]> params) {
        for (Map.Entry<String, Object[]> entry : params.entrySet()) {
            String layer = entry.getKey();
            Object[] weights = entry.getValue();
            ReplicationUtils.loadSubModelWeightsByLayer(Integer.parseInt(layer), weights, true);
        }
    }
}
