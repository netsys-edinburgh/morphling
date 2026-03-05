package com.example.confidant.faultTolerance;

import android.util.Log;
import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.FaultToleranceGRPCRequest;
import com.example.confidant.utils.General;
import com.example.confidant.utils.Model;
import com.example.confidant.utils.Optimizer;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class RedistributionUtils {
    private static final String tag = "RedistributionUtils";

    public static void syncWorker(List<String> failedIdx, List<Integer> points) {
        Map<String, String> workers = Common.getWorkers();

        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            String url = entry.getValue();
            if (!idx.equals("0")) {
                String res = FaultToleranceGRPCRequest.sendWeightsRedistribute(url, failedIdx, points);
                assert(res.equals("ok"));
            }
        }
    }

    /*
        Called by the central node, find the desired weights and fetch them
    */
    public static Map<String, Object[]> weightRedistributeCentralHandler(List<Integer> partitionPoint) {
        List<Integer> prevPoint = Training.getPartitionPoint();
        int curIdx = 0, newIdx = 0;

        Pair<Integer, Integer> oldLayerPair = General.getLayerFromPoint(prevPoint, curIdx);
        int oldStart = oldLayerPair.first, oldEnd = oldLayerPair.second;

        if (oldEnd == -1) {
            oldEnd = Training.getTotalLayers() - 1;
        }

        Pair<Integer, Integer> newLayerPair = General.getLayerFromPoint(partitionPoint, newIdx);
        int newStart = newLayerPair.first, newEnd = newLayerPair.second;

        if (newEnd == -1) {
            newEnd = Training.getTotalLayers() - 1;
        }

        Map<String, Object[]> params = new HashMap<>();
        for (int l = newStart; l <= newEnd; l++) {
            if (l >= oldStart && l <= oldEnd) {
                // fetch weights from the local model
                params.put(String.valueOf(l), ReplicationUtils.getSubModelWeightsByLayer(l, true));
            } else {
                params.put(String.valueOf(l), FaultTolerance.getGlobalWeightsByLayer(String.valueOf(l)));
            }
        }
        return params;
    }

    /*
        Newly added for the weight redistribution phase
        Waiting for all workers to finish the redistribution
     */
    public static void awaitSync() {
        // wait for the redistributedDevice.size() == workers.size() - 1
        while (FaultTolerance.getRedistributedDeviceNum() != Common.getWorkerNum() - 1) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    /*
        After weight redistribution, ask all nodes to create new sub model and load new parameters
     */
    public static void commitWorkers(List<Integer> partitionPoint, int iterId) {
        Map<String, String> workers = Common.getWorkers();
        List<String> done = new java.util.ArrayList<>();

        FaultTolerance.clearPassiveCommitDoneWorkers();

        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            String url = entry.getValue();
            if (!idx.equals("0")) {
                Thread thread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        String res = FaultToleranceGRPCRequest.commitFaultSync(url, partitionPoint, iterId);
                        if (!res.equals("ok")) {
                            Common.printLog("Commit fault sync failed, url: " + url);
                        }
                    }
                }, "SyncThread_" + idx);
                thread.start();
            }
        }

        while (FaultTolerance.getPassiveCommitDoneWorkerNum() != workers.size() - 1) {
            try {
                Thread.sleep(500); // 100毫秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        Common.printLog("Commit fault sync finished ...");
    }

    /*
        Called by workers, commit the fault sync operation
     */
    public static void commitFaultSyncHandler(List<Integer> points, int iterId) {
        Map<String, Object[]> params = FaultTolerance.getNeededParams();
        Common.printLog("Re-creating sub-model according the new partition points with iterId " + iterId);
        Model.createSubModel(points);

        Common.printLog("Loading backup parameters ...");
        PassiveFTUtils.loadBackupParams(params);
        Training.setPartitionPoint(points);
        FaultTolerance.clearNeededParams();

        // re-initialize the optimizer
        Common.printLog("Re-initializing optimizer ...");
        Optimizer.createSubOptimizer();

        // get commit
        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        commitLock.lock();
        try {
            FaultTolerance.setStartIterId(iterId);
            FaultTolerance.updateTerm();
            commit.put("forwardId", iterId - 1);
            commit.put("backwardId", iterId - 1);
            commitCondition.signalAll();
        } finally {
            commitLock.unlock();
        }

        String centralUrl = Common.getUrlFromWorker(0);
        String res = FaultToleranceGRPCRequest.notifyPassiveCommitFinish(centralUrl);
        Common.printLog("Commit fault sync finished ...");
    }

    public static void weightRedistributeWorkerHandler(List<String> failedIdx, List<Integer> points) {
        Common.printLog("Redistributing weights with failed index " + failedIdx.toString() + ", new partition points " + points.toString() + " ...");

        List<Integer> prevPoint = Training.getPartitionPoint();
        int prevIdx = Common.getPrevDeviceIdx();
        int newIdx = Common.getDeviceIdx();

        Pair<Integer, Integer> oldLayerPair = General.getLayerFromPoint(prevPoint, prevIdx);
        int oldStart = oldLayerPair.first, oldEnd = oldLayerPair.second;
        if (oldEnd == -1) {
            oldEnd = Training.getTotalLayers() - 1;
        }

        // The layer of the previous node
        Pair<Integer, Integer> prevLayerPair = General.getLayerFromPoint(points, prevIdx - 1);
        int prevStart = prevLayerPair.first, prevEnd = prevLayerPair.second;
        if (prevEnd == -1) {
            prevEnd = Training.getTotalLayers() - 1;
        }

        Pair<Integer, Integer> newLayerPair = General.getLayerFromPoint(points, newIdx);
        int newStart = newLayerPair.first, newEnd = newLayerPair.second;

        if (newEnd == -1) {
            newEnd = Training.getTotalLayers() - 1;
        }

        Map<String, List<Integer> > neededLayers = new HashMap<>();
        for (int l = newStart; l <= newEnd; l++) {
            String prevIdxStr = String.valueOf(prevIdx);
            if (!failedIdx.contains(prevIdxStr) && ((l >= oldStart && l <= oldEnd) || (l >= prevStart && l <= prevEnd))) {
                // Fetch weights locally
                neededLayers.computeIfAbsent(prevIdxStr, k -> new java.util.ArrayList<>());
                neededLayers.get(prevIdxStr).add(l);
            } else {
                int j = General.findIdxByLayer(points, l);
                String jStr = String.valueOf(j);
                if (failedIdx.contains(jStr)) {
                    if (j + 1 == prevPoint.size() + 1 || failedIdx.contains(String.valueOf(j + 1))) {
                        // fetch weights from the central node
                        neededLayers.computeIfAbsent("0", k -> new java.util.ArrayList<>());
                        neededLayers.get("0").add(l);
                    } else {
                        // fetch from next node
                        neededLayers.computeIfAbsent(String.valueOf(j + 1), k -> new java.util.ArrayList<>());
                        neededLayers.get(String.valueOf(j + 1)).add(l);
                    }
                } else {
                    neededLayers.computeIfAbsent(jStr, k -> new java.util.ArrayList<>());
                    neededLayers.get(jStr).add(l);
                }
            }
        }

        // fetch the needed params
        Map<String, Object[]> params = new HashMap<>();
        for (Map.Entry<String, List<Integer>> entry : neededLayers.entrySet()) {
            String idx = entry.getKey();
            int idxInt = Integer.parseInt(idx);
            List<Integer> layers = entry.getValue();
            Common.printLog("Fetching weights from " + idx + " for layers: " + layers.toString());
            if (idxInt == prevIdx) {
                // fetch from local
                for (int l : layers) {
                    if (l >= oldStart && l <= oldEnd) {
                        Common.printLog("Fetching weights from local model for layer " + l);
                        params.put(String.valueOf(l), ReplicationUtils.getSubModelWeightsByLayer(l, true));
                    } else {
                        Common.printLog("Fetching weights from local replications for layer " + l);
                        params.put(String.valueOf(l), FaultTolerance.getLocalWeightsByLayer(String.valueOf(l)));
                    }
                }
            } else {
                // fetch via network
                Common.printLog("Fetching weights from remote for layers: " + layers.toString());
                if (idxInt == prevPoint.size() + 1) {
                    idx = "0";
                }
                String targetUrl = Common.getUrlFromWorker(Integer.parseInt(idx));
                Map<String, Object[]> weights = FaultToleranceGRPCRequest.getParamsFromRemote(targetUrl, layers);
                params.putAll(weights);
            }
        }
        FaultTolerance.storeNeededParams(params);

        // Notify the central node that the redistribution is finished
        notifyCentralNode();
        Common.printLog("Weight redistribution finished ...");
    }

    /*
        Notify the central node that the redistribution is finished
     */
    public static void notifyCentralNode() {
        Map<String, String> workers = Common.getWorkers();
        String url = workers.get("0");
        String res = FaultToleranceGRPCRequest.notifyRedistributionFinish(url, Common.getDeviceIdx());
        Log.i(tag, "Notify central node: " + res);
        assert(res.equals("ok"));
    }
}
