package com.example.confidant.globalStates;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class FaultTolerance {
    private static int backwardTimeout = 20000; // Unit: ms

    private static int localReplicationInterval = 100;
    private static int globalReplicationInterval = 200;

    // Define two types indicating the replication type
    public enum ReplicationType {
        LOCAL_REPLICATION,
        GLOBAL_REPLICATION
    }

    public enum SystemStatus {
        NORMAL,
        FAULT,
        Passive_Handling,
        Proactive_Handling,
        RETRAINING
    }

    // records the batches that have finished training to avoid repeatedly triggering the fault tolerance mechanism
    private static Set<Integer> receivedIterIds = new java.util.HashSet<>();
    private static SystemStatus systemStatus = SystemStatus.NORMAL;

    // the iterId that the training starts, when the training recovers from the faults, its value will be the iterId that the training starts
    private static int startIterId = 0;
    private static int term = 0; // when partition point updates or recovery happens, term is changed to avoid training the old data

    // Used for the weight redistribution phase, indicating the device that has been redistributed
    private static Set<String> redistributedDevice = new java.util.HashSet<>();

    private static Map<String, Object[]> globalWeightsMap = new java.util.HashMap<>();
    private static Map<String, Object[]> localWeightsMap = new java.util.HashMap<>();

    private static Map<String, Object[]> neededParams = new java.util.HashMap<>();

    private static List<Integer> passiveCommitDoneWorkers = new ArrayList<>();

    private static final float computingCapacityThreshold = 0.8f;

    private static boolean proactiveFetchWeightsFinished = false;

    public static void addReceivedIterId(int iterId) {
        receivedIterIds.add(iterId);
    }

    public static boolean isReceived(int iterId) {
        return receivedIterIds.contains(iterId);
    }

    public static void clearReceivedIterIds() {
        receivedIterIds.clear();
    }

    public static void removeReceivedIterId(int iterId) {
        receivedIterIds.remove(iterId);
    }

    public static void setSystemStatus(SystemStatus status) {
        systemStatus = status;
    }

    public static SystemStatus getSystemStatus() {
        return systemStatus;
    }

    public static void addRedistributedDevice(String idx) {
        // add the idx to the redistributedDevice atomically
        synchronized (redistributedDevice) {
            boolean isAdded = redistributedDevice.add(idx);
            assert (isAdded);
        }
    }

    public static void resetRedistributedDevice() {
        redistributedDevice.clear();
    }

    public static int getRedistributedDeviceNum() {
        synchronized (redistributedDevice) {
            return redistributedDevice.size();
        }
    }

    public static void addGlobalWeightsMap(String idx, Object[] weights) {
        globalWeightsMap.put(idx, weights);
    }

    public static void addLocalWeightsMap(String idx, Object[] weights) {
        localWeightsMap.put(idx, weights);
    }

    public static void addWeightsMap(String layer, Object[] weights, ReplicationType replicationType) {
        if (replicationType == ReplicationType.LOCAL_REPLICATION) {
            localWeightsMap.put(layer, weights);
        } else {
            globalWeightsMap.put(layer, weights);
        }
    }

    public static Object[] getGlobalWeightsByLayer(String l) {
        if (globalWeightsMap.get(l) == null) {
            return new Object[0];
        }

        return globalWeightsMap.get(l);
    }

    public static Object[] getLocalWeightsByLayer(String l) {
        // return an empty Object[] if the key does not exist
        if (localWeightsMap.get(l) == null) {
            return new Object[0];
        }
        return localWeightsMap.get(l);
    }


    public static void setStartIterId(int iterId) {
        startIterId = iterId;
    }

    public static int getStartIterId() {
        return startIterId;
    }

    public static void updateTerm() {
        term += 1;
    }

    public static void storeNeededParams(Map<String, Object[]> params) {
        neededParams = params;
    }

    public static Map<String, Object[]> getNeededParams() {
        return neededParams;
    }

    public static void clearNeededParams() {
        neededParams.clear();
    }

    public static float getComputingCapacityThreshold() {
        return computingCapacityThreshold;
    }

    public static void setBackwardTimeout(int timeout) {
        backwardTimeout = timeout;
    }

    public static int getBackwardTimeout() {
        return backwardTimeout;
    }

    public static int addPassiveCommitDoneWorker(int idx) {
        synchronized (passiveCommitDoneWorkers) {
            passiveCommitDoneWorkers.add(idx);
            return passiveCommitDoneWorkers.size();
        }
    }

    public static int getPassiveCommitDoneWorkerNum() {
        synchronized (passiveCommitDoneWorkers) {
            return passiveCommitDoneWorkers.size();
        }
    }

    public static void clearPassiveCommitDoneWorkers() {
        synchronized (passiveCommitDoneWorkers) {
            passiveCommitDoneWorkers.clear();
        }
    }

    public static int getLocalReplicationInterval() {
        return localReplicationInterval;
    }

    public static int getGlobalReplicationInterval() {
        return globalReplicationInterval;
    }

    public static void setLocalReplicationInterval(int interval) {
        localReplicationInterval = interval;
    }

    public static void setGlobalReplicationInterval(int interval) {
        globalReplicationInterval = interval;
    }

    public static int getLastReceivedIterId() {
        int maxIterId = -1;
        for (int iterId : receivedIterIds) {
            if (iterId > maxIterId) {
                maxIterId = iterId;
            }
        }
        return maxIterId;
    }

    public static void setProactiveFetchWeightsFinished(boolean finished) {
        proactiveFetchWeightsFinished = finished;
    }

    public static boolean isProactiveFetchWeightsFinished() {
        return proactiveFetchWeightsFinished;
    }
}
