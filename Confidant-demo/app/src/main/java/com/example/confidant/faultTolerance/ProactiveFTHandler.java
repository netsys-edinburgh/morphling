package com.example.confidant.faultTolerance;

import android.annotation.SuppressLint;
import android.util.Log;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.FaultToleranceGRPCRequest;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;
import com.example.confidant.request.OfflineRequest;
import com.example.confidant.utils.Model;
import com.example.confidant.utils.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class ProactiveFTHandler {
    private static final String tag = "ProactiveFT";

    /*
        Notify the central node that the device is to exit
     */
    public static void notifyExit() {
        String centralUrl = Common.getUrlFromWorker(0);
        int deviceIdx = Common.getDeviceIdx();
        FaultToleranceGRPCRequest.notifyWorkerExit(centralUrl, deviceIdx);
    }

    /*
        Main entry of the proactive handler
     */
    public static void handlerHelper(String quitIdx) {
        // Phase 1: get the required information from the device in LAN to calculate the device compatibility
        // Phase 2: find the device with the most device compatibility
        // Phase 3: If found, ask the substitute node to fetch all weights from quitting node, if not, fall back to reactive FT
        // Phase 4: Update the worker list
        // Phase 5: Restart training

        Common.printLog("Proactive fault tolerance handler start ...");
        Long startPFTTime = System.currentTimeMillis();

        // Phase 1
        List<String> urls = Common.getUrls();
        Map<String, String> workers = Common.getWorkers();

        // find url in urls that are not in workers
        Map<String, DeviceCompatibilityInfo> deviceCompatibilities = new HashMap<>();
//        AtomicInteger atomicDeviceCounter = new AtomicInteger(0);

//        for (String url : urls) {
//            // TODO: Currently we assume the devices that are not in the workers list but in the urls list are the available devices
//            if (!workers.containsValue(url)) {
//                Thread thread = new Thread(new Runnable() {
//                    @Override
//                    public void run() {
//                        String res = OfflineRequest.checkAvailable(url);
//                        if (res.equals("ok")) {
//                            atomicDeviceCounter.getAndIncrement();
//                            // device is available
//                            DeviceCompatibilityInfo deviceCompatibilityInfo = FaultToleranceGRPCRequest.getDeviceCompatibilityInfo(url);
//                            deviceCompatibilities.put(url, deviceCompatibilityInfo);
//                        }
//                    }
//                }, "ProactiveFaultHandler.CheckAvailability");
//                thread.start();
//            }
//        }
        List<String> idleUrls = Common.getIdleUrls();
        List<String> availableUrls = new ArrayList<>();

        for (String url : idleUrls) {
            String res = OfflineRequest.checkAvailable(url);
            if (res.equals("ok")) {
                // device is available
                availableUrls.add(url);
            }
        }

        for (String url : availableUrls) {
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    DeviceCompatibilityInfo deviceCompatibilityInfo = FaultToleranceGRPCRequest.getDeviceCompatibilityInfo(url);
                    deviceCompatibilities.put(url, deviceCompatibilityInfo);
                    Common.printLog("Device " + url + " finish checking");
                }
            }, "ProactiveFaultHandler.CheckAvailability");
            thread.start();
        }

        // TODO: If no available devices, fall back to passive FT
        // wait for all threads finish
        while (deviceCompatibilities.size() < availableUrls.size()) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        Long findDeviceTime = System.currentTimeMillis();
        Common.printLog("Find device time: " + (findDeviceTime - startPFTTime) + " ms");

        // find the device with the similar computing capacity (Larger or equal)
        Long selectDeviceStartTime = System.currentTimeMillis();
        String targetDeviceUrl = ProactiveFTUtils.selectSubstituteDevice(deviceCompatibilities);
        Common.printLog("Selected device: " + targetDeviceUrl);
        Long deviceSelectEndTime = System.currentTimeMillis();
        Common.printLog("Select device time: " + (deviceSelectEndTime - selectDeviceStartTime) + " ms");

        if (targetDeviceUrl.length() > 0) {
            // device found
            replaceDevice(quitIdx, targetDeviceUrl);
        }
    }

    @SuppressLint("DefaultLocale")
    public static void replaceDevice(String quitIdx, String targetDeviceUrl) {
        // recalculate the partition point
        Long replaceDeviceStartTime = System.currentTimeMillis();

        List<Integer> curPoints = Training.getPartitionPoint();

        // only need to replace weights between the quitDevice and targetDevice
        Common.printLog("Ask the substitute node to fetch all weights from quitting node ...");
        String quitUrl = Common.getUrlFromWorker(Integer.parseInt(quitIdx));
        FaultTolerance.setProactiveFetchWeightsFinished(false);

        String centralUrl = Common.getUrlFromWorker(0);
        String res = FaultToleranceGRPCRequest.notifySubstituteDevice(targetDeviceUrl, quitUrl, centralUrl, quitIdx, curPoints);
        assert res.equals("ok");

        Common.printLog("Waiting for the substitute node to finish fetching weights ...");
        // wait for the fetch finish
        while (!FaultTolerance.isProactiveFetchWeightsFinished()) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        Common.printLog("The substitute node has finished fetching weights ...");

        Long fetchWeightsTime = System.currentTimeMillis();
        Common.printLog("Fetch weights time: " + (fetchWeightsTime - replaceDeviceStartTime) + " ms");

        // start to resume training
        Long retrainStartTime = System.currentTimeMillis();
        FaultTolerance.setSystemStatus(FaultTolerance.SystemStatus.Proactive_Handling);

        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        // get the backwardedId, and the retrain starts from the backwardedId + 1
        int backwardedId = (int) commit.get("backwardId");
//        int startForwardId = backwardedId + 1;
        int startForwardId = FaultTolerance.getLastReceivedIterId();
        commitLock.lock();

        FaultTolerance.setStartIterId(startForwardId);

        // Update workers by replacing the quit device with the target device
        Common.printLog("Update workers ...");
        Map<String, String> workers = Common.getWorkers();
        workers.put(quitIdx, targetDeviceUrl);

        // notify all workers to update workers
        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            String url = entry.getValue();
            if (!idx.equals("0")) {
                String res2 = FaultToleranceGRPCRequest.proactiveFTSendWorkers(url, idx, workers, startForwardId + 1);
//                String res2 = OnlineGRPCRequest.sendWorkers(url, idx, workers);
                assert res2.equals("ok");
            }
        }

        // retraining
        int workerNum = workers.size();
        Semaphore sem = Common.getSemaphore();

        commit.put("forwardId", FaultTolerance.getLastReceivedIterId());
        commit.put("backwardId", FaultTolerance.getLastReceivedIterId());

        int iterId = FaultTolerance.getLastReceivedIterId() + 1;
        Common.printLog("Start retraining from last received iter id: " + iterId);
        // First release all the semaphore
        for (int i = 0; i < workerNum; i++) {
            try {
                sem.release();
            } catch (Exception e) {
                // e.printStackTrace();
                Log.e(tag, "semaphore release failed: " + e.getMessage());
            }
        }

        try {
            commitCondition.signalAll();
        } finally {
            if (((ReentrantLock) commitLock).isLocked()) {
                commitLock.unlock();
            }
        }

        Common.initSemaphore();
        Semaphore newSem = Common.getSemaphore();
        FaultTolerance.updateTerm();

//        FaultTolerance.setSystemStatus(FaultTolerance.SystemStatus.RETRAINING);

        while (true) {
            try {
                newSem.acquire();
                commitLock.lock();

                int batchDiff = Common.getWorkerNum();
                // Key condition check to guarantee the 1F1B rule
                // If the iterId does not satisfy the condition, the related thread is blocked
                while (!((int) commit.get("forwardId") == iterId - 1 && (iterId - FaultTolerance.getStartIterId() - batchDiff < 0 || (int) commit.get("backwardId") == iterId - batchDiff))) {
                    Log.i(tag, String.format("Proactive FT retrain: Forward need to wait: forward id: %d, backward id: %d, iter id: %d", (int) commit.get("forwardId"), (int) commit.get("backwardId"), iterId));
                    try {
                        commitCondition.await();
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }

                Object[] output = Model.retrainBatchWithIterId(iterId);
                if (((Object[]) output[0]).length == 0) {
                    // retrain finished
                    try {
                        commitCondition.signalAll();
                    } finally {
                        if (((ReentrantLock) commitLock).isLocked()) {
                            commitLock.unlock();
                        }
                    }
                    newSem.release();
                    Common.printLog("Retrain finished");

                    break;
                }

                Common.printLog(String.format("Retrain Batch %d Forward finished", iterId));

                try {
                    int forwardId = (int) commit.get("forwardId");
                    commit.put("forwardId", forwardId + 1);
                    commitCondition.signalAll();
                } finally {
                    if (((ReentrantLock) commitLock).isLocked()) {
                        commitLock.unlock();
                    }
                }

                String nextUrl = Common.getUrlFromWorker(Common.getDeviceIdx() + 1);
                String lossUrl = Common.getUrlFromWorker(Common.getWorkerNum() - 1);

                int modelVersion = 1;
                double lr = Optimizer.getLearningRate();

                OnlineGRPCRequest.sendLabels(lossUrl, iterId, (Object[]) output[1]);
                OnlineGRPCRequest.sendTrainForward(nextUrl, iterId,1, modelVersion, lr, (Object[]) output[0]);

                iterId += 1;
                newSem.release();
            } catch (InterruptedException e) {
                Log.e(tag, "retrain(): semaphore acquire failed: " + e.getMessage());
            }
        }

        Long retrainEndTime = System.currentTimeMillis();
        Common.printLog("Retrain time: " + (retrainEndTime - retrainStartTime) + " ms");

        FaultTolerance.setSystemStatus(FaultTolerance.SystemStatus.NORMAL);
    }
}
