package com.example.confidant.faultTolerance;

import android.util.Log;
import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.FaultToleranceGRPCRequest;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;
import com.example.confidant.utils.Model;
import com.example.confidant.utils.Offline;
import com.example.confidant.utils.DynamicScheduler;
import com.example.confidant.utils.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class PassiveFTHandler {
    private static final String tag = "PassiveFTHandler";

    public static void backwardTimeoutHandler(int iterId) {
        Map<String, Object> commit = Training.getCommit();

        if (iterId < (int) commit.get("backwardId") || FaultTolerance.isReceived(iterId)) {
            // Avoid calling the handler for the subsequent iterId
            return ;
        }

        if (FaultTolerance.getSystemStatus() == FaultTolerance.SystemStatus.NORMAL) {
            Common.printLog("The batch is not received in time, passive fault tolerance triggered, iter_id = " + iterId);

            // measure the time
            long recoverStart = System.currentTimeMillis();

            FaultTolerance.setSystemStatus(FaultTolerance.SystemStatus.Passive_Handling);
            int res = faultHandler(iterId);
            long recoverEnd = System.currentTimeMillis();
            Common.printLog("Recover time: " + (recoverEnd - recoverStart) + "ms");

            if (res > 0) {
                // fault happens
                long retrainStart = System.currentTimeMillis();
                Common.printLog("Retrain start ...");
                retrainBatch(res, iterId);
                long retrainEnd = System.currentTimeMillis();
                Common.printLog("Retrain time: " + (retrainEnd - retrainStart) + "ms");
            }

            long totalEnd = System.currentTimeMillis();
            Common.printLog("Total fault tolerance time: " + (totalEnd - recoverStart) + "ms");
            FaultTolerance.setSystemStatus(FaultTolerance.SystemStatus.NORMAL);
        }
    }

    /*
       If fault happens, after three-phase recovery, the batch should be retrained
    */
    public static void retrainBatch(int prevNum, int iterId) {
        Common.printLog("Retrain the batch that did not receive the backward data ...");
        Long retrainStart = System.currentTimeMillis();

        Semaphore sem = Common.getSemaphore();
        // First release all the semaphore
        for (int i = 0; i < prevNum; i++) {
            try {
                sem.release();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // reset the commit status
        Map<String, Object> commit = Training.getCommit();
        Lock commitLock = (Lock) commit.get("lock");
        Condition commitCondition = (Condition) commit.get("lockCondition");

        assert commitLock != null;
        assert commitCondition != null;

        commitLock.lock();

        try {
            commit.put("forwardId", iterId - 1);
            commit.put("backwardId", iterId - 1);
            commitCondition.signalAll();
        } finally {
            commitLock.unlock();
        }

        Common.initSemaphore();
        Semaphore newSem = Common.getSemaphore();
        FaultTolerance.setStartIterId(iterId);
        FaultTolerance.updateTerm();
        while (true) {
            try {
                newSem.acquire();
                commitLock.lock();

                int batchDiff = Common.getWorkerNum();
                // Key condition check to guarantee the 1F1B rule
                // If the iterId does not satisfy the condition, the related thread is blocked
                while (!((int) commit.get("forwardId") == iterId - 1 && (iterId - FaultTolerance.getStartIterId() - batchDiff < 0 || (int) commit.get("backwardId") == iterId - batchDiff))) {
                    Log.i(tag, String.format("Forward need to wait: forward id: %d, backward id: %d, iter id: %d", (int) commit.get("forwardId"), (int) commit.get("backwardId"), iterId));
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
                double lr = 0.1;

                OnlineGRPCRequest.sendLabels(lossUrl, iterId, (Object[]) output[1]);
                OnlineGRPCRequest.sendTrainForward(nextUrl, iterId,1, modelVersion, lr, (Object[]) output[0]);

                iterId += 1;
                newSem.release();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        Long retrainEnd = System.currentTimeMillis();
        Common.printLog("Retrain finished, time: " + (retrainEnd - retrainStart) + "ms");
    }

    public static int faultHandler(int iterId) {
        Pair<List<String>, List<String>> failedRestartIdx = PassiveFTUtils.findFailedDevice();
        List<String> failedIdx = failedRestartIdx.first;
        List<String> restartIdx = failedRestartIdx.second;

        if (failedIdx.size() == 0 && restartIdx.size() == 0) {
            Common.printLog("No failed or restart device, iter_id = " + iterId);
            return 0;
        }

        // print the failedIdx and restartIdx
        if (failedIdx.size() > 0) {
            Common.printLog("Failed device: " + failedIdx);
        }

        if (restartIdx.size() > 0) {
            Common.printLog("Restart device: " + restartIdx);
        }

        // Three-phase Recovery
        List<Integer> prevPoint = Training.getPartitionPoint(); // for calculating the needed layer after recovery
        if (restartIdx.size() > 0) {
            // TODO: What args should be transmitted
            Common.printLog("Some workers restart immediately");
            Map<String, String> workers = Common.getWorkers();
            for (String idx : restartIdx) {
                String curUrl = Common.getUrlFromWorker(Integer.parseInt(idx));
                String res = FaultToleranceGRPCRequest.sendRestartSyncState(curUrl, idx, workers, prevPoint);
                assert res.equals("ok");
            }
        }

        PassiveFTUtils.updateWorkersByFailedIdx(failedIdx);

        // put the restartIdx into the failedIdx
        failedIdx.addAll(restartIdx);

        // Phase 1: Model Repartitioning
        Common.printLog("Phase 1: Re-partitioning start...");
        Offline.distributeWorkers();
        List<Integer> partitionPoint = Training.getPartitionPoint();
        if (failedIdx.size() > 0) {
//            partitionPoint = DynamicScheduler.calculatePartitionPoint(false);
            partitionPoint = new ArrayList<Integer>(){{
                add(10);
            }};
            Common.printLog("Repartitioning point: " + partitionPoint);
        }

        // Phase 2: Weights Redistribution
        Common.printLog("Phase 2: Weights redistribution start...");
        RedistributionUtils.syncWorker(failedIdx, partitionPoint);

        Map<String, Object[]> centralParams = RedistributionUtils.weightRedistributeCentralHandler(partitionPoint);
        RedistributionUtils.awaitSync();

        // Phase 3: Commit the redistribution
        Common.printLog("Phase 3: Commit the redistribution start...");
        RedistributionUtils.commitWorkers(partitionPoint, iterId);

        // Central node create new model
        Common.printLog("Re-creating model ...");
        Model.createSubModel(partitionPoint);
        PassiveFTUtils.loadBackupParams(centralParams);
        Common.printLog("Re-initializing optimizer ...");
        Optimizer.createSubOptimizer();
        Training.setPartitionPoint(partitionPoint);

        // return the length of the previous workers, which is used for retraining
        return prevPoint.size() + 1;
    }

    /*
        Called by restarted worker, sync the states before training
     */
    public static void restartSyncStates(int idx, Map<String, String> workers, List<Integer> partitionPoint) {
        Common.printLog("Restart device, syncing states ...");
        Common.setDeviceIdx(idx);
        Common.updateWorkers(workers);
        Training.setPartitionPoint(partitionPoint);

        Model.createSubModel(partitionPoint);
        Optimizer.createSubOptimizer();
    }
}
