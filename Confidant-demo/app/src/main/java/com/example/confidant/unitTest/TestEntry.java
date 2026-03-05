package com.example.confidant.unitTest;

import android.app.ActivityManager;
import android.content.Context;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.utils.Dataset;
import com.example.confidant.utils.Model;
import com.example.confidant.utils.Optimizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestEntry {
    public static void testEntry(Context context) {
        int deviceIdx = 0;
        initDatasetForTest();
        initWorkersForTest();
        initTrainingForTest(deviceIdx);

        if (deviceIdx == 0) {
               OfflineProfilerTest.offlineProfilerTestEntry();
            // GrpcClientTest.GrpcClientTestEntry();
            // TrainCentralTest.TrainCentralTestEntry();
//            MultiProcessorSchedulerTest.multiProcessorSchedulerTestEntry();
//            CrossFrameworkAdapterTest.crossFrameworkAdapterEntry();
//            DatasetTest.datasetTestEntry();
            FaultToleranceTest.faultToleranceTestEntry();
        } else {
            GrpcServerTest.GrpcServerTestEntry(1);
            // HttpServerTest.HttpServerTestEntry();
        }
    }

    // The following function simulates the initialization of the collaborative training
    public static void initDatasetForTest() {
        Common.printLog("Setting up dataset: " + Common.getDatasetName() + " with batch size " + Common.getBatchSize());
        Dataset.createDataset(Common.getDatasetName(), Common.getDatasetPath(), Common.getBatchSize());
    }

    public static void initOptimizerForTest() {
        Common.printLog("Initializing optimizer ...");
        Optimizer.createSubOptimizer();
    }

    public static void initTrain() {
        Common.printLog("Initializing executor and batch size in jni ...");
        Model.initTrain(Common.getBatchSize(), Common.getWorkerNum(), Common.getDeviceIdx());
    }

    public static void initWorkersForTest() {
        Map<String, String> workers = new HashMap<String, String>() {{
//            put("0", "192.168.1.119:6800");
//            put("1", "192.168.1.122:6800");
            put("0", "10.193.223.143:6800");
            put("1", "10.192.158.70:6800");
        }};

        Common.updateWorkers(workers);
    }

    /*
        Prepare the sub model for the test.
    */
    public static void initSubModelForTest(int deviceIdx) {
        List<Integer> point = new ArrayList<Integer>(){{
            add(5);
            // add(9);
        }};
        Training.setPartitionPoint(point);

        Common.printLog("Creating sub model ...");
        Common.setDeviceIdx(deviceIdx);
        Model.createSubModel(point);

        Common.printLog("Loading pre-trained weights from " + Common.getWeightsPath());
        // Model.loadPretrainedSubWeights(point);
    }


    public static void initTrainingForTest(int deviceIdx) {
        initSubModelForTest(deviceIdx);
        initWorkersForTest();
        initDatasetForTest();
        initOptimizerForTest();
        initTrain();

        Training.initCommit();
        Training.setCurrentEpoch(0);
    }
}
