package com.example.confidant.utils;

import android.util.Log;

import com.example.confidant.globalStates.Backend;
import com.example.confidant.globalStates.Common;

import java.util.Map;

public class TrainSingle {
    private static final String tag = "utils.TrainSingle";

    public static void startTrain() {
        Common.printLog(String.format("Creating model %s...", Common.getModelName()));
        Model.createModel();

        Common.printLog("Setting up dataset: " + Common.getDatasetName() + " with batch size " + Common.getBatchSize());
        Dataset.createDataset(Common.getDatasetName(), Common.getDatasetPath(), Common.getBatchSize());

        // TODO: Initializing the optimizer

        // load pretrained weights
        String weightsPath = Common.getWeightsPath();
        Model.loadModelWeights("", weightsPath, 0);

        Log.i(tag, "Initializing executor and batch size in jni ...");
        Model.initTrain(Common.getBatchSize(), Common.getWorkerNum(), 0);
        trainSingle();
    }

    public static void trainSingle() {
        Common.printLog("Single train start");

        long startTime = System.currentTimeMillis();
        int epochs = 1;

        Map<Integer, Integer> backends = Backend.getBackendsMap();
        for (Map.Entry<Integer, Integer> entry : backends.entrySet()) {
            Common.printLog(String.format("Backend %d, NumThreads %d", entry.getKey(), entry.getValue()));
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            Model.singleTrainOneEpoch(epoch);
        }
        Common.printLog(String.format("Finish, time:%f s", (System.currentTimeMillis() - startTime)/1000.0));
    }
}
