package com.example.confidant.utils;

import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Model {
    private static final String tag = "utils.Model";
    static {
        System.loadLibrary("confidant");
    }

    public static native void initModel(String name, Map<String, Double> args);
    public static native void loadModelWeights(String modelName, String weightsPath, int numLayers);

    public static native void initSubModel(String name, Map<String, Double> args, int startLayer, int endLayer);
    public static native void loadSubModelWeights(String weightsPath, int start, int end);

    public static native void initTrainEpoch();

    public static native void initTrain(int batchSize, int deviceNum, int deviceIdx);
    public static native void singleTrainOneEpoch(int epoch);

    // Offline profiling related
    public static native Object[] profileModelTime(int totalLayer, int batchSize, int seqLen);
    public static native Object[] profileDataSize(int totalLayer, int batchSize, int seqLen);

    // Fault tolerance related
    public static native Object[] retrainBatchWithIterId(int iterId);

    // Multi-processor Scheduler
    public static native void initMultiProcessorScheduler(String name, Map<String, Double> args);

    /*
        Create a whole model with given name and arguments
     */
    public static void createModel() {
        initModel(Common.getModelName(), Common.getModelArgs());
    }

    /*
        Create a sub-model with given partition point
     */
    public static void createSubModel(List<Integer> point) {
        int curIdx = Common.getDeviceIdx();
        Pair<Integer, Integer> layerPair = General.getLayerFromPoint(point, curIdx);
        Common.printLog("Partitioned layer: " + layerPair.first + " " + layerPair.second);

        Training.setTotalLayers(Common.getModelArgs().get("total_layer").intValue());
        initSubModel(Common.getModelName(), Common.getModelArgs(), layerPair.first, layerPair.second);

        Common.printLog("Creating sub-model success!");
    }

    /*
        Load the pre-trained weights for the sub-model with the given partition point
     */
    public static void loadPretrainedSubWeights(List<Integer> point) {
        String weightsPath = Common.getWeightsPath();
        int curIdx = Common.getDeviceIdx();
        Pair<Integer, Integer> layerPair = General.getLayerFromPoint(point, curIdx);
        int endLayer = layerPair.second;
        if (layerPair.second == -1) {
            endLayer = Training.getTotalLayers() - 1;
        }
        loadSubModelWeights(weightsPath, layerPair.first, endLayer);
    }

    /*
        Profile the model execution time by calling the native function
     */
    public static Pair<List<Float>, List<Float>> profileModelTimeHelper() {
        int totalLayer = Training.getTotalLayers();
        int batchSize = Common.getBatchSize();
        // int batchSize = 8;
        int seqLen = 128;
        Object[] res = profileModelTime(totalLayer, batchSize, seqLen);
        List<Float> forwardTime = new ArrayList<>();
        List<Float> backwardTime = new ArrayList<>();

        if (res != null) {
            for (int i = 0; i < res.length; i++) {
                if (i < totalLayer) {
                    forwardTime.add((Float) res[i]);
                } else {
                    backwardTime.add((Float) res[i]);
                }
            }
            return new Pair<>(forwardTime, backwardTime);
        }
        return new Pair<>(new ArrayList<>(), new ArrayList<>());
    }

    /*
        Profile the model data size by calling the native function
     */
    public static List<Float> profileDataSizeHelper() {
        int totalLayer = Training.getTotalLayers();
        int batchSize = Common.getBatchSize();
        int seqLen = 128;
        Object[] res = profileDataSize(totalLayer, batchSize, seqLen);
        if (res != null) {
            List<Float> dataSize = new ArrayList<>();
            for (int i = 0; i < res.length; i++) {
                dataSize.add((Float) res[i]);
            }
            return dataSize;
        }

        return new ArrayList<>();
    }
}
