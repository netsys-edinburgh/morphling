package com.example.confidant.utils;

import android.util.Log;

import com.example.confidant.globalStates.Training;

import java.util.Map;

public class Optimizer {
    private static final String tag = "utils.Optimizer";
    static {
        System.loadLibrary("confidant");
    }

    public static native void initSubOptimizer(String name, Map<String, Double> args);

    public static native double getLearningRate();
    public static native void setLearningRate(double lr);

    public static native void resetOptimizer();

    public static void createSubOptimizer() {
        String optName = Training.getOptName();
        Map<String, Double> optArgs = Training.getOptArgs();

        initSubOptimizer(optName, optArgs);
        Log.i(tag, "Creating sub-optimizer success!");
    }
}
