package com.example.confidant.unitTest;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.utils.Model;

import java.util.Map;

public class MultiProcessorSchedulerTest {
    static {
        System.loadLibrary("confidant");
    }

    public static native void MPSProfileTest(String name, Map<String, Double> args);

    public static void multiProcessorSchedulerTestEntry() {
        Common.printLog("Multi Processor Scheduler Test Entry");

        String modelName = Common.getModelName();
        Map<String, Double> modelArgs = Common.getModelArgs();

        MPSProfileTest(modelName, modelArgs);
    }

}
