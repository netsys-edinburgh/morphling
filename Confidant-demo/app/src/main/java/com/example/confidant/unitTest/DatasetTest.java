package com.example.confidant.unitTest;

import static com.example.confidant.utils.TrainCentral.skipOneBatch;

import android.util.Log;

public class DatasetTest {
    public static void datasetTestEntry() {
        iterateOverDatasetTest();
    }

    public static void iterateOverDatasetTest() {
        int dataLen = 24879;
        for (int i = 0; i < dataLen; i++) {
            Log.i("iterateOverDatasetTest()", "Skipping iterId: " + i);
            skipOneBatch();
        }
    }
}
