package com.example.confidant.utils;

import android.graphics.Bitmap;

import com.example.confidant.globalStates.Common;

public class Dataset {
    static {
        System.loadLibrary("confidant");
    }

    public static native void initDataset(String basePath, String name, String path, int batchSize);
    public static native int getDataLen();

    public static void createDataset(String name, String path, int batchSize) {
        String basePath = Common.getDatasetBasePath();
        initDataset(basePath, name, path, batchSize);
    }
}
