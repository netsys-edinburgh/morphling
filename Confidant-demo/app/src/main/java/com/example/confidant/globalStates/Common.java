package com.example.confidant.globalStates;

import android.app.Activity;
import android.content.Context;
import android.os.Environment;

import androidx.recyclerview.widget.RecyclerView;

import com.example.confidant.utils.CustomView;
import com.example.confidant.utils.OfflineProfiler;
//import com.example.confidant.utils.OfflineProfiler;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class Common {
    // The same as the enum in cpp
    public enum MNNDataType {
        MNN_FLOAT,
        MNN_INT,
        MNN_NOT_YET_SET
    }

    // Used for specifying the model to be trained
    public enum ModelName {
        BERT,
        GPT2,
        Phi2,
        LLaMA
    }

    private static final boolean isSaveLog = true; // Set to false to disable log saving into the file
//    private static final boolean isSaveLog = false; // Set to false to disable log saving into the file

    public static final ModelName globalModelName = ModelName.BERT;
//    public static final ModelName globalModelName = ModelName.GPT2;
//    public static final ModelName globalModelName = ModelName.LLaMA;
//    public static final ModelName globalModelName = ModelName.Phi2;

    public static final int grpcPort = 6800;
    public static final int httpPort = 50000;

    private static String modelName;
    // Temporarily, the value of the arguments should be double for simplicity
    // When the type is int, it should be converted to int
    private static Map<String, Double> modelArgs;

    private static String datasetName;
    private static String datasetPath;
    private static String datasetBasePath;

    private static String weightsPath;

    private static String basePath;

    private static int batchSize;
    private static List<Integer> inputSize;

    private static int deviceIdx = -1;
    private static int prevDeviceIdx = -1; // used for fault tolerance

    private static Semaphore semaphore;

    // Storing the workers that participating the training, the url of the central node is specified here before training
    public static Map<String, String> workers = new HashMap<String, String>() {{
        put("0", "192.168.8.107"); //redmi k50
    }};

    // Specifying the url list that the central node searches for the available workers
    // The urls should be modified before training
    public static List<String> urls = new ArrayList<String>(Arrays.asList(
//            "192.168.8.107",//redmi red button
//            "192.168.1.242",
            "192.168.8.108", //mi
            "192.168.8.102"

    ));

    // For proactive fault tolerance, the urls of the idle workers are specified here
    public static List<String> idleUrls = new ArrayList<String>(Arrays.asList(
            "192.168.8.100"
//            "192.168.1.238"
    ));

//    public static Map<String, String> workers = new HashMap<String, String>() {{
//        put("0", "http://192.168.31.141:50000");
//    }};
//
//    // Specifying the url list that the central node searches for the available workers
//    // The urls should be modified before training
//    public static List<String> urls = new ArrayList<String>(Arrays.asList(
//            "http://192.168.31.250:50000",
//            "http://192.168.31.128:50000"
//    ));

    private static Map<String, List<Float>> workersCapacity = new HashMap<String, List<Float>>() {{
        put("0", new ArrayList<Float>(Arrays.asList(0.0f, 0.0f, 0.0f)));
    }};

    private static List<Float> currentDeviceCapacity = new ArrayList<Float>();

    private static CustomView.LogAdapter logAdapter;

    private static String logPath;

    private static RecyclerView logView;

    private static Context context;

    private static OfflineProfiler offlineProfiler;

    // for network ping
    private static String pingResult = "";

    // for grpc communication
    // key: url, value: channel
    public static Map<String, ManagedChannel> channels = new HashMap<>();

    public static ManagedChannel getChannel(String url) {
        // use lock to keep channels threads save
        synchronized (channels) {
            if (!channels.containsKey(url)) {
                Common.printLog("Creating channel for url " + url);
                int port = Common.getGrpcPort();
                ManagedChannel channel = ManagedChannelBuilder.forAddress(url, port)
                        .usePlaintext()
                        .build();
                channels.put(url, channel);
            }

            if (channels.get(url).isTerminated()) {
                Common.printLog("Recreating channel for url " + url);
                int port = Common.getGrpcPort();
                ManagedChannel channel = ManagedChannelBuilder.forAddress(url, port)
                        .usePlaintext()
                        .build();
                channels.put(url, channel);
            }

            return channels.get(url);
        }
    }

    public static void shutdownAllChannels() {
        synchronized (channels) {
            for (Map.Entry<String, ManagedChannel> entry : channels.entrySet()) {
                entry.getValue().shutdown();
            }
            channels.clear();
        }
    }

    public static void setModelName(String _modelName) {
        modelName = _modelName;
    }

    public static void setModelArgs(Map<String, Double> args) {
        modelArgs = args;
    }

    public static Map<String, String> getWorkers() {
        // The reference of the workers is returned
        return workers;
    }

    public static void updateWorkers(Map<String, String> newWorkers) {
        workers = newWorkers;
    }

    public static List<String> getUrls() {
        return urls;
    }

    public static List<String> getIdleUrls() {
        return idleUrls;
    }

    public static int getDeviceIdx() {
        return deviceIdx;
    }

    public static int getPrevDeviceIdx() {
        return prevDeviceIdx;
    }

    public static void setPrevDeviceIdx(int _prevDeviceIdx) {
        prevDeviceIdx = _prevDeviceIdx;
    }

    public static String getModelName() {
        return modelName;
    }

    public static Map<String, Double> getModelArgs() {
        return modelArgs;
    }

    public static void setDeviceIdx(int _deviceIdx) {
        deviceIdx = _deviceIdx;
    }

    public static void setWorkers(Map<String, String> _workers) {
        workers = _workers;
    }

    public static void setDatasetName(String _dataset) {
        datasetName = _dataset;
    }

    public static String getDatasetName() {
        return datasetName;
    }

    public static void setDatasetPath(String _path) {
        datasetPath = _path;
    }

    public static String getDatasetPath() {
        return datasetPath;
    }

    public static int getBatchSize() {
        return batchSize;
    }

    public static void setBatchSize(int _batchSize) {
        batchSize = _batchSize;
    }

    public static List<Integer> getInputSize() {
        return inputSize;
    }

    public static void setInputSize(List<Integer> _inputSize) {
        inputSize = _inputSize;
    }

    public static String getDatasetBasePath() {
        return datasetBasePath;
    }

    public static void setWeightsPath(String path) {
        weightsPath = path;
    }

    public static String getWeightsPath() {
        return weightsPath;
    }

    public static void setDatasetBasePath(String datasetBasePath) {
        Common.datasetBasePath = datasetBasePath;
    }

    public static String getUrlFromWorker(int idx) {
        if (idx == workers.size()) {
            idx = 0;
        }
        return workers.get(String.valueOf(idx));
    }

    public static int getWorkerNum() {
        return workers.size();
    }

    public static void initSemaphore() {
        semaphore = new Semaphore(getWorkerNum());
    }

    public static Semaphore getSemaphore() {
        return semaphore;
    }

    public static void setLogAdapter(Context _context, CustomView.LogAdapter _logAdapter) {
        logAdapter = _logAdapter;
        context = _context;

        if (isSaveLog) {
            // Create a log file in the local disk, the file name includes the current time
            String rootDir = Environment.getExternalStorageDirectory().getAbsolutePath();
            Date date = new Date();

            logPath = rootDir + File.separator + "confidant" + File.separator + "log_" + date.getTime() + ".txt";
        }
    }

    public static void setLogView(RecyclerView _logView) {
        logView = _logView;
    }

    public static void printLog(String message) {
        Activity activity = (Activity) context;
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                logAdapter.addLog(new CustomView.LogItem(message), logView);
            }
        });

        if (isSaveLog) {
            // Save the log into the file
            try {
                FileWriter writer = new FileWriter(logPath, true);
                writer.append(message);
                writer.append("\n");
                writer.flush();
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void setBasePath(String _basePath) {
        basePath = _basePath;
    }

    public static String getBasePath() {
        return basePath;
    }

    public static List<Float> getWorkerCapacity(String idx) {
        if (workersCapacity.get(idx) == null) {
            return new ArrayList<Float>();
        }

        return workersCapacity.get(idx);
    }

    public static void setWorkerCapacity(String idx, List<Float> capacity) {
        workersCapacity.put(idx, capacity);
    }

    public static List<Float> getCurrentDeviceCapacity() {
        return currentDeviceCapacity;
    }

    public static void setCurrentDeviceCapacity(List<Float> capacity) {
        currentDeviceCapacity = capacity;
    }

    public static Context getContext() {
        return context;
    }

    public static void setOfflineProfiler(OfflineProfiler _offlineProfiler) {
        offlineProfiler = _offlineProfiler;
    }

    public static OfflineProfiler getOfflineProfiler() {
        return offlineProfiler;
    }

    public static void setPingResult(String result) {
        pingResult = result;
    }

    public static String getPingResult() {
        return pingResult;
    }

    public static int getGrpcPort() {
        return grpcPort;
    }

    public static int getHttpPort() { return httpPort; }

}
