package com.example.confidant.utils;

import android.app.ActivityManager;
import android.content.Context;
import android.net.TrafficStats;
import android.os.Bundle;
import android.util.Pair;

import androidx.annotation.NonNull;

import com.alibaba.fastjson.JSONArray;
import com.example.confidant.R;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Config;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.OnlineGRPCRequest;

import org.yaml.snakeyaml.Yaml;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class PingValues {
    private int packetSize;
    private int totalPacketSize;
    private double min;
    private double avg;
    private double max;
    private double mdev;

    public PingValues(int packetSize, int totalPacketSize, double min, double avg, double max, double mdev) {
        this.packetSize = packetSize;
        this.totalPacketSize = totalPacketSize;
        this.min = min;
        this.avg = avg;
        this.max = max;
        this.mdev = mdev;
    }

    public int getPacketSize() {
        return packetSize;
    }

    public double getMin() {
        return min;
    }

    public double getAvg() {
        return avg;
    }

    public double getMax() {
        return max;
    }

    public double getMdev() {
        return mdev;
    }

    public int getTotalPacketSize() {
        return totalPacketSize;
    }

    @Override
    public String toString() {
        return "PingValues{" +
                "packetSize=" + packetSize +
                ", totalPacketSize=" + totalPacketSize +
                ", min=" + min +
                ", avg=" + avg +
                ", max=" + max +
                ", mdev=" + mdev +
                '}';
    }
}

public class General {

    public static native void syncGlobalModelName(int globalModalName);
    public static native float profileTransformerBlock(String modelName, Map<String, Double> args, int numBlocks);

    /*
        Get the ip address of the device
     */
    public static String getDeviceIPAddress() {
        try {
            Enumeration<NetworkInterface> networkInterfaces = NetworkInterface.getNetworkInterfaces();
            while (networkInterfaces.hasMoreElements()) {
                NetworkInterface networkInterface = networkInterfaces.nextElement();
                Enumeration<InetAddress> inetAddresses = networkInterface.getInetAddresses();
                while (inetAddresses.hasMoreElements()) {
                    InetAddress inetAddress = inetAddresses.nextElement();
                    if (!inetAddress.isLoopbackAddress() && inetAddress.getAddress().length == 4) {
                        return inetAddress.getHostAddress();
                    }
                }
            }
        } catch (SocketException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void printTrainedModel() {
        switch (Common.globalModelName) {
            case BERT:
                Common.printLog("Model to be trained: BERT");
                break;
            case GPT2:
                Common.printLog("Model to be trained: GPT2");
                break;
            case Phi2:
                Common.printLog("Model to be trained: Phi2");
                break;
            case LLaMA:
                Common.printLog("Model to be trained: LLaMA");
                break;
            default:
                Common.printLog("Model to be trained: Unknown");
                break;
        }
    }

    /*
        Sync the global states from Java to C++
     */
    public static void syncGlobalStates() {
        syncGlobalModelName(Common.globalModelName.ordinal());
    }

    // Get config file by the model name
    public static InputStream getInputStream(@NonNull Context context) {
        switch (Common.globalModelName) {
            case BERT:
                return context.getResources().openRawResource(R.raw.bert_config);
            case GPT2:
                return context.getResources().openRawResource(R.raw.gpt2_config);
            case Phi2:
                return context.getResources().openRawResource(R.raw.phi2_config);
            case LLaMA:
                return context.getResources().openRawResource(R.raw.llama_config);
            default:
                return null;
        }
    }

    /*
        Load config from the config file and store the related arguments
     */
    public static void loadConfig(@NonNull Context context) {
        InputStream inputStream = General.getInputStream(context);

        if (inputStream == null) {
            Common.printLog("General.loadConfig: Failed to load config file");
            return;
        }

        Yaml yaml = Config.getYaml();
        Map<String, Object> cfg;
        cfg = yaml.load(inputStream);

        // model
        Common.setModelName((String) cfg.get("model_name"));
        Common.setModelArgs((Map<String, Double>) cfg.get("model_args"));
        Training.setAggregateInterval((int) cfg.get("weight_aggregation_interval"));

        // optimizer
        Map<String, Object> schedule = (Map<String, Object>) cfg.get("schedule");
        Training.setOptName((String) schedule.get("opt_name"));
        Training.setOptArgs((Map<String, Double>) schedule.get("opt_args"));

        // datasets
        Map<String, Object> datasets = (Map<String, Object>) cfg.get("data");
        Common.setDatasetName((String) datasets.get("name"));
        Common.setDatasetPath((String) datasets.get("path"));
        Common.setBatchSize((int) datasets.get("batch_size"));

        Training.setEpochs((int) schedule.get("total_epochs"));

    }

    public static Map<String, Object> getConfigInfo(@NonNull Context context){
        InputStream inputStream = General.getInputStream(context);

        if (inputStream == null) {
            Common.printLog("General.loadConfig: Failed to load config file");
            return null;
        }

        Yaml yaml = Config.getYaml();
        Map<String, Object> cfg;
        cfg = yaml.load(inputStream);

        return cfg;
    }


    public static class PingThread extends Thread {
        private Process process;

        private String ipAddr;
        private int testTimes = 10;
        private int packetSize = 1472;

        public PingThread(String ipAddr, int testTimes, int packetSize) {
            this.ipAddr = ipAddr;
            this.testTimes = testTimes;
            this.packetSize = packetSize;
        }

        public PingThread(String ipAddr) {
            this.ipAddr = ipAddr;
        }

        @Override
        public void run() {
            super.run();
            boolean isRun = true;
            do {
                String line = null;
                StringBuilder pingResult = new StringBuilder();
                BufferedReader reader = null;
                // Here we ping 20 times with the packet size of 1472 bytes
                // The header size is 28 bytes, so the total packet size is 1500 bytes
                String command = "ping -c 20 -s 1472 " + ipAddr;
                Bundle bundle = new Bundle();

                try {
                    process = Runtime.getRuntime().exec(command);
                    reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    while ((line = reader.readLine()) != null) {
                        pingResult.append(line).append("\n");
                    }
                    Common.setPingResult(pingResult.toString());
                    reader.close();
                } catch (IOException e) {
                    pingResult.append("Ping failed!").append("\n");
                }
                isRun = false;
            } while (isRun);
        }
    }

    static public PingValues extractPingValues(String pingOutput) {
        Pattern packetSizePattern = Pattern.compile("PING .* (\\d+)\\((\\d+)\\) bytes of data.");
        Pattern rttPattern = Pattern.compile("rtt min/avg/max/mdev = ([\\d.]+)/([\\d.]+)/([\\d.]+)/([\\d.]+) ms");

        Matcher packetSizeMatcher = packetSizePattern.matcher(pingOutput);
        Matcher rttMatcher = rttPattern.matcher(pingOutput);
        int packetSize = -1, totalDataSize = -1;
        double min = -1, avg = -1, max = -1, mdev = -1;
        if (packetSizeMatcher.find()) {
            packetSize = Integer.parseInt(packetSizeMatcher.group(1));
            totalDataSize = Integer.parseInt(packetSizeMatcher.group(2));
        }
        if (rttMatcher.find()) {
            min = Double.parseDouble(rttMatcher.group(1));
            avg = Double.parseDouble(rttMatcher.group(2));
            max = Double.parseDouble(rttMatcher.group(3));
            mdev = Double.parseDouble(rttMatcher.group(4));

        }
        return new PingValues(packetSize, totalDataSize, min, avg, max, mdev);
    }

    static public double calculateBandwidth(String pingResults) {
        PingValues pingValues = extractPingValues(pingResults);
        double bandwidth = pingValues.getTotalPacketSize() * 2 / pingValues.getAvg(); // getAvg(): ms
        return bandwidth < 0.0 ? Float.MAX_VALUE : bandwidth;
    }

//    public static double measureNeighborBandwidth(String url) {
//        Common.printLog(String.format("Measuring bandwidth between local device and %s", url));
//        // extract ip from url like "http://[ip addr]:[port]"
//        String ip = url.split(":")[0];
//        PingThread pingThread = new General.PingThread(ip);
//        pingThread.start();
//
//        try {
//            pingThread.join();
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//
//        double bandwidth = General.calculateBandwidth(Common.getPingResult());
//        return bandwidth;
//    }

    /**
     * Measure the bandwidth between local device and device of url by transmitting a tensor
     * @param url
     * @return
     */
    public static double measureNeighborBandwidth(String url) {
        Common.printLog(String.format("Measuring bandwidth between local device and %s", url));

        // get the uid of the current process
        int uid = android.os.Process.myUid();

        // Random tensor
        int batchSize = Common.getBatchSize();
        int seqLen = 256;
        int hiddenSize = 768;

        // generate a random value [0, 1] List<Float> with size batchSize * seqLen * hiddenSize
        List<Float> data = new ArrayList<>();
        for (int i = 0; i < batchSize * seqLen * hiddenSize; i++) {
            data.add((float) Math.random());
        }

        List<Integer> dataShape = new ArrayList<>();
        dataShape.add(batchSize);
        dataShape.add(seqLen);
        dataShape.add(hiddenSize);

        Date startDate = new Date();
        long startTime = startDate.getTime();
        long prevTxBytes = TrafficStats.getUidTxBytes(uid);

        OnlineGRPCRequest.sendTensorTest(url, data, dataShape);

        long curTxBytes = TrafficStats.getUidTxBytes(uid);
        Date endDate = new Date();
        long endTime = endDate.getTime();

        double transmissionTime = (endTime - startTime);
        double bandwidth = (curTxBytes - prevTxBytes) / transmissionTime; // Bytes / ms
        return bandwidth;
    }

    /*
        Get the corresponding layers according to the point and device index
     */
    public static Pair<Integer, Integer> getLayerFromPoint(List<Integer> point, int deviceIdx) {
        int workerNum = point.size() + 1;
        if (deviceIdx == 0) {
            return new Pair(0, point.get(0));
        } else if (deviceIdx == workerNum - 1) {
            return new Pair(point.get(deviceIdx - 1) + 1, -1);
        } else {
            return new Pair(point.get(deviceIdx - 1) + 1, point.get(deviceIdx));
        }
    }

    /*
        Find the index of the device according to the layer
     */
    public static int findIdxByLayer(List<Integer> point, int layer) {
        for (int i = 0; i < point.size(); i++) {
            if (layer <= point.get(i)) {
                return i;
            }
        }
        return point.size(); // last device
    }

    /*
        Convert the Object[] into byte array for transmission
     */
    public static byte[] convertToByteArray(Object[] objects) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ObjectOutputStream oos = new ObjectOutputStream(bos)) {

            oos.writeObject(objects);
            oos.flush();
            return bos.toByteArray();
        }
    }

    /*
        Convert the byte array back to Object[]
     */
    public static Object[] convertToObjectArray(byte[] byteArray) throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(byteArray);
             ObjectInputStream ois = new ObjectInputStream(bis)) {

            return (Object[]) ois.readObject();
        }
    }

    /*
        Parse the JSON object sent by other nodes
     */
    public static Object[] parseJSONObject(JSONArray data) {
        Object[] output = new Object[data.size()];

        for (int i = 0; i < data.size(); i += 4) {
            int type = data.getIntValue(i + 3);
            JSONArray cur = data.getJSONArray(i);
            int dataSize = cur.size();

            long copyTime = System.currentTimeMillis();
            if (type == Common.MNNDataType.valueOf("MNN_FLOAT").ordinal()) {
                float[] inputData = new float[dataSize];
                for (int j = 0; j < inputData.length; j++) {
                    inputData[j] = cur.getFloatValue(j);
                }
                // double[] inputData = IntStream.range(0, dataSize).mapToDouble(cur::getDoubleValue).toArray();
                output[i] = inputData;
            } else if (type == Common.MNNDataType.valueOf("MNN_INT").ordinal()) {
                int[] inputData = new int[dataSize];
                for (int j = 0; j < inputData.length; j++) {
                    inputData[j] = cur.getIntValue(j);
                }
                output[i] = inputData;
            } else {
                Common.printLog("General.parseJSONObject: Unknown data type");
                throw new RuntimeException("Unknown data type");
            }
            Common.printLog("Copy data time: " + (System.currentTimeMillis() - copyTime)/1000.0 + " s");

            // inputDatas.add(inputData);

            JSONArray dimJsonArr = data.getJSONArray(i + 1);
            ArrayList<Integer> dim = new ArrayList<>();
            for (int j = 0; j < dimJsonArr.size(); j++) {
                dim.add(dimJsonArr.getIntValue(j));
            }
            output[i + 1] = dim;
            output[i + 2] = data.getIntValue(i + 2);
            output[i + 3] = type;
        }
        return output;
    }

    /**
     * Convert the JSONObject to List<Integer> Type
     */
    public static List<Integer> convertJSONToIntegerList(JSONArray listObject) {
        List<Integer> list = new ArrayList<>();

        for (Object obj : listObject) {
            list.add((Integer) obj);
        }
        return list;
    }

    /**
     * Call the JNI function to profile the computation time of a transformer block
     * @return
     */
    public static float profileTransformerBlockHelper(int numBlocks) {
        Common.printLog("Profiling transformer block ...");

        String modelName = Common.getModelName();
        Map<String, Double> modelArgs = Common.getModelArgs();

        return profileTransformerBlock(modelName, modelArgs, numBlocks);
    }

    /**
     * Acquire the memory info of the device, including the available memory and the total memory
     */
    public static List<Float> getCurrentMemoryInfo() {
        Context context = Common.getContext();
        ActivityManager activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memoryInfo);

        // unit: byte, converted to GB
        List<Float> memoryList = new ArrayList<>();
        memoryList.add(memoryInfo.availMem / 1024f / 1024f / 1024f);
        memoryList.add(memoryInfo.totalMem / 1024f / 1024f / 1024f);

        return memoryList;
    }
}
