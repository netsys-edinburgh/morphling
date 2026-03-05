package com.example.confidant.utils;

import static android.os.SystemClock.sleep;

import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.OfflineGRPCRequest;
import com.example.confidant.request.OfflineRequest;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class OfflineProfiler {
    private static final String tag = "OfflineProfiler";
    private List<Float> presumForwardTime;
    private List<Float> presumBackwardTime;

    private List<Float> bandwidth;

    private List<Float> availableMem;
    private List<Float> totalMem;

    private List<Float> transformerBlockTime;

    private List<Float> outputSize;
    private List<Float> computingCapacities;

    private int profilingRounds = 1;

    public OfflineProfiler(int totalLayer, int deviceNum) {
        presumForwardTime = new ArrayList<>();
        presumBackwardTime = new ArrayList<>();

        bandwidth = new ArrayList<>();
        outputSize = new ArrayList<>();

        computingCapacities = new ArrayList<>();
        transformerBlockTime = new ArrayList<>();

        availableMem = new ArrayList<>();
        totalMem = new ArrayList<>();

        for (int i = 0; i < deviceNum; i++) {
            computingCapacities.add(1.0f);
            bandwidth.add(-1.0f);
            transformerBlockTime.add(1.0f);
            availableMem.add(-1.0f);
            totalMem.add(-1.0f);
        }
    }

    /*
        Measure the computing time and the output datasize of the model
     */
    public void staticProfiling() {
        timeProfiling();
        dataSizeProfiling();
    }

    public void timeProfiling() {
        Date startDate = new Date();
        long startTime = startDate.getTime();

        Common.printLog("Profiling execution time ...");
        Pair<List<Float>, List<Float>> computeTime = Model.profileModelTimeHelper();
        List<Float> forwardTime = computeTime.first;
        List<Float> backwardTime = computeTime.second;

        presumForwardTime.add(forwardTime.get(0));
        presumBackwardTime.add(backwardTime.get(0));

        for (int i = 1; i < forwardTime.size(); i++) {
            presumForwardTime.add(presumForwardTime.get(i - 1) + forwardTime.get(i));
            presumBackwardTime.add(presumBackwardTime.get(i - 1) + backwardTime.get(i));
        }

        Date endDate = new Date();
        long endTime = endDate.getTime();
        Common.printLog("Profiling execution time ends, elapsed time: " + (endTime - startTime) + "ms");
    }

    public void dataSizeProfiling() {
        Date startDate = new Date();
        long startTime = startDate.getTime();

        Common.printLog("Profiling output datasize of each layer...");
        outputSize = Model.profileDataSizeHelper();

        Date endDate = new Date();
        long endTime = endDate.getTime();
        Common.printLog("Profiling output datasize ends, elapsed time: " + (endTime - startTime) + "ms");
    }

    public float getTimeInterval(int start, int end, int type) {
        if (start > end) {
            return Float.MAX_VALUE;
        }

        if (type == 0) {
            return start == 0 ? presumForwardTime.get(end) : presumForwardTime.get(end) - presumForwardTime.get(start - 1);
        }

        return start == 0 ? presumBackwardTime.get(end) : presumBackwardTime.get(end) - presumBackwardTime.get(start - 1);

    }

    public List<Float> getOutputSize() {
        return outputSize;
    }

    public List<Float> getAvailableMemory(){
        return availableMem;
    }

    public List<Float> getBandwidth() {
        return bandwidth;
    }

    public float getComputingCapacityById(int idx) {
        return computingCapacities.get(idx);
    }

    public void calculateComputingCapacities(List<Float> executionTime) {
        return ;
    }

    /*
        Measure the bandwidth between two adjacent worker nodes (i and i + 1)
     */
    public void profileBandwidth() {
        Date startDate = new Date();
        long startTime = startDate.getTime();

        Map<String, String> workers = Common.getWorkers();
        Common.setDeviceIdx(0); // for distinguishing the central node
        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            if (!idx.equals("0")) {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        String url = entry.getValue();
                        // float bw = OfflineRequest.sendMeasureBandwidth(url);
                        float bw = OfflineGRPCRequest.sendMeasureBandwidth(url);
                        if (bw < 0.0f) {
                            Common.printLog(String.format("Bandwidth %s measurement fails", url));
                            return;
                        }
                        Common.printLog(String.format("Bandwidth %s measurement ends, the value is %f", url, bw));
                        bandwidth.set(Integer.parseInt(idx), bw);
                    }
                }).start();
            }
        }
        String nextUrl = Common.getUrlFromWorker(1);
        double currentBw = General.measureNeighborBandwidth(nextUrl);
        bandwidth.set(0, (float) currentBw);
        Common.printLog(String.format("Bandwidth %s measurement ends, the value is %f", nextUrl, currentBw));

        // wait until all bandwidths are measured
        while (true) {
            sleep(1000);
            boolean allMeasured = true;
            for (int i = 0; i < bandwidth.size(); i++) {
                if (bandwidth.get(i) < 0.0f) {
                    allMeasured = false;
                    break;
                }
            }
            if (allMeasured) {
                break;
            }
        }

        Date endDate = new Date();
        long endTime = endDate.getTime();
        Common.printLog("Bandwidth measurement ends, elapsed time: " + (endTime - startTime) + "ms");
    }

    /*
        Acquire the memory information of each device
     */
    public void getMemoryInfo() {
        Map<String, String> workers = Common.getWorkers();

        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            if (!idx.equals("0")) {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        String url = entry.getValue();
                        List<Float> memInfo = OfflineGRPCRequest.getMemoryInfo(url);
                        if (memInfo.get(0) < 0.0f) {
                            Common.printLog(String.format("Fetching %s memory info fails", url));
                            return;
                        }

                        availableMem.set(Integer.parseInt(idx), memInfo.get(0));
                        totalMem.set(Integer.parseInt(idx), memInfo.get(1));
                    }
                }).start();
            }
        }
        List<Float> currentMemory = General.getCurrentMemoryInfo();
        availableMem.set(0, currentMemory.get(0));
        totalMem.set(0, currentMemory.get(1));

        // wait until all memory info are acquired
        while (true) {
            sleep(1000);
            boolean allMeasured = true;
            for (int i = 0; i < availableMem.size(); i++) {
                if (availableMem.get(i) < 0.0f) {
                    allMeasured = false;
                    break;
                }
            }
            if (allMeasured) {
                break;
            }
        }
    }

    /**
     * Profile the transformer block of each worker
     */
    public void blockProfiling() {
        Date startDate = new Date();
        long startTime = startDate.getTime();

        Map<String, String> workers = Common.getWorkers();

        for (Map.Entry<String, String> entry : workers.entrySet()) {
            String idx = entry.getKey();
            if (!idx.equals("0")) {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        String url = entry.getValue();
                        float blockTime = OfflineGRPCRequest.sendProfileTransformerBlock(url);
                        if (blockTime < 0.0f) {
                            Common.printLog(String.format("Transformer block of %s measurement fails", url));
                            return;
                        }
                        Common.printLog(String.format("Transformer block of %s measurement ends, the value is %f", url, blockTime));
                        transformerBlockTime.set(Integer.parseInt(idx), blockTime);
                    }
                }).start();
            }
        }
        float currentBlockTime = General.profileTransformerBlockHelper(1 );
        transformerBlockTime.set(0, currentBlockTime);
        Common.printLog(String.format("Local Transformer block measurement ends, the value is %f", currentBlockTime));

        // wait until all bandwidths are measured
        while (true) {
            sleep(1000);
            boolean allMeasured = true;
            for (int i = 0; i < transformerBlockTime.size(); i++) {
                if (transformerBlockTime.get(i) < 0.0f) {
                    allMeasured = false;
                    break;
                }
            }
            if (allMeasured) {
                break;
            }
        }

        Date endDate = new Date();
        long endTime = endDate.getTime();
        Common.printLog("Transformer block time measurement ends, elapsed time: " + (endTime - startTime) + "ms");
    }

    /**
     * Update the computing capacity of each worker node
     */
    public void updateComputingCapacity(List<Float> time) {
        List<Integer> prevPoint = Training.getPartitionPoint();
        for (int i = 1; i < time.size(); i++) {
            String url = Common.getUrlFromWorker(i);
            Pair<Integer, Integer> layerPair = General.getLayerFromPoint(prevPoint, i);
            if (layerPair.second == -1) {
                layerPair = new Pair<>(layerPair.first, Training.getTotalLayers() - 1);
            }
            float centralForwardTime = getTimeInterval(layerPair.first, layerPair.second, 0);
            float centralBackwardTime = getTimeInterval(layerPair.first, layerPair.second, 1);
            computingCapacities.set(i, (time.get(i) + time.get(i + 1)) / (centralForwardTime + centralBackwardTime));
        }
    }

    /**
     * Update the computing capacity of each worker node by block profiling
     */
    public void updateComputingCapacityByBlockProfiling() {
        for (int i = 1; i < transformerBlockTime.size(); i++) {
            computingCapacities.set(i, transformerBlockTime.get(i) / transformerBlockTime.get(0));
        }

        Common.printLog("Computing capacity update: " + computingCapacities);
    }
}
