package com.example.confidant.faultTolerance;

import android.util.Log;
import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.FaultToleranceGRPCRequest;
import com.example.confidant.utils.General;
import com.example.confidant.utils.Model;
import com.example.confidant.utils.Optimizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ProactiveFTUtils {
    private static final String tag = "ProactiveFTUtils";

    /*
        Find the substitute device by choosing the device with the largest DC value
     */
    public static String selectSubstituteDevice(Map<String, DeviceCompatibilityInfo> deviceCompatibilities) {
        String ret = "";
        float eta = 1e-5F; // avoid division by zero

        if (deviceCompatibilities.size() == 1) {
            // only one device, return the first url
            for (Map.Entry<String, DeviceCompatibilityInfo> entry : deviceCompatibilities.entrySet()) {
                ret = entry.getKey();
            }
            return ret;
        }

        // find the max/min ccv and charge counter for normalization
        int maxChargeCounter = 0;
        int minChargeCounter = Integer.MAX_VALUE;

        float maxComputingCapacity = 0;
        float minComputingCapacity = Float.MAX_VALUE;

        for (Map.Entry<String, DeviceCompatibilityInfo> entry : deviceCompatibilities.entrySet()) {
            DeviceCompatibilityInfo curDcInfo = entry.getValue();
            if (curDcInfo.getChargeCounter() > maxChargeCounter) {
                maxChargeCounter = curDcInfo.getChargeCounter();
            }
            if (curDcInfo.getChargeCounter() < minChargeCounter) {
                minChargeCounter = curDcInfo.getChargeCounter();
            }

            if (curDcInfo.getTotalComputingCapacity() > maxComputingCapacity) {
                maxComputingCapacity = curDcInfo.getTotalComputingCapacity();
            }
            if (curDcInfo.getTotalComputingCapacity() < minComputingCapacity) {
                minComputingCapacity = curDcInfo.getTotalComputingCapacity();
            }
        }

        // compute p, the remaining training process
        int currentEpoch = Training.getCurrentEpoch();
        int totalEpoch = Training.getEpochs();
        int dataLen = Training.getDataLen();
        int remainingBatch = dataLen - (FaultTolerance.getLastReceivedIterId() + 1);
        float p = (remainingBatch + (totalEpoch - currentEpoch) * dataLen) * 1.0f / (totalEpoch * dataLen);

        // compute DC and find the max DC
        float maxDC = 0.0f;
        for (Map.Entry<String, DeviceCompatibilityInfo> entry : deviceCompatibilities.entrySet()) {
            String url = entry.getKey();
            DeviceCompatibilityInfo curDcInfo = entry.getValue();
            float cc = curDcInfo.getTotalComputingCapacity();

            // normalize cc
            float ccNorm = (cc - minComputingCapacity) / (maxComputingCapacity - minComputingCapacity);

            // normalize chargeCounter
            float chargeCounterNorm = (curDcInfo.getChargeCounter() - minChargeCounter) * 1.0f / (maxChargeCounter - minChargeCounter);

            float DC = p * chargeCounterNorm / (ccNorm + eta);
            if (DC >= maxDC) {
                maxDC = DC;
                ret = url;
            }
        }

        return ret;
    }

    /**
     * Profile the computing capacity vector
     */
    public static List<Float> profileComputingCapacityVector() {
        int N = 3;
        List<Float> ret = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            float curTime = General.profileTransformerBlockHelper(i);
            ret.add(curTime);
        }
        Common.setCurrentDeviceCapacity(ret);
        return ret;
    }

    /**
     * Called by substitute device, fetch weights from the quit device
     */
    public static void fetchWeightsFromQuitDeviceHandler(String quitUrl, String quitIdx, String centralUrl, List<Integer> points) {
        Common.printLog("Set system status to Proactive_Handling ...");
        FaultTolerance.setSystemStatus(FaultTolerance.SystemStatus.Proactive_Handling);

        Common.printLog("Fetching weights from " + quitUrl);
        Map<String, Object[]> weights = FaultToleranceGRPCRequest.fetchAllWeightsFromQuittingDevice(quitUrl);

        Common.printLog("Notify central node fetch finish");
        String res2 = FaultToleranceGRPCRequest.notifyCentralFetchFinish(centralUrl);
        assert res2.equals("ok");

        // init subModel
        Common.printLog("Init sub-model ...");
        Common.setDeviceIdx(Integer.parseInt(quitIdx));
        Training.setPartitionPoint(points);
        Model.createSubModel(points);
        Optimizer.createSubOptimizer();

        Common.printLog("Load corresponding weights");
        for (Map.Entry<String, Object[]> entry : weights.entrySet()) {
            String layer = entry.getKey();
            Object[] weight = entry.getValue();
            ReplicationUtils.loadSubModelWeightsByLayer(Integer.parseInt(layer), weight, true);
        }

        Common.printLog("Load weights success!");
    }
}
