package com.example.confidant.faultTolerance;

import android.util.Log;
import android.util.Pair;

import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.grpcRequest.FaultToleranceGRPCRequest;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.rpc.api.UnifiedFloatTensorList;
import com.example.confidant.utils.General;
import com.example.confidant.utils.GrpcUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ReplicationUtils {
    private static final String tag = "Replication";
    static {
        System.loadLibrary("confidant");
    }

    /*
        layer: Absolute layer index.
        If isTrainable equals false, it gets all weights from the model.
    */
    public static native Object[] getSubModelWeightsByLayer(int layer, boolean isTrainable);
    public static native void loadSubModelWeightsByLayer(int layer, Object[] weights, boolean isTrainable);

    public static void replicateWeights(FaultTolerance.ReplicationType replicationType) {
        int deviceIdx = Common.getDeviceIdx();
        List<Integer> point = Training.getPartitionPoint();
        Pair<Integer, Integer> layerPair = General.getLayerFromPoint(point, deviceIdx);
        int startLayer = layerPair.first, endLayer = layerPair.second;

        if (endLayer == -1) {
            endLayer = Training.getTotalLayers() - 1;
        }

        int layerNum = endLayer - startLayer + 1;

        // Key: origin layer idx, Value: weights
        Map<String, List<UnifiedFloatTensor> > weightsMap = new java.util.HashMap<>();
        for (int layer = 0; layer < layerNum; layer++) {
            Common.printLog("replicateWeights(): Getting layer: " + layer);
            Object[] weights = getSubModelWeightsByLayer(layer, true);
            List<UnifiedFloatTensor> curWeightTensor = GrpcUtils.convertObjectArrToUnifiedFloatTensorList(weights);

            weightsMap.put(String.valueOf(startLayer + layer), curWeightTensor);
        }


        String res = "";
        if (replicationType == FaultTolerance.ReplicationType.LOCAL_REPLICATION) {
            Common.printLog("Performing local replication");
            int nextIdx = (deviceIdx + 1) % Common.getWorkerNum();
            String targetUrl = Common.getUrlFromWorker(nextIdx);
            res = FaultToleranceGRPCRequest.sendWeightsReplication(targetUrl, weightsMap, FaultTolerance.ReplicationType.LOCAL_REPLICATION);
            Common.printLog("Updating local replication interval ...");
            FaultTolerance.setLocalReplicationInterval(FaultTolerance.getLocalReplicationInterval() * 10);
        } else {
            Common.printLog("Performing global replication");
            int nextIdx = 0;
            String targetUrl = Common.getUrlFromWorker(nextIdx);
            res = FaultToleranceGRPCRequest.sendWeightsReplication(targetUrl, weightsMap, FaultTolerance.ReplicationType.GLOBAL_REPLICATION);
            Common.printLog("Updating global replication interval ...");
            FaultTolerance.setGlobalReplicationInterval(FaultTolerance.getGlobalReplicationInterval() * 10);
        }

        if (res.equals("ok")) {
            Common.printLog("Replicate weights finished: " + res);
        } else {
            Common.printLog("Replicate weights failed: " + res);
        }
    }
}
