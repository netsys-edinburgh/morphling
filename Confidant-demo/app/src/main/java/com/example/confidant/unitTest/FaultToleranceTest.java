package com.example.confidant.unitTest;

import android.util.Pair;

import com.example.confidant.faultTolerance.ReplicationUtils;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.FaultTolerance;
import com.example.confidant.globalStates.Training;
import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.utils.General;
import com.example.confidant.utils.GrpcUtils;

import java.util.List;
import java.util.Map;

public class FaultToleranceTest {
    public static void faultToleranceTestEntry() {
        weightsReplicationTest();
//        getModelLayerTest();
    }

    public static void weightsReplicationTest() {
        ReplicationUtils.replicateWeights(FaultTolerance.ReplicationType.LOCAL_REPLICATION);
    }

    public static void getModelLayerTest() {
        int deviceIdx = Common.getDeviceIdx();
        List<Integer> point = Training.getPartitionPoint();
        Pair<Integer, Integer> layerPair = General.getLayerFromPoint(point, deviceIdx);
        int startLayer = layerPair.first, endLayer = layerPair.second;

        if (endLayer == -1) {
            endLayer = Training.getTotalLayers() - 1;
        }

        int layerNum = endLayer - startLayer + 1;

        Map<String, Object[]> weightsMap = new java.util.HashMap<>();
        for (int layer = 0; layer < layerNum; layer++) {
            Common.printLog("replicateWeights(): Getting layer: " + layer);
            Object[] weights = ReplicationUtils.getSubModelWeightsByLayer(layer, true);

            weightsMap.put(String.valueOf(startLayer + layer), weights);
        }
    }
}
