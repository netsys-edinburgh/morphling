package com.example.confidant.unitTest;

import com.example.confidant.globalStates.Common;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class CrossFrameworkAdapterTest {
    static {
        System.loadLibrary("confidant");
    }

    public static native void convertObjectIntoVARPTest(Object[] data);
    public static native Object[] convertVARPIntoObjectTest(int batchSize, int seqLen, int hiddenSize);
    public static native void dataConversionOverheadTest(int batchSize, int seqLen, int hiddenSize);

    public static void crossFrameworkAdapterEntry() {
        Common.printLog("Cross Framework Adapter Test Entry");

        String modelName = Common.getModelName();
        Map<String, Double> modelArgs = Common.getModelArgs();

//        convertObjectIntoVARPTest(modelName, modelArgs);
//        convertVARPIntoObjectTest(modelName, modelArgs);

        dataConversionOverheadTest(modelName, modelArgs);
    }

    public static void convertObjectIntoVARPTest(String name, Map<String, Double> args) {
        int batchSize = 4;
        int seqLen = 128;
        int hiddenSize = args.get("hidden_size").intValue();

        // put it into Object[]
        Object[] inputObj = new Object[4];

        // generate a random value [0, 1] List<Float> with size batchSize * seqLen * hiddenSize
        List<Float> data = new ArrayList<>();
        for (int i = 0; i < batchSize * seqLen * hiddenSize; i++) {
            data.add((float) Math.random());
        }

        List<Integer> dataShape = new ArrayList<>();
        dataShape.add(batchSize);
        dataShape.add(seqLen);
        dataShape.add(hiddenSize);

        inputObj[0] = data;
        inputObj[1] = dataShape;
        inputObj[2] = 2; // order
        inputObj[3] = 0; // dataType

        // convert it into VARP
        Date startDate = new Date();
        long startTime = startDate.getTime();

        convertObjectIntoVARPTest(inputObj);

        Date endDate = new Date();
        long endTime = endDate.getTime();

        Common.printLog("Convert Object into VARP Time: " + (endTime - startTime) + "ms");
    }

    public static void convertVARPIntoObjectTest(String name, Map<String, Double> args) {
        int batchSize = 4;
        int seqLen = 128;
        int hiddenSize = args.get("hidden_size").intValue();

        // convert it into VARP
        Date startDate = new Date();
        long startTime = startDate.getTime();

        Object[] dataObj = convertVARPIntoObjectTest(batchSize, seqLen, hiddenSize);

        Date endDate = new Date();
        long endTime = endDate.getTime();

        Common.printLog("Convert VARP into Object Time: " + (endTime - startTime) + "ms");
    }

    public static void dataConversionOverheadTest(String name, Map<String, Double> args) {
        int batchSize = 4;
        int seqLen = 128;
        int hiddenSize = args.get("hidden_size").intValue();

        dataConversionOverheadTest(batchSize, seqLen, hiddenSize);
    }
}
