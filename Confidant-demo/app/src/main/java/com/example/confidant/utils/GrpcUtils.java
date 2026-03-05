package com.example.confidant.utils;

import com.example.confidant.rpc.api.UnifiedFloatTensor;
import com.example.confidant.rpc.api.UnifiedIntTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class GrpcUtils {
    public static Object[] convertUnifiedIntTensorToObjectArr(UnifiedIntTensor tensor) {
        List<Integer> data = tensor.getDataList();
        // int[] data = tensor.getDataList().stream().mapToInt(Integer::intValue).toArray();
        int dataType = tensor.getDataType();
        List<Integer> dataShape = tensor.getDataShapeList();

        Object[] objArr = new Object[4];
        objArr[0] = data;
        objArr[1] = dataShape;
        objArr[2] = 2; // order set to 2
        objArr[3] = dataType;

        return objArr;
    }

    public static Object[] convertUnifiedIntTensorListToObjectArr(List<UnifiedIntTensor> tensor) {
        Object[] objArr = new Object[tensor.size() * 4];
        for (int i = 0; i < tensor.size(); i++) {
            List<Integer> dataList = tensor.get(i).getDataList();
            int[] dataArray = new int[dataList.size()];
            for (int j = 0; j < dataList.size(); j++) {
                dataArray[j] = dataList.get(j);
            }

//            List<Integer> dataArray = tensor.get(i).getDataList();
            int dataType = tensor.get(i).getDataType();
            List<Integer> dataShape = tensor.get(i).getDataShapeList();

            objArr[i * 4] = dataArray;
            objArr[i * 4 + 1] = dataShape;
            objArr[i * 4 + 2] = 2; // order set to 2
            objArr[i * 4 + 3] = dataType;
        }

        return objArr;
    }

    public static List<UnifiedIntTensor> convertObjectArrToUnifiedIntTensorList(Object[] objArr) {
        List<UnifiedIntTensor> tensorList = new ArrayList<>();
        for (int i = 0; i < objArr.length; i += 4) {
            List<Integer> data = new ArrayList<>();
            int[] dataArr = (int[]) objArr[i];
            for (int j = 0; j < dataArr.length; j++) {
                data.add(dataArr[j]);
            }

            List<Integer> dataShape = (List<Integer>) objArr[i + 1];
            int order = (int) objArr[i + 2];
            int dataType = (int) objArr[i + 3];

            UnifiedIntTensor tensor = UnifiedIntTensor.newBuilder()
                    .addAllData(data)
                    .setDataType(dataType)
                    .addAllDataShape(dataShape)
                    .build();
            tensorList.add(tensor);
        }

        return tensorList;
    }

    public static Object[] convertUnifiedFloatTensorListToObjectArr(List<UnifiedFloatTensor> tensor) {
        Object[] objArr = new Object[tensor.size() * 4];
        for (int i = 0; i < tensor.size(); i++) {
            List<Float> dataList = tensor.get(i).getDataList();
            float[] dataArray = new float[dataList.size()];
            for (int j = 0; j < dataList.size(); j++) {
                dataArray[j] = dataList.get(j);
            }

//            List<Float> dataArray = tensor.get(i).getDataList();
            int dataType = tensor.get(i).getDataType();
            List<Integer> dataShape = tensor.get(i).getDataShapeList();

            objArr[i * 4] = dataArray;
            objArr[i * 4 + 1] = dataShape;
            objArr[i * 4 + 2] = 2; // order set to 2
            objArr[i * 4 + 3] = dataType;
        }

        return objArr;
    }

    public static List<UnifiedFloatTensor> convertObjectArrToUnifiedFloatTensorList(Object[] objArr) {
        List<UnifiedFloatTensor> tensorList = new ArrayList<>();
        for (int i = 0; i < objArr.length; i += 4) {
            // convert float[] objectArr[i] into List<Float>
            float[] dataArr = (float[]) objArr[i];
            List<Float> data = new ArrayList<>();

            for (int j = 0; j < dataArr.length; j++) {
                data.add(dataArr[j]);
            }

            List<Integer> dataShape = (List<Integer>) objArr[i + 1];
            int order = (int) objArr[i + 2];
            int dataType = (int) objArr[i + 3];

            UnifiedFloatTensor tensor = UnifiedFloatTensor.newBuilder()
                    .addAllData(data)
                    .setDataType(dataType)
                    .addAllDataShape(dataShape)
                    .build();
            tensorList.add(tensor);
        }

        return tensorList;
    }
}
