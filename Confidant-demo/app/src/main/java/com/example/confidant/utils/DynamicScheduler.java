package com.example.confidant.utils;

import android.net.TrafficStats;

import com.example.confidant.R;
import com.example.confidant.globalStates.Common;
import com.example.confidant.globalStates.Training;
import com.example.confidant.utils.General;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DynamicScheduler {
    private static int deviceNum = Common.getWorkerNum();

    public static List<Integer> calculatePartitionPoint(boolean isAverage) {
        List<Integer> ret = new ArrayList<>();
        int totalLayer = Training.getTotalLayers();
        //int deviceNum = Common.getWorkerNum();
        OfflineProfiler profiler = Common.getOfflineProfiler();

        // Create test data
        Map<String, Float> testBandwidth = new HashMap<>();
        for (int i = 0; i < deviceNum; i++) {
            testBandwidth.put(String.valueOf(i), 50000000f / (8.0f * 1000));
        }

        List<Float> outputSize = profiler.getOutputSize();
        List<Float> bandwidth = profiler.getBandwidth();

        // create an all-zero 2-d arraylist with length totalLayer * deviceNum
        List<List<Float>> transmissionTime = new ArrayList<>();
        for (int i = 0; i < deviceNum; i++) {
            List<Float> tmp = new ArrayList<>();
            for (int j = 0; j < totalLayer; j++) {
                tmp.add(0.0f);
            }
            transmissionTime.add(tmp);
        }

        // enumerate testBandwidth and outputSize
        for (int i = 0; i < deviceNum; i++) {
            for (int j = 0; j < totalLayer; j++) {
                transmissionTime.get(i).set(j, 2 * outputSize.get(j) / testBandwidth.get(String.valueOf(i)));
//                 transmissionTime.get(i).set(j, 2 * outputSize.get(j) / bandwidth.get(i));
            }
        }

        // initialize partition point
        List<List<Integer>> partitionPoint = new ArrayList<>();
        for (int i = 0; i < totalLayer; i++) {
            List<Integer> tmp = new ArrayList<>();
            for (int j = 0; j < deviceNum; j++) {
                tmp.add(0);
            }
            partitionPoint.add(tmp);
        }

        List<List<Float>> dp = new ArrayList<>();
        for (int i = 0; i < totalLayer; i++) {
            List<Float> tmp = new ArrayList<>();
            for (int j = 0; j < deviceNum; j++) {
                tmp.add(Float.POSITIVE_INFINITY);
            }
            dp.add(tmp);
        }

        for (int i = 0; i < totalLayer; i++) {
            dp.get(i).set(0, profiler.getTimeInterval(0, i, 0) + profiler.getTimeInterval(0, i, 1));
        }
        
        for (int device = 1; device < deviceNum; device++) {
            for (int j = device; j < totalLayer - deviceNum + device + 1; j++) {
                for (int i = 0; i < j; i++) {
                    float lastDevTime = profiler.getTimeInterval(i + 1, j, 0) + profiler.getTimeInterval(i + 1, j, 1);
                    if (!isAverage) {
                        lastDevTime *= profiler.getComputingCapacityById(device);
                    }
                    float curTransTime = transmissionTime.get(device - 1).get(i);
                    float slowestTime = Math.max(dp.get(i).get(device - 1), Math.max(curTransTime, lastDevTime));
                    if (slowestTime < dp.get(j).get(device)) {
                        dp.get(j).set(device, slowestTime);
                        partitionPoint.get(j).set(device, i);
                    }
                }
            }
        }

        for (int i = 0; i < deviceNum - 1; i++) {
            ret.add(0);
        }

        int curLayer = totalLayer - 1;
        for (int i = 0; i < deviceNum - 1; i++) {
            int point = partitionPoint.get(curLayer).get(deviceNum - 1 - i);
            ret.set(deviceNum - 2 - i, point);
            curLayer = point;
        }

        return ret;
    }

    public static Float estimateMemory(int submodel, int Num,Map<String, Object> config){
        Float Memory = 8.0f;

        Map<String,Object> args = (Map<String, Object>) config.get("model_args");
        Map<String,Object> data = (Map<String, Object>) config.get("data");
        Double h = (Double) args.get("hidden_size"); //768
        Double v = (Double) args.get("vocab_size"); //30522
        int b = (Integer) data.get("batch_size"); //1
        //Double num_hidden_layers = (Double) args.get("num_hidden_layers"); //12
        Double a = (Double) args.get("num_attention_heads"); //12
        Double hff = (Double) args.get("intermediate_size"); //3072
        Double p = (Double) args.get("parallel_size"); //1
        Double s = (Double) args.get("max_seq_len"); //512
        Double sublayer = (double) submodel;

        double embedding = Num == 1 ? 1.0 : 0.0;
        double end = Num == deviceNum  ? 1.0 : 0.0;
        double node = deviceNum - Num;

        double m1,m2,m3,m4;

        switch(Common.globalModelName){
            case BERT:
                m1 = ( embedding * h * v + (10 * h * h+12 * h)*sublayer + end * 2 * h) *4 * Math.pow(2,-30);  // model
                m2 = (10 * h * h + 12 * h) * sublayer  *(1+3) *4* Math.pow(2,-30);     // optimizer adamw  + grad
                m3 = sublayer *  (16 * s * b * h + 8 * s * b * hff + a * s * s * b)  *Math.pow(2,-30); // activation
                m4 = node*(10 * h * h + 12 * h) * sublayer *4* Math.pow(2,-30);  // optimizer
                Memory = (float)(m1+m2+m3+m4);
                break;
            case GPT2:
                m1 = ( embedding * h * v + (10 * h * h+12 * h)*sublayer + end * 2 * h) *4 * Math.pow(2,-30);  // model
                m2 = (10 * h * h + 12 * h) * sublayer  *(1+3) *4* Math.pow(2,-30);     // optimizer adamw  + grad
                m3 = sublayer *  (16 * s * b * h + 8 * s * b * hff + a * s * s * b)  *Math.pow(2,-30); // activation
                m4 = node*(10 * h * h + 12 * h) * sublayer *4* Math.pow(2,-30);  // optimizer
                Memory = (float)(m1+m2+m3+m4);
                break;
            case Phi2:
                m1 = ( embedding * h * v + (10 * h * h+12 * h)*sublayer + end * 2 * h) *4 * Math.pow(2,-30);  // model
                m2 = (10 * h * h + 12 * h) * sublayer  *(1+3) *4* Math.pow(2,-30);     // optimizer adamw  + grad
                m3 = sublayer *  (16 * s * b * h + 8 * s * b * hff + a * s * s * b)  *Math.pow(2,-30); // activation
                m4 = node*(10 * h * h + 12 * h) * sublayer *4* Math.pow(2,-30);  // optimizer
                Memory = (float)(m1+m2+m3+m4);
                break;
            case LLaMA:
                double kv = (Double) args.get("num_kv_heads");//: 32.0
                //s = (Double) args.get("max_seq_len");//: 512.0
                m1 = ( embedding * v * h  + (10 * h * h+12 * h)*sublayer + end * 2 * h) *4 * Math.pow(2,-30);  // model
                m2 = (10 * h * h + 12 * h*1) * sublayer  *(1+3) *4* Math.pow(2,-30);     // optimizer adamw  + grad
                m3 = sublayer *  (16 * s * b * h + 8 * s * b * hff + a * s * s * b)  *Math.pow(2,-30); // activation
                m4 = node*(10 * h * h + 12 * h) * sublayer *4* Math.pow(2,-30);  // optimizer
                Memory = (float)(m1+m2+m3+m4);
                break;
            default:
                break;
        }
        return Memory;
    }
    public static List<Integer> calculatePartitionPointMemory(boolean isAverage, Map<String, Object> config) {
        List<Integer> ret = new ArrayList<>();
        int totalLayer = Training.getTotalLayers();
        OfflineProfiler profiler = Common.getOfflineProfiler();

        // Create test data
        Map<String, Float> testBandwidth = new HashMap<>();
        for (int i = 0; i < deviceNum; i++) {
            testBandwidth.put(String.valueOf(i), 50000000f / (8.0f * 1000));
        }

        List<Float> outputSize = profiler.getOutputSize();
        List<Float> bandwidth = profiler.getBandwidth();

        // Memory of each devices
        List<Float> MaxMemory = profiler.getAvailableMemory();
        for (int i = 0;i <deviceNum;i++) {
            MaxMemory.set(i,(float)8.0);
        }
        Float inf = Float.POSITIVE_INFINITY;

        // create an all-zero 2-d arraylist with length totalLayer * deviceNum
        List<List<Float>> transmissionTime = new ArrayList<>();
        for (int i = 0; i < deviceNum; i++) {
            List<Float> tmp = new ArrayList<>();
            for (int j = 0; j < totalLayer; j++) {
                tmp.add(0.0f);
            }
            transmissionTime.add(tmp);
        }

        // enumerate testBandwidth and outputSize
        for (int i = 0; i < deviceNum; i++) {
            for (int j = 0; j < totalLayer; j++) {
                transmissionTime.get(i).set(j, 2 * outputSize.get(j) / testBandwidth.get(String.valueOf(i)));
//                 transmissionTime.get(i).set(j, 2 * outputSize.get(j) / bandwidth.get(i));
            }
        }

        // initialize partition point
        List<List<Integer>> partitionPoint = new ArrayList<>();
        for (int i = 0; i < totalLayer; i++) {
            List<Integer> tmp = new ArrayList<>();
            for (int j = 0; j < deviceNum; j++) {
                tmp.add(0);
            }
            partitionPoint.add(tmp);
        }

        List<List<Float>> dp = new ArrayList<>();
        for (int i = 0; i < totalLayer; i++) {
            List<Float> tmp = new ArrayList<>();
            for (int j = 0; j < deviceNum; j++) {
                tmp.add(Float.POSITIVE_INFINITY);
            }
            dp.add(tmp);
        }

        Float Memory = estimateMemory(1,0,config);
        for (int i = 0; i < totalLayer; i++) {
            if (MaxMemory.get(0)>estimateMemory(i+1,1,config)){break;} //Memory limit
            dp.get(i).set(0, profiler.getTimeInterval(0, i, 0) + profiler.getTimeInterval(0, i, 1));
        }

        for (int device = 1; device < deviceNum; device++) {
            for (int j = device; j < totalLayer - deviceNum + device + 1; j++) {
                for (int i = 0; i < j; i++) {
                    if (MaxMemory.get(device)>estimateMemory(i+1,device+1,config)){continue;} //Memory limit


                    float lastDevTime = profiler.getTimeInterval(i + 1, j, 0) + profiler.getTimeInterval(i + 1, j, 1);
                    if (!isAverage) {
                        lastDevTime *= profiler.getComputingCapacityById(device);
                    }
                    float curTransTime = transmissionTime.get(device - 1).get(i);
                    float slowestTime = Math.max(dp.get(i).get(device - 1), Math.max(curTransTime, lastDevTime));
                    if (slowestTime < dp.get(j).get(device)) {
                        dp.get(j).set(device, slowestTime);
                        partitionPoint.get(j).set(device, i);
                    }
                }
            }
        }

        for (int i = 0; i < deviceNum - 1; i++) {
            ret.add(0);
        }

        int curLayer = totalLayer - 1;
        for (int i = 0; i < deviceNum - 1; i++) {
            int point = partitionPoint.get(curLayer).get(deviceNum - 1 - i);
            ret.set(deviceNum - 2 - i, point);
            curLayer = point;
        }

        return ret;
    }
    /*
        Used by proactive fault handler, profile the computing capacity of each device.
     */
//    public static List<Float> profileComputingCapacityVector() {
//        List<Float> ccv = new ArrayList<>();
//
//        // profile the time of computing different number of encoders
//        int N = 5;
//        Object[] ccvObjectArr = Model.profileEncoderTime(Common.getModelName(), Common.getModelArgs(), N);
//
//        // put ccvObjectArr into ccv
//        for (int i = 0; i < N; i++) {
//            ccv.add((Float) ccvObjectArr[i]);
//        }
//
//        Common.setCurrentDeviceCapacity(ccv);
//        return ccv;
//    }
}
