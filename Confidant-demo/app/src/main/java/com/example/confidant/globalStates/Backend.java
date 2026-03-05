package com.example.confidant.globalStates;

import java.util.Map;

public class Backend {
    public enum MNNForwardType {
        MNN_FORWARD_CPU(0),
        MNN_FORWARD_AUTO(4),
        MNN_FORWARD_METAL(1),
        MNN_FORWARD_CUDA(2),
        MNN_FORWARD_OPENCL(3),
        MNN_FORWARD_OPENGL(6),
        MNN_FORWARD_VULKAN(7),
        MNN_FORWARD_NN(5),
        MNN_FORWARD_USER_0(8),
        MNN_FORWARD_USER_1(9),
        MNN_FORWARD_USER_2(10),
        MNN_FORWARD_USER_3(11);

        private final int value;

        MNNForwardType(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    public static native Map<Integer, Integer> getBackendsMap();
}
