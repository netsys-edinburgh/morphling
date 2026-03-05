package com.example.confidant.faultTolerance;

import java.util.List;

public class DeviceCompatibilityInfo {
    private List<Float> computingCapacity;
    private float totalComputingCapacity;

    private int chargeCounter;
    private float compatibility;

    public DeviceCompatibilityInfo(List<Float> computingCapacity, int chargeCounter) {
        this.computingCapacity = computingCapacity;
        this.chargeCounter = chargeCounter;
        this.compatibility = 0;
        this.totalComputingCapacity = -1.0f;
    }

    public float computeCompatibility(float maxComputingCapacity, float minComputingCapacity, int maxChargeCounter, int minChargeCounter, float eta, float p) {
        float cc = 0.0f;
        for (float c : computingCapacity) {
            cc += c;
        }
        float ccNorm = (cc - minComputingCapacity) / (maxComputingCapacity - minComputingCapacity);
        float chargeCounterNorm = (chargeCounter - minChargeCounter) * 1.0f / (maxChargeCounter - minChargeCounter);
        this.compatibility = p * chargeCounterNorm / (ccNorm + eta);
        return this.compatibility;
    }

    public float getTotalComputingCapacity() {
        if (this.totalComputingCapacity == -1.0f) {
            this.totalComputingCapacity = 0.0f;
            for (float c : computingCapacity) {
                this.totalComputingCapacity += c;
            }
        }
        return this.totalComputingCapacity;
    }

    public int getChargeCounter() {
        return this.chargeCounter;
    }
}
