//
//  LearningRateScheduler.cpp
//  MNN
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "LearningRateScheduler.hpp"
#include <algorithm>
#include <cmath>

namespace MNN {
namespace Train {

float LrScheduler::multiStep(const float baseLr, const int step, std::vector<int> stepIterations,
                             std::vector<float> lrMulti) {
    float lr = baseLr;
    std::sort(stepIterations.begin(), stepIterations.end());
    for (int i = 0; i < stepIterations.size(); i++) {
        if (step == stepIterations[i]) {
            float multi;
            if ((i + 1) > lrMulti.size()) {
                multi = lrMulti[lrMulti.size() - 1];
            } else {
                multi = lrMulti[i];
            }
            lr *= multi;
        }
    }

    return lr;
}

float LrScheduler::inv(const float baseLr, const int step, const float gamma, const float power) {
    float lr = baseLr * std::pow(1 + gamma * step, -power);
    return lr;
}

float LrScheduler::exp(const float baseLr, const int step, const float gamma) {
    float lr = baseLr * std::pow(gamma, step);
    return lr;
}

float LrScheduler::linear(const float baseLr, const int step, const int numTrainStep, const int numWarmupStep) {
    float lr = baseLr;
    if (step < numWarmupStep) {
        lr = baseLr * (float)(step + 1) / (float)(numWarmupStep + 1);
    } else {
        auto temp = std::max((float)0.0, (float)(numTrainStep - step) / (float)(std::max(1, numTrainStep - numWarmupStep)));
        lr = baseLr * std::max((float)0.0, (float)(numTrainStep - step) / (float)(std::max(1, numTrainStep - numWarmupStep)));
    }
    return lr;
}

} // namespace Train
} // namespace MNN
