//
//  LearningRateScheduler.hpp
//  MNN
//
//  Created by MNN on 2019/12/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LearningRateScheduler_hpp
#define LearningRateScheduler_hpp

#include <MNN/MNNDefine.h>
#include <vector>

namespace MNN {
namespace Train {

class MNN_PUBLIC LrScheduler {
public:
    static float multiStep(const float baseLr, const int step, std::vector<int> stepIterations,
                           std::vector<float> lrMulti);

    static float inv(const float baseLr, const int step, const float gamma, const float power);

    static float exp(const float baseLr, const int step, const float gamma);

    // added by CYH
    static float linear(const float baseLr, const int step, const int numTrainStep, const int numWarmupStep = 0);
};

} // namespace Train
} // namespace MNN

#endif // LearningRateScheduler_hpp
