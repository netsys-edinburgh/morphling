//
// Created by Yuhao Chen on 2024/1/11.
//

#ifndef CONFIDANT_TRAIN_H
#define CONFIDANT_TRAIN_H

#include <string>
#include "MNN/expr/Module.hpp"
#include "LearningRateScheduler.hpp"
#include "SGD.hpp"
#include "MNN/expr/Executor.hpp"

using namespace MNN;
using namespace MNN::Train;
using namespace MNN::Express;

namespace Confidant {
    void initTrain(int batchSize, int deviceNum, int deviceIdx);
    void initTrainEpoch();
    std::pair<std::vector<VARP>, std::vector<VARP> > trainForward(int iterId);
    std::vector<VARP> trainIntermediate(int iterId, std::vector<VARP> intermediate);
    std::pair<std::vector<VARP>, std::vector<VARP> > trainIntermediateLast(int iterId, std::vector<VARP>& intermediate, std::vector<VARP>& labels);

    void trainBackwardCentral(int iterId, VARP grad);
    std::vector<VARP> trainBackwardWorker(int iterId, VARP grad);
}

#endif //CONFIDANT_TRAIN_H