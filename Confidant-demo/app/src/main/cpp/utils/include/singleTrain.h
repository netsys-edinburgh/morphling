//
// Created by Yuhao Chen on 2023/6/5.
//

#ifndef CONFIDANT_SINGLETRAIN_H
#define CONFIDANT_SINGLETRAIN_H

#include "MNN/expr/Module.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    void singleTrainOneEpoch(int epoch);
}

#endif //CONFIDANT_SINGLETRAIN_H
