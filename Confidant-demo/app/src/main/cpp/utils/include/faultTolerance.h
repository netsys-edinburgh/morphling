//
// Created by Yuhao Chen on 2023/11/12.
//

#ifndef CONFIDANT_FAULTTOLERANCE_H
#define CONFIDANT_FAULTTOLERANCE_H
#include <vector>
#include "MNN/expr/Module.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    std::pair<std::vector<VARP>, std::vector<VARP> > retrainBatchWithIterId(int iterId);
}
#endif //CONFIDANT_FAULTTOLERANCE_H
