//
// Created by Yuhao Chen on 2023/11/24.
//

#ifndef CONFIDANT_PROFILER_H
#define CONFIDANT_PROFILER_H

#include "BERTLayer.h"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    std::vector<float> profileEncoders(std::string& modelName, std::map<std::string, double> &modelArgs, int numEncoders);
}

#endif //CONFIDANT_PROFILER_H
