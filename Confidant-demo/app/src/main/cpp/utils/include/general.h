//
// Created by Yuhao Chen on 2023/1/10.
//


#ifndef CONFIDANT_GENERAL_H
#define CONFIDANT_GENERAL_H

#include <jni.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "MNN/expr/Expr.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    std::vector<VARP> cloneParams(std::vector<VARP>& params);
    void copyParameters(std::vector<VARP>& src, std::vector<VARP>& dst);
}

#endif //CONFIDANT_GENERAL_H

