//
// Created by Yuhao Chen on 2024/2/25.
//

#ifndef CONFIDANT_MULTIPLICATIONTEST_H
#define CONFIDANT_MULTIPLICATIONTEST_H
#include <string>
#include <map>
#include "ParameterOptimizer.hpp"

using namespace MNN;
using namespace MNN::Train;

namespace Confidant {
    void testMultiplication();
    void testIntermediateMultiplication();
    void testMatMulParallel();
}



#endif //CONFIDANT_MULTIPLICATIONTEST_H