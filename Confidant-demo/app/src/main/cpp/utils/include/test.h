//
// Created by Yuhao Chen on 2023/1/11.
//

#ifndef CONFIDANT_TEST_H
#define CONFIDANT_TEST_H

#include "SGD.hpp"
#include "MNN/expr/Executor.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    void test(std::shared_ptr<Executor> exe, int epoch);
}

#endif //FTPIPEHD_MNN_TEST_H