//
// Created by Yuhao Chen on 2023/1/10.
//

#ifndef CONFIDANT_MODEL_H
#define CONFIDANT_MODEL_H

#include <string>
#include "MNN/expr/Module.hpp"
#include "SubModel.h"

using namespace MNN::Express;
namespace Confidant {
    class ModelZoo {
    public:
        static void createModel(std::string& modelName, std::map<std::string, double> &modelArgs);
        static void createSubModel(std::string& modelName, std::map<std::string, double> &modelArgs, int start, int end);
        static std::shared_ptr<SingleModel> modelPtr;
        static std::shared_ptr<SubModel> subModelPtr;
    };
}

#endif //CONFIDANT_MODEL_H
