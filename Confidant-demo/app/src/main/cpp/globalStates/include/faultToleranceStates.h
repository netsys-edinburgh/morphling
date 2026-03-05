//
// Created by Yuhao Chen on 2023/11/12.
//

#ifndef CONFIDANT_FAULTTOLERANCESTATES_H
#define CONFIDANT_FAULTTOLERANCESTATES_H

#include "MNN/expr/Module.hpp"
#include "map"
#include "Example.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace std;

namespace Confidant {
    class FaultToleranceStates {
    private:
        static unordered_map<int, std::vector<Train::Example> > dataPool;

    public:
        static void storeTrainData(int iterId, std::vector<Train::Example> data);
        static std::vector<Train::Example> getTrainData(int iterId);
        static void removeTrainData(int iterId);
    };
}

#endif //CONFIDANT_FAULTTOLERANCESTATES_H
