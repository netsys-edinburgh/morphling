//
// Created by 陈宇豪 on 2023/11/12.
//
#include "faultToleranceStates.h"

using namespace MNN;
using namespace MNN::Express;
using namespace std;

namespace Confidant {
    unordered_map<int, std::vector<Train::Example> > FaultToleranceStates::dataPool{};

    void FaultToleranceStates::storeTrainData(int iterId, std::vector<Train::Example> data) {
        dataPool[iterId] = data;
    }

    std::vector<Train::Example> FaultToleranceStates::getTrainData(int iterId) {
        if (dataPool.find(iterId) == dataPool.end()) {
            return std::vector<Train::Example>();
        }
        return dataPool[iterId];
    }

    void FaultToleranceStates::removeTrainData(int iterId) {
        dataPool.erase(iterId);
    }

}