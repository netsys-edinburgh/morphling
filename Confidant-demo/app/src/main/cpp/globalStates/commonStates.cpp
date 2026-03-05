//
// Created by Yuhao Chen on 2023/2/24.
//

#include "commonStates.h"

using namespace MNN;
using namespace MNN::Express;
using namespace std;

namespace Confidant {
    int CommonStates::batchSize = 0;
    int CommonStates::deviceNum = 0;
    int CommonStates::deviceIdx = -1;
    string CommonStates::modelWeightsPath;
    string CommonStates::modelDatasetPath;

    CommonStates::ModelName CommonStates::globalModelName = CommonStates::ModelName::BERT;

    std::map<int, std::vector<VARP> > CommonStates::intermediatePool{};
    std::map<int, int> CommonStates::idToWeight{};

    void CommonStates::setBatchSize(int _batchSize) {
        batchSize = _batchSize;
    }

    int CommonStates::getBatchSize() {
        return batchSize;
    }

    void CommonStates::storeIntermediate(int iterId, VARP data, int type) {
        if (intermediatePool.find(iterId) == intermediatePool.end()) {
            intermediatePool[iterId] = std::vector<VARP>(2);
        }

        intermediatePool[iterId][type] = data;
    }

    VARP CommonStates::getIntermediate(int iterId, int type) {
        return intermediatePool[iterId][type];
    }

    void CommonStates::removeIntermediate(int iterId) {
        intermediatePool.erase(iterId);
    }

    void CommonStates::setWeightVersion(int iterId, int version) {
        idToWeight[iterId] = version;
    }

    int CommonStates::getWeightVersion(int iterId) {
        return idToWeight[iterId];
    }

    int CommonStates::getDeviceNum() {
        return deviceNum;
    }

    int CommonStates::getDeviceIdx() {
        return deviceIdx;
    }

    void CommonStates::setDeviceNum(int _deviceNum) {
        CommonStates::deviceNum = _deviceNum;
    }

    void CommonStates::setDeviceIdx(int _deviceIdx) {
        CommonStates::deviceIdx = _deviceIdx;
    }

    void CommonStates::setModelWeightsPath(const string &_modelWeightsPath) {
        CommonStates::modelWeightsPath = _modelWeightsPath;
    }

    void CommonStates::setModelDatasetPath(const string &_modelDatasetPath) {
        CommonStates::modelDatasetPath = _modelDatasetPath;
    }

    const string &CommonStates::getModelWeightsPath() {
        return modelWeightsPath;
    }

    const string &CommonStates::getModelDatasetPath() {
        return modelDatasetPath;
    }

    void CommonStates::setGlobalModelName(CommonStates::ModelName modelName) {
        globalModelName = modelName;
    }

    CommonStates::ModelName CommonStates::getGlobalModelName() {
        return globalModelName;
    }
}