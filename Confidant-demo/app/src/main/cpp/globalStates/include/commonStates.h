//
// Created by Yuhao Chen on 2023/2/24.
//

#ifndef CONFIDANT_COMMONSTATES_H
#define CONFIDANT_COMMONSTATES_H

#include "map"
#include "MNN/expr/Module.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace std;

namespace Confidant {
    class CommonStates {
    public:
        enum MNNDataType {
            MNN_FLOAT = 0,
            MNN_INT = 1,
            MNN_NOT_YET_SET = 2
        };

        enum ModelName {
            BERT = 0,
            GPT2 = 1,
            LLaMA = 2,
            Phi2 = 3,
        };

    private:
        static int batchSize;
        static int deviceNum;
        static int deviceIdx;
        static string modelWeightsPath;
        static string modelDatasetPath;
        static std::map<int, std::vector<VARP> > intermediatePool;
        static std::map<int, int> idToWeight;
        static ModelName globalModelName;

    public:
        static void setBatchSize(int _batchSize);
        static int getBatchSize();
        static void storeIntermediate(int iterId, VARP data, int type);
        static void removeIntermediate(int iterId);
        static VARP getIntermediate(int iterId, int type);

        static void setWeightVersion(int iterId, int version);
        static int getWeightVersion(int iterId);

        static void setDeviceNum(int deviceNum);

        static void setDeviceIdx(int deviceIdx);

        static int getDeviceNum();

        static int getDeviceIdx();

        static void setModelWeightsPath(const string &modelWeightsPath);

        static void setModelDatasetPath(const string &modelDatasetPath);

        static const string &getModelWeightsPath();

        static const string &getModelDatasetPath();

        static void setGlobalModelName(ModelName modelName);
        static ModelName getGlobalModelName();
    };
}



#endif //CONFIDANT_COMMONSTATES_H
