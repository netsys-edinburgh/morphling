//
// Created by Yuhao Chen on 2023/11/12.
//
#include "faultTolerance.h"
#include "faultToleranceStates.h"
#include "commonStates.h"
#include "model.h"
#include "optimizer.h"

namespace Confidant {
    std::pair<std::vector<VARP>, std::vector<VARP> > retrainBatchWithIterId(int iterId) {
        auto curBatch = FaultToleranceStates::getTrainData(iterId);
        if (curBatch.empty()) {
            return {{}, {}};
        }

        auto example    = curBatch[0];
        auto model = ModelZoo::subModelPtr;
        auto opt = OptimizerWeightVersion::opt;

        int latestVersion = opt->getLatestVersion();
        CommonStates::setWeightVersion(iterId, latestVersion);

        auto output = model->onForward({example.first[0]});

        // we assume that only the gradient of output[0] should be backpropagated
        CommonStates::storeIntermediate(iterId, example.first[0], 0);
        CommonStates::storeIntermediate(iterId, output[0], 1);

        return {output, example.second};
    }
}