//
// Created by Yuhao Chen on 2024/1/11.
//

#include "train.h"

#include "log.h"
#include "model.h"
#include "Loss.hpp"
#include "datasets.h"
#include "test.h"
#include "commonStates.h"
#include "faultToleranceStates.h"
#include "optimizer.h"
#include "dlfcn.h"

using namespace MNN;
using namespace MNN::Train;
using namespace MNN::Express;

namespace Confidant {
    /*
     * Initialize the variables used in the training
     */
    void initTrain(int batchSize, int deviceNum, int deviceIdx) {
        CommonStates::setBatchSize(batchSize);
        CommonStates::setDeviceNum(deviceNum);
        CommonStates::setDeviceIdx(deviceIdx);
        assert(CommonStates::getDeviceNum() > 0);
        assert(CommonStates::getDeviceIdx() > -1);

        auto exe = Executor::getGlobalExecutor();
        BackendConfig config;
//        exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
    }

    std::pair<std::vector<VARP>, std::vector<VARP> > trainForward(int iterId) {
        // for language model
        auto dataLoader = Datasets::trainSetLoader;
        auto model = ModelZoo::subModelPtr;
        auto opt = OptimizerWeightVersion::opt;

        auto trainData  = dataLoader->next();
        auto example    = trainData[0];
        auto label = example.second;
        // moveBatchSize += example.first[0]->getInfo()->dim[0];

        // Storing the data for fault tolerance
        FaultToleranceStates::storeTrainData(iterId, trainData);

        int latestVersion = opt->getLatestVersion();
        CommonStates::setWeightVersion(iterId, latestVersion);

        auto output = model->onForward({example.first[0]});

        // Here we assume that only the first output should be propagated
        CommonStates::storeIntermediate(iterId, example.first[0], 0);
        CommonStates::storeIntermediate(iterId, output[0], 1);

        return { output, label };
    }

    void trainBackwardCentral(int iterId, VARP grad) {
        std::map<Express::VARP, Express::VARP> gradMap;
        auto opt = OptimizerWeightVersion::opt;

        auto interInput = CommonStates::getIntermediate(iterId, 0);
        auto interOutput = CommonStates::getIntermediate(iterId, 1);

        int curVersion = CommonStates::getWeightVersion(iterId);

        // load the weights with the given version
//        if (!opt->loadVersionWeights(curVersion)) {
//            return ;
//        }

        opt->step(interOutput, interInput, grad, gradMap);

//        LOGI("Learning rate: %f", opt->getLearningRate());

        CommonStates::removeIntermediate(iterId);
    }

    std::vector<VARP> trainBackwardWorker(int iterId, VARP grad) {
        std::map<Express::VARP, Express::VARP> gradMap;
        auto opt = OptimizerWeightVersion::opt;
// interInput, interOutput, grad
        auto interInput = CommonStates::getIntermediate(iterId, 0);
        auto interOutput = CommonStates::getIntermediate(iterId, 1);

        int curVersion = CommonStates::getWeightVersion(iterId);

        // load the weights with the given version
//        if (!opt->loadVersionWeights(curVersion)) {
//            return {};
//        }

        auto inputGrad = opt->step(interOutput, interInput, grad, gradMap);

        CommonStates::removeIntermediate(iterId);

        if (iterId > 0) {
            // Here we just delete the previous train data, in case the dynamic scheduling can not find the data
            FaultToleranceStates::removeTrainData(iterId - 1);
        }

        return inputGrad;
    }

    std::vector<VARP> trainIntermediate(int iterId, std::vector<VARP> intermediates) {
        auto model = ModelZoo::subModelPtr;
        auto opt = OptimizerWeightVersion::opt;

        int latestVersion = opt->getLatestVersion();
        CommonStates::setWeightVersion(iterId, latestVersion);

        auto output = model->onForward(intermediates);

        CommonStates::storeIntermediate(iterId, intermediates[0], 0);
        CommonStates::storeIntermediate(iterId, output[0], 1);
        return output;
    }

    /**
     * Trains the intermediate for the last part, which should calculate the loss and backward the last part
     * @param intermediate
     * @param labels
     * @return
     */
    std::pair<std::vector<VARP>, std::vector<VARP> > trainIntermediateLast(int iterId, std::vector<VARP>& intermediates, std::vector<VARP>& labels) {
        auto model = ModelZoo::subModelPtr;
        auto opt = OptimizerWeightVersion::opt;

        auto logits = model->onForward(intermediates);
        // for bert, the attention mask is in logits[1]
        auto loss = model->getLoss(logits, labels);

        auto loss_backup = loss;
        float lossVal = loss[0]->readMap<float>()[0];
        LOGE("Batch %d Loss: %f", iterId, lossVal);

        // backward
        std::map<Express::VARP, Express::VARP> gradMap;
        auto shape = loss[0]->getInfo();
        auto init= _Const(1.0f, shape->dim, shape->order);

        auto grad = opt->step(loss[0], intermediates[0], init, gradMap);

//        auto loss2 = loss_backup[0]->readMap<float>()[0];
        return {loss_backup, {grad[0]}};
    };


    /**
     * Initialize the training for the current epoch
     */
    void initTrainEpoch() {
        auto dataLoader = Datasets::trainSetLoader;
        auto model = ModelZoo::subModelPtr;

        model->clearCache();

        auto exe = Executor::getGlobalExecutor();
        exe->gc(Executor::FULL);
        exe->resetProfile();

        if (dataLoader) {
            dataLoader->reset();
        }
        model->setIsTraining(true);
    }
}
