//
// Created by Yuhao Chen on 2023/2/23.
//

#include "include/optimizer.h"
 #include "SGD.hpp"
#include "ADAMW.hpp"
#include "model.h"
#include "log.h"
#include "general.h"
#include "commonStates.h"
#include "BERT.h"
#include "GPT2.h"
#include "LLaMA.h"
#include "Phi2.h"

using namespace MNN;
using namespace MNN::Train;
using namespace MNN::Train::Model;

namespace Confidant {
    std::shared_ptr<OptimizerWeightVersion> OptimizerWeightVersion::opt = nullptr;

    void OptimizerWeightVersion::initOptimizer(std::string &optName,
                                                   std::map<std::string, double> &optArgs) {
        auto modelPtr = ModelZoo::subModelPtr;
        if (optName == "SGD") {
            baseOpt = std::shared_ptr<SGD>(new MNN::Train::SGD(modelPtr));
            std::shared_ptr<SGD> sgdOptPtr = std::static_pointer_cast<SGD>(baseOpt);
            sgdOptPtr->setMomentum(optArgs["momentum"]);
            sgdOptPtr->setWeightDecay(optArgs["weight_decay"]);
            sgdOptPtr->setLearningRate(optArgs["lr"]);
        } else if (optName == "ADAMW") {
            baseOpt = std::shared_ptr<ADAMW>(new MNN::Train::ADAMW(modelPtr));
            std::shared_ptr<ADAMW> adamOptPtr = std::static_pointer_cast<ADAMW>(baseOpt);
            adamOptPtr->setLearningRate(optArgs["lr"]);
        } else {
            LOGE("Optimizer not supported!");
        }
    }

    void OptimizerWeightVersion::initWeightPool() {
        weightPool.clear();
        latestVersion = 0;
        addWeight();
    }

    OptimizerWeightVersion::OptimizerWeightVersion(std::string &optName,
                                                        std::map<std::string, double> &optArgs) {
        initOptimizer(optName, optArgs);
        initWeightPool();
        latestVersion = 0;
        batchCounter = 0;
        updateInterval = -1;
    }

    void OptimizerWeightVersion::addWeight() {
        // TODO: Add the weights to the pool
        auto weightMaxNum = CommonStates::getDeviceNum() - CommonStates::getDeviceIdx();
        if (weightPool.size() > weightMaxNum){
            weightPool.erase(latestVersion -  weightMaxNum);
        }
        assert(weightPool.size() <= weightMaxNum);

        auto modelPtr = ModelZoo::subModelPtr;
//        CommonStates::ModelName globalModelName = CommonStates::getGlobalModelName();
//        if (globalModelName == CommonStates::ModelName::BERT) {
//            modelPtr = std::dynamic_pointer_cast<SubBERTForClassification>(ModelZoo::subModelPtr);
//        } else if (globalModelName == CommonStates::ModelName::GPT2) {
//            modelPtr = std::dynamic_pointer_cast<SubGPT2>(ModelZoo::subModelPtr);
//        } else if (globalModelName == CommonStates::ModelName::LLaMA) {
//            modelPtr = std::dynamic_pointer_cast<SubLLaMA>(ModelZoo::subModelPtr);
//        } else if (globalModelName == CommonStates::ModelName::Phi2) {
//            modelPtr = std::dynamic_pointer_cast<SubPhi2>(ModelZoo::subModelPtr);
//        }
//        auto modelPtr = std::dynamic_pointer_cast<SubBERTForClassification>(ModelZoo::subModelPtr);
//        auto modelPtr = std::dynamic_pointer_cast<SubLLaMA>(ModelZoo::subModelPtr);
//        auto modelPtr = std::dynamic_pointer_cast<SubPhi2>(ModelZoo::subModelPtr);
        // auto modelPtr = std::dynamic_pointer_cast<SubGPT2>(ModelZoo::subModelPtr);
        // auto modelPtr = std::dynamic_pointer_cast<SubLLaMA>(ModelZoo::subModelPtr);
        if (CommonStates::getDeviceIdx() == 0) {
            if (modelPtr->layers.size() > 1) {
                // more than embedding, save weight
                std::vector<VARP> params;
                for (int i = 1; i < modelPtr->layers.size(); i++) {
                    auto layer = modelPtr->layers[i];
                    auto layerParams = layer->parameters();
                    for (int j = 0; j < layerParams.size(); j++) {
                        params.push_back(layerParams[j]);
                    }
                }
                auto copiedParams = cloneParams(params);
                weightPool[latestVersion] = copiedParams;
            }
        } else{
            auto params = modelPtr->parameters();
            // TODO: clone of params?
            auto copiedParams = cloneParams(params);
            weightPool[latestVersion] = copiedParams;
        }

    }

    double OptimizerWeightVersion::getLearningRate() {
        std::shared_ptr<SGD> sgdOptPtr = std::static_pointer_cast<SGD>(baseOpt);
        return sgdOptPtr->currentLearningRate();
    }

    void OptimizerWeightVersion::setLearningRate(double lr) {
        std::shared_ptr<SGD> sgdOptPtr = std::static_pointer_cast<SGD>(baseOpt);
        sgdOptPtr->setLearningRate(lr);
    }

    void OptimizerWeightVersion::resetOptimizer() {
        initWeightPool();
        batchCounter = 0;
    }

    std::vector<Express::VARP> OptimizerWeightVersion::step(Express::VARP loss, Express::VARP input, Express::VARP lossGrad, std::map<Express::VARP, Express::VARP>& gradMap) {
        // TODO: SGD only
        std::shared_ptr<SGD> sgdOptPtr = std::static_pointer_cast<SGD>(baseOpt);
        auto grad = sgdOptPtr->backward(loss, input, lossGrad, gradMap);

        sgdOptPtr->stepNew(gradMap);

        latestVersion += 1;
        addWeight();

        batchCounter += 1;
        return grad;
    }

    int OptimizerWeightVersion::getLatestVersion() {
        return latestVersion;
    }

    bool OptimizerWeightVersion:: loadVersionWeights(int version) {
        if (weightPool.find(version) == weightPool.end()) {
            LOGE("Version number invalid! Latest version: %d, version: %d", latestVersion, version);
            return false;
        }

        if (version == latestVersion) {
            LOGE("The target version equals to the latest version");
            return true;
        }

        CommonStates::ModelName globalModelName = CommonStates::getGlobalModelName();
        if (CommonStates::getDeviceIdx() == 0) {
            auto curWeights = weightPool[version];
            if (globalModelName == CommonStates::ModelName::BERT) {
                auto modelPtr = std::dynamic_pointer_cast<SubBERTForClassification>(ModelZoo::subModelPtr);
                auto embeddingParams = modelPtr->layers[0]->parameters();
                curWeights.insert(curWeights.begin(), embeddingParams.begin(), embeddingParams.end());
                modelPtr->loadParameters(curWeights);
            } else if (globalModelName == CommonStates::ModelName::GPT2) {
                auto modelPtr = std::dynamic_pointer_cast<SubGPT2>(ModelZoo::subModelPtr);
                auto embeddingParams = modelPtr->layers[0]->parameters();
                curWeights.insert(curWeights.begin(), embeddingParams.begin(), embeddingParams.end());
                modelPtr->loadParameters(curWeights);
            } else if (globalModelName == CommonStates::ModelName::LLaMA) {
                auto modelPtr = std::dynamic_pointer_cast<SubLLaMA>(ModelZoo::subModelPtr);
                auto embeddingParams = modelPtr->layers[0]->parameters();
                curWeights.insert(curWeights.begin(), embeddingParams.begin(), embeddingParams.end());
                modelPtr->loadParameters(curWeights);
            } else if (globalModelName == CommonStates::ModelName::Phi2) {
                auto modelPtr = std::dynamic_pointer_cast<SubPhi2>(ModelZoo::subModelPtr);
                auto embeddingParams = modelPtr->layers[0]->parameters();
                curWeights.insert(curWeights.begin(), embeddingParams.begin(), embeddingParams.end());
                modelPtr->loadParameters(curWeights);
            }

//            auto modelPtr = std::dynamic_pointer_cast<SubBERTForClassification>(ModelZoo::subModelPtr);
            //auto modelPtr = std::dynamic_pointer_cast<SubLLaMA>(ModelZoo::subModelPtr);
//            auto modelPtr = std::dynamic_pointer_cast<SubPhi2>(ModelZoo::subModelPtr);
//            auto embeddingParams = modelPtr->layers[0]->parameters();
//            curWeights.insert(curWeights.begin(), embeddingParams.begin(), embeddingParams.end());
//            modelPtr->loadParameters(curWeights);
        } else {
            auto curWeights = weightPool[version];
            auto modelPtr = ModelZoo::subModelPtr;
            modelPtr->loadParameters(curWeights);
        }

        return true;
    }
}