//
// Created by Yuhao Chen on 2023/6/5.
//

#include "singleTrain.h"
#include "MNN/expr/Module.hpp"
#include "NN.hpp"
#include "model.h"
#include "SGD.hpp"
#include "Loss.hpp"
#include "LearningRateScheduler.hpp"
#include "log.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "datasets.h"
#include "BERTLayer.h"
#include "GPT2Layer.h"
#include "LLaMALayer.h"

#include "MNN/expr/ExecutorScope.hpp"

#include "LLaMA.h"
#include "GPT2.h"
#include "commonStates.h"
#include "ADAMW.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::Train::Model;

namespace Confidant {
    void singleTrainOneEpoch(int epoch) {
        auto model = ModelZoo::modelPtr;
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        // For defining different backends
//        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
//        std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
//        std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN};
//        std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN, MNN_FORWARD_CPU};
//        for (auto& type : types) {
//            exe->setGlobalExecutorConfig(type, config, 1);
//        }
//        auto ab = exe->getAvailableBackends();
//        for(const auto& pair: ab) {
//            LOGI("Backend: %d NumThread: %d bts: %d", pair.first, pair.second, 8);
//        }

        auto dataLoader = Datasets::trainSetLoader;
        size_t iterations = dataLoader->iterNumber();

        std::shared_ptr<SGD> sgd(new SGD(model));
        sgd->setMomentum(0.9f);
        sgd->setWeightDecay(0.0005f);

        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);

            int lastIndex = 0;
            int moveBatchSize = 0;
            float timeSum = 0.0;

            for (int i = 0; i < iterations; i++) {
                // AUTOTIME
                Timer _100Time;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto labels = example.second;
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                LOGI("Start forwarding...\n");
                Timer _trainTime;
                auto logits = model->onForward({example.first[0]});

                auto loss = model->getLoss(logits, labels);
                auto lossPtr = loss[0]->readMap<float>();
                LOGI("Epoch %d Iter %d Loss: %f\n", epoch, i, lossPtr[0]);

                LOGI("Forward Time: %f ms\n", (float)_trainTime.durationInUs() / 1000.0f);
                _trainTime.reset();

                float rate = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);

                sgd->step(loss[0]);
                LOGI("Epoch %d Iter %d Backward Time: %f ms\n",epoch, i,  (float)_trainTime.durationInUs() / 1000.0f);
                if (i != 0)
                    timeSum += (float)_100Time.durationInUs() / 1000.0f;
                if(i == 9)
                    LOGI("Sum time over 10 batchs: %f ms\n", timeSum/1000);
            }
            LOGI("Average Time over 5 tests: %f ms\n", timeSum/4.0/1000);
        }

        exe->dumpProfile();
    }
}