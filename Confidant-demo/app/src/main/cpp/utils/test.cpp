//
// Created by Yuhao Chen on 2023/1/11.
//
#include "test.h"
#include "datasets.h"
#include "model.h"
#include "log.h"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    void test(std::shared_ptr<Executor> exe, int epoch) {
//        auto testDataLoader = Datasets::testSetLoader;
//        // auto model = FTPipeHD::Model::modelPtr;
//        auto model = FTPipeHD::Model::subModel1;
//
//        size_t testIterations = testDataLoader->iterNumber();
//        int correct = 0;
//        testDataLoader->reset();
//        model->setIsTraining(false);
//        int moveBatchSize = 0;
//        for (int i = 0; i < testIterations; i++) {
//            auto data       = testDataLoader->next();
//            auto example    = data[0];
//            moveBatchSize += example.first[0]->getInfo()->dim[0];
//            if ((i + 1) % 100 == 0) {
//                LOGE("test: %d / %d", moveBatchSize, testDataLoader->size());
//            }
//            auto cast       = _Cast<float>(example.first[0]);
//            example.first[0] = cast * _Const(1.0f / 255.0f);
//            auto predict    = model->forward(example.first[0]);
//            predict         = _ArgMax(predict, 1);
//            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
//            correct += accu->readMap<int32_t>()[0];
//        }
//        auto accu = (float)correct / (float)testDataLoader->size();
//        LOGE("epoch: %d, accuracy: %f", epoch, accu);
//
//        exe->dumpProfile();
    }
}