//
// Created by Yuhao Chen on 2024/2/25.
//

#include "multiplicationTest.h"
#include <MNN/AutoTime.hpp>
#include "MNN/expr/Module.hpp"
#include "NN.hpp"
#include "log.h"
#include "RandomGenerator.hpp"
#include <random>
#include <thread>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

namespace Confidant {
    void testMultiplication() {
        RandomGenerator::generator(17);
        std::random_device gRandom;

        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);

        std::vector<int> batchSizes = {1, 2, 4, 8, 16};
        std::vector<int> seqLens = {128, 256, 512, 1024};
        std::vector<int> hiddenSizes = {768, 1024, 4096};

        int repeatTime = 10;

        for (int batchSize : batchSizes) {
            for (int seqLen : seqLens) {
                for (int hiddenSize : hiddenSizes) {

                    double totalTime = 0.0;
                    for (int i = 0; i < repeatTime; i++) {
//                        int xSize = batchSize * seqLen * hiddenSize, x2Size = batchSize * hiddenSize * hiddenSize;
//                        std::vector<float> xData(xSize), x2Data(x2Size);
//                        for (int i = 0; i < xSize; ++i) {
//                            xData[i] = ((float)(gRandom() % 2000) - 1000.0f) / 1000.0f;
//                        }
//
//                        for (int i = 0; i < x2Size; ++i) {
//                            x2Data[i] = ((float)(gRandom() % 2000) - 1000.0f) / 1000.0f;
//                        }

                        MNN::Timer _timer;
//                        auto x = _Const(xData.data(), {batchSize, seqLen, hiddenSize}, NCHW);
//                        auto x2 = _Const(x2Data.data(), {batchSize, hiddenSize, hiddenSize}, NCHW);
                        auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                        auto x2 = _Const(1.0f, {batchSize, hiddenSize, hiddenSize}, NCHW);

                        auto c = _MatMul(x, x2);
                        auto ptr = c->readMap<float>();
                        LOGI("c: %f\n", ptr[0]);
                        auto time = (float)_timer.durationInUs() / 1000.0f;
                        totalTime += time;
                    }

                    LOGI("BatchSize: %d, SeqLen: %d, HiddenSize: %d, OriginTime: %f\n", batchSize, seqLen, hiddenSize, totalTime / repeatTime);
                }
            }
        }
    }

    void testIntermediateMultiplication() {
        RandomGenerator::generator(17);
        std::random_device gRandom;

        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);

        std::vector<int> batchSizes = {1, 2, 4, 8};
        std::vector<int> seqLens = {128, 256, 512};
//        std::vector<int> hiddenSizes = {768, 1024, 1280,
//                                        4096}; // bert-base gpt2-medium, gpt2-large, llama

        std::vector<int> hiddenSizes = {768, 1024, 2048, 4096};
        int repeatTime = 5;

        for (int batchSize: batchSizes) {
            for (int seqLen: seqLens) {
                for (int hiddenSize: hiddenSizes) {

                    double totalTime = 0.0;
                    for (int i = 0; i < repeatTime; i++) {
                        MNN::Timer _timer;

                        auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                        // here the intermediate hidden size is 4 * hiddenSize
                        auto x2 = _Const(1.0f, {batchSize, hiddenSize, 4 * hiddenSize}, NCHW);

                        auto c = _MatMul(x, x2);
                        auto ptr = c->readMap<float>();

                        auto time = (float) _timer.durationInUs() / 1000.0f;
                        if (ptr[0] == 0.0) {
                            LOGI("testIntermediateMultiplication(): compute failed: BatchSize: %d, SeqLen: %d, HiddenSize: %d\n", batchSize,
                                 seqLen, hiddenSize);
                            time = 1e4;
                        }
                        totalTime += time;
                    }

                    LOGI("testIntermediateMultiplication(): BatchSize: %d, SeqLen: %d, HiddenSize: %d, OriginTime: %f\n", batchSize,
                         seqLen, hiddenSize, totalTime / repeatTime);
                }
            }
        }
    }

    void testMatMulParallel() {
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        int batchSize = 32;
        int seqLen = 1024;
        int hiddenSize = 768;

        MNN::Timer _noParallelTimer;
        MNN::Timer _gpuTimer;
        auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
        auto x2 = _Const(1.0f, {batchSize, hiddenSize, hiddenSize}, NCHW);

        auto gpuC = _MatMul(x, x2);
        Variable::prepareComputeByForwardType({gpuC}, false, {types[0], 1});
        auto gpuPtr = gpuC->readMap<float>();
        auto gpuTime = (float)_gpuTimer.durationInUs() / 1000.0f;
        LOGI("testMatMulParallel(): GPUTime: %f\n", gpuTime);

        MNN::Timer _cpuTimer;
        auto x3 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
        auto x4 = _Const(1.0f, {batchSize, hiddenSize, hiddenSize}, NCHW);

        auto cpuC = _MatMul(x3, x4);
        // Variable::prepareComputeByForwardType({gpuC}, true);
        auto cpuPtr = cpuC->readMap<float>();
        auto cpuTime = (float)_cpuTimer.durationInUs() / 1000.0f;
        LOGI("testMatMulParallel(): CPUTime: %f\n", cpuTime);
        LOGI("testMatMulParallel(): No parallel time %f\n", (float)_noParallelTimer.durationInUs() / 1000.0f);

        // parallel
        MNN::Timer _parallelTimer;
        auto x5 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
        auto x6 = _Const(1.0f, {batchSize, hiddenSize, hiddenSize}, NCHW);

        auto parallelGPUC = _MatMul(x5, x6);
        Variable::prepareComputeByForwardType({parallelGPUC}, false, {types[0], 1});

        auto x7 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
        auto x8 = _Const(1.0f, {batchSize, hiddenSize, hiddenSize}, NCHW);

        auto parallelCPUC = _MatMul(x7, x8);
        std::vector<VARP> outputs = {parallelGPUC, parallelCPUC};

        std::vector<std::thread> threads(0);
        auto executeHelper = [&](int i) {
            MNN::Timer _threadTimer;
            auto ptr = outputs[i]->readMap<float>();
            LOGI("testMatMulParallel(): Current %d: %f, Time %f\n", i, ptr[0], (float)_threadTimer.durationInUs() / 1000.0f);
        };

        for (int i = 0; i < 2; ++i) {
            threads.emplace_back(executeHelper, i);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        auto parallelTime = (float)_parallelTimer.durationInUs() / 1000.0f;
        LOGI("testMatMulParallel(): ParallelTime: %f\n", parallelTime);

        auto paraGpuPtr = outputs[0]->readMap<float>();
        auto paraCpuPtr = outputs[1]->readMap<float>();
        LOGI("testMatMulParallel(): GPUC: %f, CPUC: %f\n", paraGpuPtr[0], paraCpuPtr[0]);
    }
}