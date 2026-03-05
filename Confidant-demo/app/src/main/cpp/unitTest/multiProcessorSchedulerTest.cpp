//
// Created by Yuhao Chen on 2024/7/18.
//
#include "multiProcessorSchedulerTest.h"
#include "multiProcessorScheduler.h"
#include "MNN/expr/Executor.hpp"
#include "BERTLayer.h"
#include "GPT2Layer.h"
#include "Phi2Layer.h"
#include "LLaMALayer.h"

#include "commonStates.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "log.h"

using namespace MNN;
using namespace MNN::Train::Model;

namespace Confidant {
    void multiProcessorSchedulerTestEntry(std::string& modelName, std::map<std::string, double> &modelArgs) {
        MultiProcessorScheduler::mpsPtr = std::shared_ptr<MultiProcessorScheduler>(new MultiProcessorScheduler());
        MultiProcessorScheduler::mpsPtr->profileProcessors(modelName, modelArgs);
        MultiProcessorScheduler::mpsPtr->computeAllocationStrategy();

//        std::pair<float, std::vector<ProcessorInfo>> allocationStrategy;
//        allocationStrategy.first = 0.0f;
//        allocationStrategy.second.emplace_back(ProcessorInfo(MNN_FORWARD_CPU, 1, 25, ONE_ATTN_HEAD));
//        allocationStrategy.second.emplace_back(ProcessorInfo(MNN_FORWARD_OPENCL, 1, 7, ONE_ATTN_HEAD));
//        MultiProcessorScheduler::mpsPtr->setAllocationStrategy(allocationStrategy);

        if (modelName == "BERTForClassification") {
            BERTMPSTest(modelArgs);
        } else if (modelName == "GPT2") {
            GPT2MPSTest(modelArgs);
        } else if (modelName == "Phi2Alpaca") {
            Phi2MPSTest(modelArgs);
        } else if (modelName == "LLaMALora") {
            LLaMAMPSTest(modelArgs);
        } else {
            LOGI("profileProcessors(): Unsupported model name\n");
        }

    }

    void BERTMPSTest(std::map<std::string, double> &modelArgs) {
        LOGI("Start BERT MPS Test\n");
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        config.precision = MNN::BackendConfig::Precision_High;

        // print allocation strategy
        std::vector<ProcessorInfo> allocationStrategy = MultiProcessorScheduler::mpsPtr->getAllocationStrategy().second;
        for (int i = 0; i < allocationStrategy.size(); i++) {
            LOGI("BERTMPSTest(): Processor %d: %d\n", i, allocationStrategy[i].numAttentionHead);
        }

        std::vector<pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_CPU, 1}};

        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type.first, config, type.second);
        }

        int batchSize = CommonStates::getBatchSize();
        int hiddenSize = (int) modelArgs["hidden_size"];
        int numAttentionHeads = (int) modelArgs["num_attention_heads"];
        int seqLen = 128;

        auto selfAttn = std::make_shared<BERTSelfAttention>(hiddenSize, numAttentionHeads, 0);

        auto parallelSelfAttn = std::make_shared<BERTParallelSelfAttention>(hiddenSize, numAttentionHeads, 0, allocationStrategy);

        int repeatTime = 6;

        float noParallelCPUSum = 0.0f, noParallelGPUSum = 0.0f, parallelSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            exe->gc(Executor::FULL);

            // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
            auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            auto attentionMask = _Const(1.0f, {batchSize, 1, 1, seqLen}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time;
            auto selfAttnOutput = selfAttn->onForward({x, attentionMask})[0];
            auto ptr = selfAttnOutput->readMap<float>();
            // MNN_PRINT("Ptr: %f\n", ptr[0]);
            if (k > 0) {
                noParallelCPUSum += (float)_100Time.durationInUs() / 1000.0f;
            }
            LOGI("BERT No CPU Parallel Time: %f ms\n", (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

            exe->gc(Executor::FULL);
            exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
            auto x2 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            auto attentionMask2 = _Const(1.0f, {batchSize, 1, 1, seqLen}, NCHW);

            // AUTOTIME;
            MNN::Timer _100Time2;
            auto selfAttnOutput2 = selfAttn->onForward({x2, attentionMask2})[0];
            auto ptr2 = selfAttnOutput2->readMap<float>();
            if (k > 0) {
                noParallelGPUSum += (float)_100Time2.durationInUs() / 1000.0f;
            }
            LOGI("BERT No GPU Parallel Time: %f ms\n", (float)_100Time2.durationInUs() / 1000.0f);
            _100Time2.reset();

            exe->gc(Executor::FULL);
            exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
            auto x3 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            auto attentionMask3 = _Const(1.0f, {batchSize, 1, 1, seqLen}, NCHW);

            MNN::Timer _100Time3;
            auto parallelSelfAttnOutput = parallelSelfAttn->onForward({x3, attentionMask3})[0];
            auto ptr3 = parallelSelfAttnOutput->readMap<float>();
            if (k > 0) {
                parallelSum += (float)_100Time3.durationInUs() / 1000.0f;
            }
            LOGI("BERT Parallel Time: %f ms\n", (float)_100Time3.durationInUs() / 1000.0f);
            _100Time3.reset();

//            int diffCnt = 0;
//            auto shape = parallelSelfAttnOutput->getInfo();
//            for (int i = 0; i < shape->size; i++) {
//                if (fabs(ptr[i] - ptr2[i]) >= 1e-5) {
//                    MNN_PRINT("Error: %d %f %f\n", i, ptr[i], ptr2[i]);
//                    diffCnt++;
//                }
//            }
//            MNN_PRINT("Compuation error num: %d/%d\n", diffCnt, shape->size);
        }
        LOGI("BERTMPSTest(): Bts %d: CPU No MPS Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelCPUSum / (repeatTime - 1));
        LOGI("BERTMPSTest(): Bts %d: GPU Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelGPUSum / (repeatTime - 1));
        LOGI("BERTMPSTest(): Bts %d: MPS Average Time over %d tests: %f ms\n", batchSize, repeatTime, parallelSum / (repeatTime - 1));
    }

    void GPT2MPSTest(std::map<std::string, double> &modelArgs) {
        LOGI("Start GPT2 MPS Test\n");
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        config.precision = MNN::BackendConfig::Precision_High;

        // print allocation strategy
        std::vector<ProcessorInfo> allocationStrategy = MultiProcessorScheduler::mpsPtr->getAllocationStrategy().second;
        for (int i = 0; i < allocationStrategy.size(); i++) {
            LOGI("GPT2MPSTest(): Processor %d: %d\n", i, allocationStrategy[i].numAttentionHead);
        }

        std::vector<pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_CPU, 1}};

        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type.first, config, type.second);
        }

        int batchSize = CommonStates::getBatchSize();
        int hiddenSize = (int) modelArgs["hidden_size"];
        int numAttentionHeads = (int) modelArgs["num_attention_heads"];
        int seqLen = 128;

        auto selfAttn = std::make_shared<GPT2SelfAttention>(hiddenSize, numAttentionHeads);

        auto parallelSelfAttn = std::make_shared<GPT2GenericParallelSelfAttention>(hiddenSize, numAttentionHeads, allocationStrategy);

        int repeatTime = 6;

        float noParallelCPUSum = 0.0f, noParallelGPUSum = 0.0f, parallelSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            exe->gc(Executor::FULL);
            auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time;
            auto selfAttnOutput = selfAttn->onForward({x})[0];
            auto ptr = selfAttnOutput->readMap<float>();
            // MNN_PRINT("Ptr: %f\n", ptr[0]);
            if (k > 0) {
                noParallelCPUSum += (float)_100Time.durationInUs() / 1000.0f;
            }

            LOGI("GPT2 CPU No Parallel Time: %f ms\n", (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

            exe->gc(Executor::FULL);
            exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
            auto x2 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time2;
            auto selfAttnOutput2 = selfAttn->onForward({x2})[0];
            auto ptr2 = selfAttnOutput2->readMap<float>();
            // MNN_PRINT("Ptr: %f\n", ptr[0]);
            if (k > 0) {
                noParallelGPUSum += (float)_100Time2.durationInUs() / 1000.0f;
            }

            LOGI("GPT2 GPU No Parallel Time: %f ms\n", (float)_100Time2.durationInUs() / 1000.0f);
            _100Time2.reset();

            exe->gc(Executor::FULL);

            auto x3 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
            MNN::Timer _100Time3;
            auto parallelSelfAttnOutput = parallelSelfAttn->onForward({x3})[0];
            auto ptr3 = parallelSelfAttnOutput->readMap<float>();
            // MNN_PRINT("Ptr2: %f\n", ptr2[0]);
            if (k > 0) {
                parallelSum += (float)_100Time3.durationInUs() / 1000.0f;
            }

            LOGI("GPT2 Parallel Time: %f ms\n", (float)_100Time3.durationInUs() / 1000.0f);
            _100Time3.reset();
        }

        LOGI("GPT2MPSTest(): Bts %d: CPU No MPS Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelCPUSum / (repeatTime - 1));
        LOGI("GPT2MPSTest(): Bts %d: GPU Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelGPUSum / (repeatTime - 1));
        LOGI("GPT2MPSTest(): Bts %d: MPS Average Time over %d tests: %f ms\n", batchSize, repeatTime, parallelSum / (repeatTime - 1));
    }

    void Phi2MPSTest(std::map<std::string, double> &modelArgs) {
        LOGI("Start Phi2 MPS Test\n");
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        config.precision = MNN::BackendConfig::Precision_High;

        // print allocation strategy
        std::vector<ProcessorInfo> allocationStrategy = MultiProcessorScheduler::mpsPtr->getAllocationStrategy().second;
        for (int i = 0; i < allocationStrategy.size(); i++) {
            LOGI("Phi2MPSTest(): Processor %d: %d\n", i, allocationStrategy[i].numAttentionHead);
        }

        std::vector<pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_CPU, 1}};

        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type.first, config, type.second);
        }

        int batchSize = CommonStates::getBatchSize();
        int hiddenSize = (int) modelArgs["hidden_size"];
        int numAttentionHeads = (int) modelArgs["num_attention_heads"];
        int seqLen = 128;

        PhiArgs phiArgs;
        phiArgs.vocabSize = (int)modelArgs["vocab_size"];
        phiArgs.nLayers = (int)modelArgs["num_hidden_layers"];
        phiArgs.hiddenSize = (int)modelArgs["hidden_size"];
        phiArgs.normEps = (float) modelArgs["norm_eps"];
        phiArgs.nHeads = (int) modelArgs["num_attention_heads"];
        phiArgs.nKVHeads = (int) modelArgs["num_kv_heads"];
        phiArgs.maxSeqLen = (int) modelArgs["max_seq_len"];
        phiArgs.maxBatchSize = (int) modelArgs["max_batch_size"];
        phiArgs.intermediateSize = (int)modelArgs["intermediate_size"];

        auto selfAttn = std::make_shared<PhiSelfAttention>(phiArgs);

        auto parallelSelfAttn = std::make_shared<Phi2ParallelSelfAttention>(phiArgs, allocationStrategy);

        int repeatTime = 5;

        float noParallelCPUSum = 0.0f, noParallelGPUSum = 0.0f, parallelSum = 0.0f;

        auto freqsComplex = precomputeThetaPosFrequencies(hiddenSize / numAttentionHeads, seqLen * 2);
        auto freqsComplexShape = freqsComplex->getInfo()->dim;
        auto sliceBegin  = _Input({3}, NCHW);
        auto sliceSize    = _Input({3}, NCHW);
        const int beginData[] = {0, 0, 0};
        memcpy(sliceBegin->writeMap<int>(), beginData, 4 * sizeof(int));
        const int sizeData[] = {2, seqLen, freqsComplexShape[2]};
        memcpy(sliceSize->writeMap<int>(), sizeData, 4 * sizeof(int));
        auto freqs = _Slice(freqsComplex, sliceBegin, sliceSize);

        for (int k = 0; k < repeatTime; k++) {
            exe->gc(Executor::FULL);
            // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
            auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time;
            auto selfAttnOutput = selfAttn->onForward({x, freqs})[0];
            auto ptr = selfAttnOutput->readMap<float>();
            // MNN_PRINT("Ptr: %f\n", ptr[0]);
            if (k > 0) {
                noParallelCPUSum += (float)_100Time.durationInUs() / 1000.0f;
            }

            LOGI("Phi2 CPU No Parallel Time: %f ms\n", (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

            exe->gc(Executor::FULL);
            exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
            auto x2 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time2;
            auto selfAttnOutput2 = selfAttn->onForward({x2, freqs})[0];
            auto ptr2 = selfAttnOutput2->readMap<float>();
            // MNN_PRINT("Ptr: %f\n", ptr[0]);
            if (k > 0) {
                noParallelGPUSum += (float)_100Time2.durationInUs() / 1000.0f;
            }

            LOGI("Phi2 GPU No Parallel Time: %f ms\n", (float)_100Time2.durationInUs() / 1000.0f);
            _100Time.reset();

            exe->gc(Executor::FULL);
            exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
            auto x3 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);

            MNN::Timer _100Time3;
            auto parallelSelfAttnOutput = parallelSelfAttn->onForward({x3, freqs})[0];
            auto ptr3 = parallelSelfAttnOutput->readMap<float>();
            // MNN_PRINT("Ptr2: %f\n", ptr2[0]);

            if (k > 0) {
                parallelSum += (float)_100Time3.durationInUs() / 1000.0f;
            }

            LOGI("Phi2 Parallel Time: %f ms\n", (float)_100Time3.durationInUs() / 1000.0f);
            _100Time3.reset();
        }
        LOGI("Phi2MPSTest(): Bts %d: CPU No MPS Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelCPUSum / (repeatTime - 1));
        LOGI("Phi2MPSTest(): Bts %d: GPU Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelGPUSum / (repeatTime - 1));
        LOGI("Phi2MPSTest(): Bts %d: MPS Average Time over %d tests: %f ms\n", batchSize, repeatTime, parallelSum / (repeatTime - 1));
    }

    void LLaMAMPSTest(std::map<std::string, double> &modelArgs) {
        LOGI("Start LLaMA MPS Test\n");
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        config.precision = MNN::BackendConfig::Precision_High;

        // print allocation strategy
        std::vector<ProcessorInfo> allocationStrategy = MultiProcessorScheduler::mpsPtr->getAllocationStrategy().second;
        for (int i = 0; i < allocationStrategy.size(); i++) {
            LOGI("LLaMAMPSTest(): Processor %d: %d\n", i, allocationStrategy[i].numAttentionHead);
        }

        std::vector<pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_CPU, 1}};

        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type.first, config, type.second);
        }

        int batchSize = CommonStates::getBatchSize();
        int hiddenSize = (int) modelArgs["hidden_size"];
        int numAttentionHeads = (int) modelArgs["num_attention_heads"];
        int seqLen = 128;

        LLaMAArgs llamaArgs;
        llamaArgs.vocabSize = (int)modelArgs["vocab_size"];
        llamaArgs.nLayers = (int)modelArgs["num_hidden_layers"];
        llamaArgs.hiddenSize = (int)modelArgs["hidden_size"];
        llamaArgs.normEps = (float) modelArgs["norm_eps"];
        llamaArgs.nHeads = (int) modelArgs["num_attention_heads"];
        llamaArgs.nKVHeads = (int) modelArgs["num_kv_heads"];
        llamaArgs.maxSeqLen = (int) modelArgs["max_seq_len"];
        llamaArgs.maxBatchSize = (int)modelArgs["max_batch_size"];

        auto selfAttn = std::make_shared<LLaMASelfAttention>(llamaArgs);

        auto parallelSelfAttn = std::make_shared<LLaMAParallelSelfAttention>(llamaArgs, allocationStrategy);

        int repeatTime = 5;

        float noParallelCPUSum = 0.0f, noParallelGPUSum = 0.0f, parallelSum = 0.0f;

        auto freqsComplex = precomputeThetaPosFrequencies(hiddenSize / numAttentionHeads, seqLen * 2);
        auto freqsComplexShape = freqsComplex->getInfo()->dim;
        auto sliceBegin  = _Input({3}, NCHW);
        auto sliceSize    = _Input({3}, NCHW);
        const int beginData[] = {0, 0, 0};
        memcpy(sliceBegin->writeMap<int>(), beginData, 4 * sizeof(int));
        const int sizeData[] = {2, seqLen, freqsComplexShape[2]};
        memcpy(sliceSize->writeMap<int>(), sizeData, 4 * sizeof(int));
        auto freqs = _Slice(freqsComplex, sliceBegin, sliceSize);

        for (int k = 0; k < repeatTime; k++) {
            exe->gc(Executor::FULL);
            auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time;
            auto selfAttnOutput = selfAttn->onForward({x, freqs})[0];
            auto ptr = selfAttnOutput->readMap<float>();
            // MNN_PRINT("Ptr: %f\n", ptr[0]);
            if (k > 0) {
                noParallelCPUSum += (float)_100Time.durationInUs() / 1000.0f;
            }

            LOGI("LLaMA CPU No Parallel Time: %f ms\n", (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

//            exe->gc(Executor::FULL);
//            exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
//            auto x2 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
//            // AUTOTIME;
//            MNN::Timer _100Time2;
//            auto selfAttnOutput2 = selfAttn->onForward({x2, freqs})[0];
//            auto ptr2 = selfAttnOutput2->readMap<float>();
//            // MNN_PRINT("Ptr: %f\n", ptr[0]);
//            if (k > 0) {
//                noParallelGPUSum += (float)_100Time2.durationInUs() / 1000.0f;
//            }
//
//            LOGI("LLaMA GPU No Parallel Time: %f ms\n", (float)_100Time2.durationInUs() / 1000.0f);
//            _100Time2.reset();

            exe->gc(Executor::FULL);
            exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);

            auto x3 = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);

            MNN::Timer _100Time3;
            auto parallelSelfAttnOutput = parallelSelfAttn->onForward({x3, freqs})[0];
            auto ptr3 = parallelSelfAttnOutput->readMap<float>();
            // MNN_PRINT("Ptr2: %f\n", ptr2[0]);

            if (k > 0) {
                parallelSum += (float)_100Time3.durationInUs() / 1000.0f;
            }

            LOGI("LLaMA Parallel Time: %f ms\n", (float)_100Time3.durationInUs() / 1000.0f);
            _100Time3.reset();
        }
        LOGI("LLaMAMPSTest(): Bts %d: CPU Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelCPUSum / (repeatTime - 1));
        LOGI("LLaMAMPSTest(): Bts %d: GPU Average Time over %d tests: %f ms\n", batchSize, repeatTime, noParallelGPUSum / (repeatTime - 1));
        LOGI("LLaMAMPSTest(): Bts %d: MPS Average Time over %d tests: %f ms\n", batchSize, repeatTime, parallelSum / (repeatTime - 1));
    }
}