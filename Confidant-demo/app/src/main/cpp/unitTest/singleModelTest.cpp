//
// Created by Yuhao Chen on 2024/2/28.
//
#include "singleModelTest.h"
#include "MNN/expr/Module.hpp"
#include "RandomGenerator.hpp"
#include "BERT.h"
#include "BERTLayer.h"
#include "LLaMALayer.h"
#include "GPT2.h"
#include "log.h"
#include <MNN/AutoTime.hpp>
#include "ModelArgs.h"
#include "datasets.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::Train::Model;

namespace Confidant {
    void BERTSelfAttnTest() {
        std::string functionName = "BERTSelfAttnTest";
        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        // config.precision = MNN::BackendConfig::Precision_High;
        // std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN, MNN_FORWARD_CPU};
        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
        // exe->setGlobalExecutorConfig(MNN_FORWARD_METAL, config, 1);

        //TODO: TODO use here as flag to find them quickly
        int bts = 8;
        int heads = 12;
        int seqLen = 128;
        int hiddenSize = 768;

        auto selfAttn = std::make_shared<BERTSelfAttention>(hiddenSize, heads, 0);

        std::vector<Confidant::ProcessorInfo> allocationStrategy;
        allocationStrategy.emplace_back(MNN_FORWARD_CPU, 1, 6, Confidant::ONE_ATTN_HEAD);
        allocationStrategy.emplace_back(MNN_FORWARD_OPENCL, 1, 6, Confidant::ONE_ATTN_HEAD);
        auto parallelSelfAttn = std::make_shared<BERTParallelSelfAttention>(hiddenSize, heads, 0, allocationStrategy);

        int repeatTime = 5;

        float noParallelSum = 0.0f, parallelSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            // exe->gc(Executor::FULL);
            auto x2 = _Const(1.0f, {bts, seqLen, hiddenSize}, NCHW);
            auto attentionMask2 = _Const(1.0f, {bts, 1, 1, seqLen}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time;
            auto selfAttnOutput = selfAttn->onForward({x2, attentionMask2})[0];
            auto ptr = selfAttnOutput->readMap<float>();
            MNN_PRINT("Ptr: %f\n", ptr[0]);
            noParallelSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("%s(): Test %d No Parallel Time: %f ms\n", functionName.c_str(), k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

            exe->gc(Executor::FULL);
            auto x3 = _Const(1.0f, {bts, seqLen, hiddenSize}, NCHW);
            auto attentionMask3 = _Const(1.0f, {bts, 1, 1, seqLen}, NCHW);

            MNN::Timer _100Time2;
            auto parallelSelfAttnOutput = parallelSelfAttn->onForward({x3, attentionMask3})[0];
            auto ptr2 = parallelSelfAttnOutput->readMap<float>();
            MNN_PRINT("Ptr2: %f\n", ptr2[0]);
            parallelSum += (float)_100Time2.durationInUs() / 1000.0f;
            LOGI("%s(): Parallel Time: %f ms\n", functionName.c_str(), (float)_100Time2.durationInUs() / 1000.0f);
            _100Time2.reset();

            int diffCnt = 0;
            auto shape = parallelSelfAttnOutput->getInfo();
            for (int i = 0; i < shape->size; i++) {
                if (fabs(ptr[i] - ptr2[i]) >= 1e-5) {
                    MNN_PRINT("Error: %d %f %f\n", i, ptr[i], ptr2[i]);
                    diffCnt++;
                }
            }
            MNN_PRINT("Compuation error num: %d/%d\n", diffCnt, shape->size);
        }
        LOGI("%s(): No Parallel Average Time over %d tests: %f ms\n", functionName.c_str(), repeatTime, noParallelSum / repeatTime);

        LOGI("%s: Parallel Average Time over %d tests: %f ms\n", functionName.c_str(), repeatTime, parallelSum / repeatTime);

    }

    void smallChunkBERTSelfAttnTest() {
        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        // config.precision = MNN::BackendConfig::Precision_High;
        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};

        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        //TODO: TODO use here as flag to find them quickly
        int bts = 8;
        int heads = 12;
        int seqLen = 256;
        int hiddenSize = 768;

        auto selfAttn = std::make_shared<BERTSelfAttention>(768, heads, 0);

        int repeatTime = 5;

        float timeSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            // exe->gc(Executor::FULL);
            MNN::Timer _100Time;
            auto x = _Const(1.0f, {bts, seqLen, hiddenSize}, NCHW);
            std::vector<std::pair<MNNForwardType, int>> types;
            types.push_back(std::make_pair(MNN_FORWARD_OPENCL, 1));
            types.push_back(std::make_pair(MNN_FORWARD_CPU, 1));

            auto splitedX = _Split(x, {2}, 0);

            auto attentionMask = _Const(1.0f, {bts, 1, 1, seqLen}, NCHW);
            auto splitedAttentionMask = _Split(attentionMask, {2}, 0);

            auto selfAttnOutput1 = selfAttn->onForward({splitedX[0], splitedAttentionMask[0]})[0];
            auto selfAttnOutput2 = selfAttn->onForward({splitedX[1], splitedAttentionMask[1]})[0];

             Variable::prepareComputeParallel({selfAttnOutput1, selfAttnOutput2}, true, types);
             auto output = _Concat({selfAttnOutput1, selfAttnOutput2}, 0);
             auto ptr = output->readMap<float>();

            LOGI("Ptr: %f\n", ptr[0]);
            timeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("smallChunkBERTSelfAttnTest(): Test %d Time: %f ms\n", k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

        }
        LOGI("smallChunkBERTSelfAttnTest(): Average Time over %d tests: %f ms\n", repeatTime, timeSum / repeatTime);
    }

    void LLaMASelfAttnTest() {
        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        // config.precision = MNN::BackendConfig::Precision_High;
        // std::vector<MNNForwardType> types = {MNN_FORWARD_METAL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_METAL};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
        // exe->setGlobalExecutorConfig(MNN_FORWARD_METAL, config, 1);

        //TODO: TODO use here as flag to find them quickly
        auto bts = 1;
        auto heads = 12;

        int vocabSize = -1;
        int nLayers = 1;
        int hiddenSize = 4096; // 4096
        float normEps = 1e-5;
        int nHeads = 32;
        int nKVHeads = -1;
        int maxSeqLen = 256; // 1024
        int maxBatchSize = 8;
        int multipleOf = 256;
        int ffnDimMultiplier = -1;
        std::string tokenizerPath = "/Users/yuhaochen/Desktop/NESC/Edge Large Model Training/data/vocab_files/llama_vocab.txt";;
        bool forParallel = false;

        LLaMAArgs args = {vocabSize, nLayers, hiddenSize, normEps, nHeads, nKVHeads, maxSeqLen, maxBatchSize, multipleOf, ffnDimMultiplier, tokenizerPath, forParallel};
        args.loraArgs.r = 8;
        args.loraArgs.enableLoRA = {true, true, true};
        args.loraArgs.alpha = 1.0f;
        args.loraArgs.dropout = 0.0f;

        auto selfAttn = std::make_shared<LLaMASelfAttention>(args);

        // auto parallelSelfAttn = std::make_shared<BERTSelfAttention>(768, heads, 0, true);
        // auto parallelSelfAttn = std::make_shared<ParallelSelfAttention>(768, heads, 0, 12);

        int repeatTime = 1;

        float noParallelSum = 0.0f, parallelSum = 0.0f;

        auto freqsComplex = precomputeThetaPosFrequencies(args.hiddenSize / args.nHeads, args.maxSeqLen * 2);
        auto freqsComplexShape = freqsComplex->getInfo()->dim;
        auto sliceBegin  = _Input({3}, NCHW);
        auto sliceSize    = _Input({3}, NCHW);
        const int beginData[] = {0, 0, 0};
        memcpy(sliceBegin->writeMap<int>(), beginData, 4 * sizeof(int));
        const int sizeData[] = {2, maxSeqLen, freqsComplexShape[2]};
        memcpy(sliceSize->writeMap<int>(), sizeData, 4 * sizeof(int));
        auto freqs = _Slice(freqsComplex, sliceBegin, sliceSize);

        for (int k = 0; k < repeatTime; k++) {
            // exe->gc(Executor::FULL);
            auto x2 = _Const(1.0f, {bts, maxSeqLen, hiddenSize}, NCHW);

            // AUTOTIME;
            MNN::Timer _100Time;
            auto selfAttnOutput = selfAttn->onForward({x2, freqs})[0];
            auto ptr = selfAttnOutput->readMap<float>();
            MNN_PRINT("Ptr: %f\n", ptr[0]);
            noParallelSum += (float)_100Time.durationInUs() / 1000.0f;
            MNN_PRINT("No Parallel Time: %f ms\n", (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

//            exe->gc(Executor::FULL);
//            auto x3 = _Const(1.0f, {bts, 180, 768}, NCHW);
//            auto attentionMask3 = _Const(1.0f, {bts, 1, 1, 180}, NCHW);
//
//            MNN::Timer _100Time2;
//            auto parallelSelfAttnOutput = parallelSelfAttn->onForward({x3, attentionMask3})[0];
//            auto ptr2 = parallelSelfAttnOutput->readMap<float>();
//            MNN_PRINT("Ptr2: %f\n", ptr2[0]);
//            parallelSum += (float)_100Time2.durationInUs() / 1000.0f;
//            MNN_PRINT("Parallel Time: %f ms\n", (float)_100Time2.durationInUs() / 1000.0f);
//            _100Time2.reset();

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
        MNN_PRINT("No Parallel Average Time over %d tests: %f ms\n", repeatTime, noParallelSum / repeatTime);

        MNN_PRINT("Parallel Average Time over %d tests: %f ms\n", repeatTime, parallelSum / repeatTime);

    }

    void smallChunkBERTTest() {
        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        std::vector<MNNForwardType> types = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        int vocabSize = 30522;
        int numHiddenLayers = 12;
        int hiddenSize = 768;
        int intermediateSize = 3072;
        int numAttentionHeads = 12;
        float attnDropout = 0.0;
        float hiddenDropout = 0.0;

        std::shared_ptr<BERTForClassification> model1(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                               false));

        std::shared_ptr<BERTForClassification> model2(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                                false));

        auto dataLoader = Datasets::trainSetLoader;

        auto trainData1  = dataLoader->next();
        auto example    = trainData1[0];
        auto inputIds1 = example.first[0];

        int repeatTime = 5;
        float noSplitTimeSum = 0.0f, splitTimeSum = 0.0f;

//        for (int k = 0; k < repeatTime; k++) {
//            MNN::Timer _100Time;
//            auto output1 = model1->onForward({inputIds1});
//            auto ptr = output1[0]->readMap<float>();
//            noSplitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
//            LOGI("smallChunkBERTTest(): No split Test %d Time: %f ms\n", k, (float)_100Time.durationInUs() / 1000.0f);
//            _100Time.reset();
//        }

        LOGI("smallChunkBERTTest(): No split Average Time over %d tests: %f ms\n", repeatTime, noSplitTimeSum / repeatTime);

        auto trainData2  = dataLoader->next();
        auto example2    = trainData2[0];
        auto inputIds2 = example2.first[0];

        for (int k = 0; k < repeatTime; k++) {
            MNN::Timer _100Time;
            auto splitInput = _Split(inputIds2, {2}, 0);
            auto output2_1 = model2->onForward({_Clone(splitInput[0], true)});
            auto output2_2 = model2->onForward({_Clone(splitInput[1], true)});

            std::vector<std::pair<MNNForwardType, int>> types;
            types.push_back(std::make_pair(MNN_FORWARD_CPU, 1));
            types.push_back(std::make_pair(MNN_FORWARD_OPENCL, 1));

            Variable::prepareComputeParallel({output2_1[0], output2_2[0]}, true, types);
            auto output2 = _Concat({output2_1[0], output2_2[0]}, 0);
            auto ptr = output2->readMap<float>();
            splitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("smallChunkBERTTest(): Split Test %d Time: %f ms\n", k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();
            
        }

        LOGI("smallChunkBERTTest(): Split Average Time over %d tests: %f ms\n", repeatTime, splitTimeSum / repeatTime);
    }

    void smallChunkBERTNewImplTest() {
        // using new GPU parallel implementation

        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL};
        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        int vocabSize = 30522;
        int numHiddenLayers = 12;
        int hiddenSize = 768;
        int intermediateSize = 3072;
        int numAttentionHeads = 12;
        float attnDropout = 0.0;
        float hiddenDropout = 0.0;

        std::shared_ptr<BERTForClassification> model1(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                                false));

        std::shared_ptr<BERTForClassification> model2(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                                false));

        std::shared_ptr<BERTForClassification> model3(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                                false));

        auto dataLoader = Datasets::trainSetLoader;

        auto trainData1  = dataLoader->next();
        auto example    = trainData1[0];
        auto inputIds1 = example.first[0];

        int repeatTime = 5;
        float noSplitTimeSum = 0.0f, splitTimeSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            MNN::Timer _100Time;
            auto output1 = model1->onForward({inputIds1});
            auto ptr = output1[0]->readMap<float>();
            noSplitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("smallChunkBERTNewImplTest(): No split Test %d Time: %f ms\n", k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();
        }

        LOGI("smallChunkBERTNewImplTest(): No split Average Time over %d tests: %f ms\n", repeatTime, noSplitTimeSum / repeatTime);

        auto trainData2  = dataLoader->next();
        auto example2    = trainData2[0];
        // auto inputIds2 = example2.first[0];
        auto inputIds2 = _Clone(inputIds1, true);

        for (int k = 0; k < repeatTime; k++) {
            MNN::Timer _100Time;
            auto splitInput = _Split(inputIds2, {2}, 0);
            auto splitedInput1 = _Clone(splitInput[0], true);
            auto splitedInput2 = _Clone(splitInput[1], true);

            auto output2_1 = model2->onForward({splitedInput1})[0];
            auto output2_2 = model3->onForward({splitedInput2})[0];

            Variable::prepareComputeByForwardType({output2_2}, false, {types[0], 1});

            std::vector<VARP> outputs = {output2_1, output2_2};
            std::vector<std::thread> threads(0);
            auto executeHelper = [&](int i) {
                MNN::Timer _threadTime;
                auto ptr = outputs[i]->readMap<float>();
                // LOGI("smallChunkBERTNewImplTest(): Current %d, Val %f, Time: %f\n", i, ptr[0], (float)_threadTime.durationInUs() / 1000.0f);
                // LOGI("smallChunkBERTNewImplTest(): Current %d: %f\n", i, ptr[0]);
            };

            for (int i = 0; i < 2; ++i) {
                threads.emplace_back(executeHelper, i);
            }

            for (auto& thread : threads) {
                thread.join();
            }

            auto output2 = _Concat(outputs, 0);
            auto ptr = output2->readMap<float>();
            splitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("smallChunkBERTNewImplTest(): Split Test %d Time: %f ms\n", k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

        }

        LOGI("smallChunkBERTNewImplTest(): Split Average Time over %d tests: %f ms\n", repeatTime, splitTimeSum / repeatTime);
    }

    void singleBERTParallelTest() {
        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL};
        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        int vocabSize = 30522;
        int numHiddenLayers = 12;
        int hiddenSize = 768;
        int intermediateSize = 3072;
        int numAttentionHeads = 12;
        float attnDropout = 0.0;
        float hiddenDropout = 0.0;

        std::shared_ptr<BERTForClassification> model(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                                false));

        std::shared_ptr<BERTForClassification> paraModel(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                               true));

        auto dataLoader = Datasets::trainSetLoader;

        auto trainData1  = dataLoader->next();
        auto example    = trainData1[0];
        auto inputIds1 = example.first[0];

        int repeatTime = 5;
        float noSplitTimeSum = 0.0f, splitTimeSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            MNN::Timer _100Time;
            auto output1 = model->onForward({inputIds1});
            auto ptr = output1[0]->readMap<float>();
            noSplitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("singleBERTParallelTest(): No split Test %d Time: %f ms\n", k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();
        }

        LOGI("singleBERTParallelTest(): No split Average Time over %d tests: %f ms\n", repeatTime, noSplitTimeSum / repeatTime);

        auto trainData2  = dataLoader->next();
        auto example2    = trainData2[0];
        // auto inputIds2 = example2.first[0];
        auto inputIds2 = _Clone(inputIds1, true);

        for (int k = 0; k < repeatTime; k++) {
            MNN::Timer _100Time;

            auto output2 = paraModel->onForward({inputIds2})[0];
            auto ptr = output2->readMap<float>();
            splitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("singleBERTParallelTest(): Split Test %d Time: %f ms\n", k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();
        }

        LOGI("singleBERTParallelTest(): Split Average Time over %d tests: %f ms\n", repeatTime, splitTimeSum / repeatTime);
    }

    void singleBERTTest() {
        std::string functionName = "singleBERTTest";
        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN};
        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        int vocabSize = 30522;
        int numHiddenLayers = 12;
        int hiddenSize = 768;
        int intermediateSize = 3072;
        int numAttentionHeads = 12;
        float attnDropout = 0.0;
        float hiddenDropout = 0.0;

        std::shared_ptr<BERTForClassification> model(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                               false));

        auto dataLoader = Datasets::trainSetLoader;

        auto trainData1  = dataLoader->next();
        auto example    = trainData1[0];
        auto inputIds1 = example.first[0];

        int repeatTime = 1;
        float noSplitTimeSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            MNN::Timer _100Time;
            auto output1 = model->onForward({inputIds1});
            auto ptr = output1[0]->readMap<float>();
            for (int i = 0; i < output1[0]->getInfo()->size; i++) {
                LOGI("singleBERTTest(): Current %d: %f\n", i, ptr[i]);
            }
            noSplitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("%s(): No split Test %d Time: %f ms\n", functionName.c_str(),  k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();
        }

        LOGI("%s(): No split Average Time over %d tests: %f ms\n", functionName.c_str(), repeatTime, noSplitTimeSum / repeatTime);
    }

    void singleGPT2Test() {
        std::string functionName = "singleGPT2Test";
        RandomGenerator::generator(17);
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_VULKAN};
        std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        GPT2Args gpt2Args;
        gpt2Args.vocabSize = 50257;
        gpt2Args.maxPositionEmbeddings = 1024;
        gpt2Args.numHiddenLayers = 24;
        gpt2Args.numAttentionHeads = 16;
        gpt2Args.hiddenSize = 1024;
        gpt2Args.intermediateSize = 3072;
        gpt2Args.dropoutProb = 0.0;

        std::shared_ptr<GPT2QAModel> model(new GPT2QAModel(gpt2Args));

        auto dataLoader = Datasets::trainSetLoader;

        auto trainData1  = dataLoader->next();
        auto example    = trainData1[0];
        auto inputIds1 = example.first[0];

        int repeatTime = 1;
        float noSplitTimeSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            MNN::Timer _100Time;
            auto output1 = model->onForward({inputIds1});
            auto ptr = output1[0]->readMap<float>();
            noSplitTimeSum += (float)_100Time.durationInUs() / 1000.0f;
            LOGI("%s(): No split Test %d Time: %f ms\n", functionName.c_str(),  k, (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();
        }

        LOGI("%s(): No split Average Time over %d tests: %f ms\n", functionName.c_str(), repeatTime, noSplitTimeSum / repeatTime);
    }
}