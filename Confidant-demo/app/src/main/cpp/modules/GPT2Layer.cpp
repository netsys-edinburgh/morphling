//
// Created by Yuhao Chen on 2023/6/13.
//

#include "GPT2Layer.h"
#include "MLayerNorm.h"
#include <memory>
#include <thread>

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            GPT2SelfAttention::GPT2SelfAttention(int hiddenSize, int numAttentionHeads) {
                this->numAttentionHeads = numAttentionHeads;
                this->attentionHeadSize = hiddenSize / numAttentionHeads;//768/12=64
                this->allHeadSize = numAttentionHeads * attentionHeadSize;//12*64=768

                query.reset(NN::Linear(hiddenSize, allHeadSize, true));
                query->setName("query");
                key.reset(NN::Linear(hiddenSize, allHeadSize, true));
                key->setName("key");
                value.reset(NN::Linear(hiddenSize, allHeadSize, true));
                value->setName("value");

                registerModel({query, key, value});
            }

            //on forward for Attention
            std::vector<Express::VARP> GPT2SelfAttention::onForward(const std::vector<Express::VARP> &inputs ) {
                using namespace Express;
                VARP x = inputs[0];

                auto mixedQueryLayer = query->forward(x);
                auto mixedKeyLayer = key->forward(x);
                auto mixedValueLayer = value->forward(x);

                auto shape = mixedKeyLayer->getInfo()->dim;// 拿到了维度信息
                auto queryLayer = _Permute(_Reshape(mixedQueryLayer, {shape[0], shape[1], numAttentionHeads, attentionHeadSize}), {0, 2, 1, 3});
                auto keyLayer = _Permute(_Reshape(mixedKeyLayer, {shape[0], shape[1], numAttentionHeads, attentionHeadSize}), {0, 2, 1, 3});
                auto valueLayer = _Permute(_Reshape(mixedValueLayer, {shape[0], shape[1], numAttentionHeads, attentionHeadSize}), {0, 2, 1, 3});

                // start
                auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 1, 3, 2}));//change
                attentionScores = _Divide(attentionScores, _Const((float) sqrt(attentionHeadSize), {}, NCHW));
                auto dim = attentionScores->getInfo()->dim;
                auto nd = dim[dim.size()-2];
                auto ns = dim[dim.size()-1];
                std::vector<float> padIdVec(ns*nd, 0.0f);
                for (int i = 0; i < nd; i++) {
                    for (int j = 0; j <= i + ns - nd; j++) {
                        padIdVec[i * ns + j] = 1.0f;
                    }
                }
                VARP self_bias = _Const(padIdVec.data(), {1, 1,nd, ns}, NCHW, halide_type_of<float>());
                auto attention_bias = attentionScores * self_bias - _Const(1e10f)*(_Const(1.0f) - self_bias);

                // Normalize the attention scores to probabilities.
                auto attentionProbs = _Softmax(attention_bias, -1);

                auto contextLayer = _MatMul(attentionProbs, valueLayer);
                contextLayer = _Permute(contextLayer, {0, 2, 1, 3});

                auto newContextLayerShape = contextLayer->getInfo()->dim;
                contextLayer = _Reshape(contextLayer, {newContextLayerShape[0], newContextLayerShape[1], allHeadSize});

                Variable::prepareCompute({contextLayer});
                return {contextLayer};
            }

            GPT2ParallelSelfAttention::GPT2ParallelSelfAttention(int hiddenSize,
                                                                 int numAttentionHeads,
                                                                 int firstPart) {
                this->numAttentionHeads = numAttentionHeads;
                this->attentionHeadSize = hiddenSize / numAttentionHeads;
                this->allHeadSize = numAttentionHeads * attentionHeadSize;
                this->firstPart = firstPart;

                query.reset(NN::Linear(hiddenSize, allHeadSize, true));
                key.reset(NN::Linear(hiddenSize, allHeadSize, true));
                value.reset(NN::Linear(hiddenSize, allHeadSize, true));

                parallelQuery.resize(numAttentionHeads);
                parallelKey.resize(numAttentionHeads);
                parallelValue.resize(numAttentionHeads);

                for (int i = 0; i < numAttentionHeads; i++) {
                    parallelQuery[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelQuery[i]->setName("query" + std::to_string(i));

                    parallelKey[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelKey[i]->setName("key" + std::to_string(i));

                    parallelValue[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelValue[i]->setName("value" + std::to_string(i));
                }

                this->allHeadSize = numAttentionHeads * attentionHeadSize;

                auto exe = Executor::getGlobalExecutor();
                auto availableBackends = exe->getAvailableBackends();

                if (availableBackends.size() > 1) {
                    concatKey.reset(NN::Linear(hiddenSize, attentionHeadSize * firstPart, true));
                    concatQuery.reset(NN::Linear(hiddenSize, attentionHeadSize * firstPart, true));
                    concatValue.reset(NN::Linear(hiddenSize, attentionHeadSize * firstPart, true));
                    registerModel({concatKey, concatQuery, concatValue});
                    registerModel(std::vector<std::shared_ptr<Module>>(parallelKey.begin() + firstPart, parallelKey.end()));
                    registerModel(std::vector<std::shared_ptr<Module>>(parallelQuery.begin() + firstPart, parallelQuery.end()));
                    registerModel(std::vector<std::shared_ptr<Module>>(parallelValue.begin() + firstPart, parallelValue.end()));
                } else {
                    registerModel(parallelKey);
                    registerModel(parallelQuery);
                    registerModel(parallelValue);
                }
            }

            std::vector<Express::VARP> GPT2ParallelSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                std::call_once(mOnceFlag, [&]() {
                    auto keyParams = key->parameters();
                    auto queryParams = query->parameters();
                    auto valueParams = value->parameters();

                    auto queryBiasParams = _Split(queryParams[0], {numAttentionHeads}, 1);
                    auto queryWeightParams = _Split(queryParams[1], {numAttentionHeads}, 0);
                    auto keyBiasParams = _Split(keyParams[0], {numAttentionHeads}, 1);
                    auto keyWeightParams = _Split(keyParams[1], {numAttentionHeads}, 0);
                    auto valueBiasParams = _Split(valueParams[0], {numAttentionHeads}, 1);
                    auto valueWeightParams = _Split(valueParams[1], {numAttentionHeads}, 0);

                    for (int i = 0; i < numAttentionHeads; i++) {
                        parallelKey[i]->loadParameters(
                                {_Clone(keyBiasParams[i], true), _Clone(keyWeightParams[i], true)});
                        parallelQuery[i]->loadParameters({_Clone(queryBiasParams[i], true),
                                                          _Clone(queryWeightParams[i], true)});
                        parallelValue[i]->loadParameters({_Clone(valueBiasParams[i], true),
                                                          _Clone(valueWeightParams[i], true)});
                    }

                    auto exe = Executor::getGlobalExecutor();
                    auto availableBackends = exe->getAvailableBackends();
                    if (availableBackends.size() > 1) {
                        auto concatKeyBiasParams = _Concat(std::vector<VARP>(keyBiasParams.begin(),
                                                                             keyBiasParams.begin() +
                                                                             firstPart), 1);
                        auto concatKeyWeightParams = _Concat(
                                std::vector<VARP>(keyWeightParams.begin(),
                                                  keyWeightParams.begin() + firstPart), 0);

                        concatKey->loadParameters({_Clone(concatKeyBiasParams, true),
                                                   _Clone(concatKeyWeightParams, true)});

                        auto concatQueryBiasParams = _Concat(
                                std::vector<VARP>(queryBiasParams.begin(),
                                                  queryBiasParams.begin() + firstPart), 1);
                        auto concatQueryWeightParams = _Concat(
                                std::vector<VARP>(queryWeightParams.begin(),
                                                  queryWeightParams.begin() + firstPart), 0);
                        concatQuery->loadParameters({_Clone(concatQueryBiasParams, true),
                                                     _Clone(concatQueryWeightParams, true)});

                        auto concatValueBiasParams = _Concat(
                                std::vector<VARP>(valueBiasParams.begin(),
                                                  valueBiasParams.begin() + firstPart), 1);
                        auto concatValueWeightParams = _Concat(
                                std::vector<VARP>(valueWeightParams.begin(),
                                                  valueWeightParams.begin() + firstPart), 0);
                        concatValue->loadParameters({_Clone(concatValueBiasParams, true),
                                                     _Clone(concatValueWeightParams, true)});
                    }
                });

                VARP x = inputs[0];
                std::vector<VARP> allReduced;

                auto exe = Executor::getGlobalExecutor();
                auto availableBackends = exe->getAvailableBackends();

                if (availableBackends.size() > 1) {
                    MNN_PRINT("Using multiple backends\n");
                    std::vector<VARP> multipleAllReduced;
                    std::vector<std::pair<MNNForwardType, int>> types;

                    // concat part
                    auto concatXClone = _Clone(x, true); // 在这里会调用readMap
                    auto concatQueryLayer = concatQuery->forward(concatXClone);
                    auto concatKeyLayer = concatKey->forward(concatXClone);
                    auto concatValueLayer = concatValue->forward(concatXClone);

                    auto shape = concatKeyLayer->getInfo()->dim;// 拿到了维度信息
                    concatQueryLayer = _Permute(_Reshape(concatQueryLayer, {shape[0], shape[1], firstPart, shape[2]/firstPart}), {0, 2, 1, 3});
                    concatKeyLayer = _Permute(_Reshape(concatKeyLayer, {shape[0], shape[1], firstPart, shape[2]/firstPart}), {0, 2, 1, 3});
                    concatValueLayer = _Permute(_Reshape(concatValueLayer, {shape[0], shape[1], firstPart, shape[2]/firstPart}), {0, 2, 1, 3});

                    // start
                    auto attentionScores = _MatMul(concatQueryLayer, _Transpose(concatKeyLayer, {0, 1, 3, 2}));//change
                    attentionScores = _Divide(attentionScores, _Const((float) sqrt(attentionHeadSize), {}, NCHW));
                    auto dim = attentionScores->getInfo()->dim;
                    auto nd = dim[dim.size()-2];
                    auto ns = dim[dim.size()-1];
                    std::vector<float> padIdVec(ns*nd, 0.0f);
                    for (int i = 0; i < nd; i++) {
                        for (int j = 0; j <= i + ns - nd; j++) {
                            padIdVec[i * ns + j] = 1.0f;
                        }
                    }
                    VARP self_bias = _Const(padIdVec.data(), {1, 1,nd, ns}, NCHW, halide_type_of<float>());
                    auto attention_bias = attentionScores * self_bias - _Const(1e10f)*(_Const(1.0f) - self_bias);

                    // Normalize the attention scores to probabilities.
                    auto attentionProbs = _Softmax(attention_bias, -1);

                    auto concatContextLayer = _MatMul(attentionProbs, concatValueLayer);
                    concatContextLayer = _Permute(concatContextLayer, {0, 2, 1, 3});

                    auto newContextLayerShape = concatContextLayer->getInfo()->dim;
                    concatContextLayer = _Reshape(concatContextLayer, {newContextLayerShape[0], newContextLayerShape[1], newContextLayerShape[2]*newContextLayerShape[3]});

                    multipleAllReduced.emplace_back(concatContextLayer);

                    for (auto &backendType: availableBackends) {
                        if (backendType.first != MNN_FORWARD_CPU) {
                            MNN_PRINT(
                                    "Parallel Using first backend %d with %d attention heads on it\n",
                                    backendType.first, firstPart);
                            types.emplace_back(backendType);
                            break;
                        }
                    }

                    for (int i = firstPart; i < numAttentionHeads; ++i) {
                        auto xClone = _Clone(x, true);
                        auto queryLayer = parallelKey[i]->forward(xClone);
                        auto keyLayer = parallelKey[i]->forward(xClone);
                        auto valueLayer = parallelValue[i]->forward(xClone);

                        auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                        auto dim = attentionScores->getInfo()->dim;
                        auto nd = dim[dim.size()-2];
                        auto ns = dim[dim.size()-1];
                        std::vector<float> padIdVec(ns*nd, 0.0f);
                        for (int i = 0; i < nd; i++) {
                            for (int j = 0; j <= i + ns - nd; j++) {
                                padIdVec[i * ns + j] = 1.0f;
                            }
                        }

                        VARP self_bias = _Const(padIdVec.data(), {1, nd, ns}, NCHW, halide_type_of<float>());
                        auto attention_bias = attentionScores * self_bias - _Const(1e10f)*(_Const(1.0f) - self_bias);

                        auto attentionProbs = _Softmax(attentionScores, -1);

                        auto contextLayer = _MatMul(attentionProbs, valueLayer);

                        multipleAllReduced.emplace_back(_Unsqueeze(contextLayer, {1}));
                        types.emplace_back(MNN_FORWARD_CPU, 1); // cpu type
                    }
                    Variable::prepareComputeParallel(multipleAllReduced, true, types);
                    auto splitedConcatContextLayer = _Split(concatContextLayer, {firstPart}, 2);
                    for (int i = 0; i < firstPart; ++i) {
                        allReduced.emplace_back(_Unsqueeze(splitedConcatContextLayer[i], {1}));
                    }
                    for (int i = 1; i < multipleAllReduced.size(); ++i) {
                        allReduced.emplace_back(multipleAllReduced[i]);
                    }
                } else {
                    for (int i = 0; i < numAttentionHeads; ++i) {
                        auto xClone = _Clone(x, true); // 在这里会调用readMap
                        auto queryLayer = parallelKey[i]->forward(xClone);
                        auto keyLayer = parallelKey[i]->forward(xClone);
                        auto valueLayer = parallelValue[i]->forward(xClone);

                        auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));

                        // Normalize the attention scores to probabilities.
                        auto attentionProbs = _Softmax(attentionScores, -1);

                        auto contextLayer = _MatMul(attentionProbs, valueLayer);

                        allReduced.emplace_back(_Unsqueeze(contextLayer, {1}));
                    }
                    Variable::prepareComputeParallel(allReduced);
                }

                auto contextLayers = _Concat(allReduced, 1);
                contextLayers = _Permute(contextLayers, {0, 2, 1, 3});
                auto newContextLayerShape = contextLayers->getInfo()->dim;
                contextLayers = _Reshape(contextLayers, {newContextLayerShape[0], newContextLayerShape[1], allHeadSize});
                return {contextLayers};
            }



            GPT2Intermediate::GPT2Intermediate(int hiddenSize, int intermediateSize) {
                dense1.reset(NN::Linear(hiddenSize, intermediateSize, true));
                dense2.reset(NN::Linear(intermediateSize, hiddenSize, true));
                registerModel({dense1,dense2});
            }

            std::vector<Express::VARP> GPT2Intermediate::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];

                auto hiddenStates = dense1->forward(x);
                //auto geluPhi = _Const(0.5f) * (_Const(1.0f) + _Erf(hiddenStates / _Const(sqrt(2.0))));//change
                auto geluPhi = _Const(0.5f) * (_Const(1.0f) + _Tanh(_Const(0.7978845608028654f) * (hiddenStates + _Const(0.044715f) * _Pow(hiddenStates, _Const(3.0f)))));
                hiddenStates = hiddenStates * geluPhi;
                auto dense2Output = dense2->forward(hiddenStates);

                return {dense2Output};
            }

            //changed by gsw
            GPT2Output::GPT2Output(int hiddenSize, int intermediateSize) {
                layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-5));//12  ->5
                intermediate = std::make_shared<GPT2Intermediate>(hiddenSize, intermediateSize);

                registerModel({ layerNorm, intermediate});
            }

            std::vector<Express::VARP> GPT2Output::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];

                auto hiddenStates = layerNorm->forward(x);
                auto intermediateOutput = intermediate->onForward({hiddenStates});
                auto output = intermediateOutput[0] + x;

                return {output};
            }

            GPT2Attention::GPT2Attention(int hiddenSize, int numAttentionHeads, bool forParallel) {
                if (forParallel) {
                    std::vector<Confidant::ProcessorInfo> allocationStrategy;
                    allocationStrategy.emplace_back(MNN_FORWARD_CPU, 1, 6, Confidant::ONE_ATTN_HEAD);
                    allocationStrategy.emplace_back(MNN_FORWARD_OPENCL, 1, 6, Confidant::ONE_ATTN_HEAD);

                    self = std::make_shared<GPT2GenericParallelSelfAttention>(hiddenSize, numAttentionHeads);
                } else {
                    self = std::make_shared<GPT2SelfAttention>(hiddenSize, numAttentionHeads);
                }
                layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-5));//12  ->5
                dense.reset(NN::Linear(hiddenSize, hiddenSize, true));

                registerModel({self, layerNorm, dense});
            }

            std::vector<Express::VARP> GPT2Attention::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP inputTensor = inputs[0];

                auto normoutput = layerNorm->forward(inputTensor);
                auto selfOutput = self->onForward({normoutput});
                auto denseOutput = dense->forward(selfOutput[0]);
                auto output = denseOutput + inputTensor;

                return {output};
            }

            GPT2Layer::GPT2Layer(int numAttentionHeads, int hiddenSize, int intermediateSize ,bool forParallel) {
                attention = std::make_shared<GPT2Attention>(hiddenSize, numAttentionHeads, forParallel);
                output = std::make_shared<GPT2Output>(hiddenSize, intermediateSize);

                registerModel({attention, output});
            }

            std::vector<Express::VARP> GPT2Layer::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;

                auto attentionOutput = attention->onForward({inputs[0]});
                auto layerOutput = output->onForward({attentionOutput[0]});
                return {layerOutput[0]};
            }

            // More generic parallel self-attn
            GPT2GenericParallelSelfAttention::GPT2GenericParallelSelfAttention(int hiddenSize, int numAttentionHeads, std::vector<Confidant::ProcessorInfo> allocationStrategy) {
                this->numAttentionHeads = numAttentionHeads;
                this->attentionHeadSize = hiddenSize / numAttentionHeads;
                this->allHeadSize = numAttentionHeads * attentionHeadSize;
                this->allocationStrategy = allocationStrategy;

                query.reset(NN::Linear(hiddenSize, allHeadSize, true));
                key.reset(NN::Linear(hiddenSize, allHeadSize, true));
                value.reset(NN::Linear(hiddenSize, allHeadSize, true));

                int allocatedAttnHeads = 0;
                for (auto& processorInfo : allocationStrategy) {
                    int curNumAttentionHead = processorInfo.numAttentionHead;
                    if (curNumAttentionHead == 0) {
                        continue;
                    }
                    auto curComputeWay = processorInfo.computeWay;

                    parallelKey.emplace_back(NN::Linear(hiddenSize, attentionHeadSize * curNumAttentionHead, true));
                    parallelQuery.emplace_back(NN::Linear(hiddenSize, attentionHeadSize * curNumAttentionHead, true));
                    parallelValue.emplace_back(NN::Linear(hiddenSize, attentionHeadSize * curNumAttentionHead, true));

                    allocatedAttnHeads += curNumAttentionHead;
                }

                MNN_ASSERT(this->numAttentionHeads == allocatedAttnHeads);

                registerModel(parallelKey);
                registerModel(parallelQuery);
                registerModel(parallelValue);
            }

            std::vector<Express::VARP> GPT2GenericParallelSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                std::call_once(mOnceFlag, [&]() {
                    // load params to parallelKey, parallelQuery, parallelValue
                    auto keyParams = key->parameters();
                    auto queryParams = query->parameters();
                    auto valueParams = value->parameters();

                    auto queryBiasParams = _Split(queryParams[0], {numAttentionHeads}, 1);
                    auto queryWeightParams = _Split(queryParams[1], {numAttentionHeads}, 0);
                    auto keyBiasParams = _Split(keyParams[0], {numAttentionHeads}, 1);
                    auto keyWeightParams = _Split(keyParams[1], {numAttentionHeads}, 0);
                    auto valueBiasParams = _Split(valueParams[0], {numAttentionHeads}, 1);
                    auto valueWeightParams = _Split(valueParams[1], {numAttentionHeads}, 0);

                    int loadedAttnHeads = 0;
                    int cnt = 0;
                    for (auto& processorInfo : allocationStrategy) {
                        int curNumAttentionHead = processorInfo.numAttentionHead;
                        if (curNumAttentionHead == 0) {
                            continue;
                        }

                        auto concatKeyBiasParams = _Concat(std::vector<VARP>(keyBiasParams.begin() + loadedAttnHeads, keyBiasParams.begin() + loadedAttnHeads + curNumAttentionHead), 1);
                        auto concatKeyWeightParams = _Concat(std::vector<VARP>(keyWeightParams.begin() + loadedAttnHeads, keyWeightParams.begin() + loadedAttnHeads + curNumAttentionHead), 0);
                        parallelKey[cnt]->loadParameters({_Clone(concatKeyBiasParams, true), _Clone(concatKeyWeightParams, true)});

                        auto concatQueryBiasParams = _Concat(std::vector<VARP>(queryBiasParams.begin() + loadedAttnHeads, queryBiasParams.begin() + loadedAttnHeads + curNumAttentionHead), 1);
                        auto concatQueryWeightParams = _Concat(std::vector<VARP>(queryWeightParams.begin() + loadedAttnHeads, queryWeightParams.begin() + loadedAttnHeads + curNumAttentionHead), 0);
                        parallelQuery[cnt]->loadParameters({_Clone(concatQueryBiasParams, true), _Clone(concatQueryWeightParams, true)});

                        auto concatValueBiasParams = _Concat(std::vector<VARP>(valueBiasParams.begin() + loadedAttnHeads, valueBiasParams.begin() + loadedAttnHeads + curNumAttentionHead), 1);
                        auto concatValueWeightParams = _Concat(std::vector<VARP>(valueWeightParams.begin() + loadedAttnHeads, valueWeightParams.begin() + loadedAttnHeads + curNumAttentionHead), 0);
                        parallelValue[cnt]->loadParameters({_Clone(concatValueBiasParams, true), _Clone(concatValueWeightParams, true)});
                        cnt++;

                        loadedAttnHeads += curNumAttentionHead;
                    }
                });

                using namespace Express;
                VARP x = inputs[0];

                auto exe = Executor::getGlobalExecutor();
                auto availableBackends = exe->getAvailableBackends();

                std::vector<VARP> allReduced;
                int cnt = 0;
                std::vector<std::pair<MNNForwardType, int>> types;
                for (auto& processorInfo : allocationStrategy) {
                    int curNumAttentionHead = processorInfo.numAttentionHead;
                    auto curType = processorInfo.type;
                    int curNumThread = processorInfo.numThread;
                    auto curComputeWay = processorInfo.computeWay;

                    if (curNumAttentionHead == 0) {
                        continue;
                    }

                    auto xClone = _Clone(x, true); // 在这里会调用readMap
                    auto queryLayer = parallelKey[cnt]->forward(xClone);
                    auto keyLayer = parallelKey[cnt]->forward(xClone);
                    auto valueLayer = parallelValue[cnt]->forward(xClone);

                    auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                    attentionScores = _Divide(attentionScores, _Const((float) sqrt(attentionHeadSize), {}, NCHW));

                    auto dim = attentionScores->getInfo()->dim;
                    auto nd = dim[dim.size()-2];
                    auto ns = dim[dim.size()-1];
                    std::vector<float> padIdVec(ns*nd, 0.0f);
                    for (int i = 0; i < nd; i++) {
                        for (int j = 0; j <= i + ns - nd; j++) {
                            padIdVec[i * ns + j] = 1.0f;
                        }
                    }
                    VARP selfBias = _Const(padIdVec.data(), {1, nd, ns}, NCHW, halide_type_of<float>());
                    auto attentionBias = attentionScores * selfBias - _Const(1e10f) * (_Const(1.0f) - selfBias);

                    // Normalize the attention scores to probabilities.
                    auto attentionProbs = _Softmax(attentionBias, -1);

                    // bts * seqLen * (attentionHeadSize * curNumAttnHead)
                    auto contextLayer = _MatMul(attentionProbs, valueLayer);
                    auto output = _Unsqueeze(contextLayer, {1});
                    allReduced.emplace_back(output);

                    if (availableBackends.find({curType, curNumThread}) != availableBackends.end() && curType != MNN_FORWARD_CPU) {
                        Variable::prepareComputeByForwardType({output}, false, {curType, curNumThread});
                    } else {
                        if (curType != MNN_FORWARD_CPU) {
                            MNN_PRINT("Backend %d with %d threads not available, falling back to CPU\n", curType,
                                      curNumThread);
                        }
                    }

                    cnt++;
                }

                // Temporarily we only modify to support two elements in allReduced
                std::vector<std::thread> threads(0);
                auto executeHelper = [&](int i) {
                    auto ptr = allReduced[i]->readMap<float>();
                    // MNN_PRINT("Current %d: %f\n", i, ptr[0]);
                };

                for (int i = 0; i < 2; ++i) {
                    threads.emplace_back(executeHelper, i);
                }

                for (auto& thread : threads) {
                    thread.join();
                }

                // split allReduced for concatenation
                std::vector<VARP> splitedAllReduced;
                for (int i = 0; i < allReduced.size(); ++i) {
                    int curNumAttentionHead = allReduced[i]->getInfo()->dim.back() / this->attentionHeadSize; // 64 is the hiddenSize / numAttentionHeads
                    if (curNumAttentionHead == 1) {
                        splitedAllReduced.emplace_back(allReduced[i]);
                        continue;
                    }
                    auto splitedContextLayer = _Split(allReduced[i], {curNumAttentionHead}, -1);
                    splitedAllReduced.insert(splitedAllReduced.end(), splitedContextLayer.begin(), splitedContextLayer.end());
                }

                auto contextLayers = _Concat(splitedAllReduced, 1);
                // bts * attnHead * seqLen * oneHeadSize
                contextLayers = _Permute(contextLayers, {0, 2, 1, 3});
                auto newContextLayerShape = contextLayers->getInfo()->dim;
                contextLayers = _Reshape(contextLayers, {newContextLayerShape[0], newContextLayerShape[1], allHeadSize});
                return {contextLayers};
            }
        }
    }
}