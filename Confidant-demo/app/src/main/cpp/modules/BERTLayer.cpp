//
// Created by Yuhao Chen on 2023/6/16.
//
#include "BERTLayer.h"
#include "MLayerNorm.h"
#include <thread>
#include "log.h"
#include <MNN/AutoTime.hpp>
#include "multiProcessorScheduler.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            BERTSelfAttention::BERTSelfAttention(int hiddenSize, int numAttentionHeads,
                                                 float attentionProbsDropoutProb, bool forParallel) {
                this->numAttentionHeads = numAttentionHeads;
                this->attentionHeadSize = hiddenSize / numAttentionHeads;
                this->allHeadSize = numAttentionHeads * attentionHeadSize;

                this->forParallel = forParallel;

                query.reset(NN::Linear(hiddenSize, allHeadSize, true));
                query->setName("query");

                key.reset(NN::Linear(hiddenSize, allHeadSize, true));
                key->setName("key");

                value.reset(NN::Linear(hiddenSize, allHeadSize, true));
                value->setName("value");

                dropout.reset(NN::Dropout(attentionProbsDropoutProb));
                registerModel({query, key, value, dropout});
            }

            std::vector<Express::VARP> BERTSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                VARP attentionMask = inputs[1];

                // squeeze the attentionMask
                attentionMask = _Squeeze(attentionMask, {1});

                // bts * seqLen * hiddenSize
                auto queryLayer = query->forward(x);
                auto keyLayer = key->forward(x);
                auto valueLayer = value->forward(x);

                auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                // auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 1, 3, 2}));
                attentionScores = _Divide(attentionScores, _Const((float) sqrt(attentionHeadSize), {}, NCHW));
                attentionScores = attentionScores + attentionMask;

                // Normalize the attention scores to probabilities.
                auto attentionProbs = _Softmax(attentionScores, -1);

                // TODO: This is actually dropping out entire tokens to attend to, which might
                //  seem a bit unusual, but is taken from the original Transformer paper.
                attentionProbs = dropout->forward(attentionProbs);
                auto contextLayer = _MatMul(attentionProbs, valueLayer);

                // Variable::prepareCompute({contextLayer});

                return {contextLayer};
            }

            ParallelSelfAttention::ParallelSelfAttention(int hiddenSize, int numAttentionHeads,
                                                 float attentionProbsDropoutProb, int firstPart) {
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
                dropouts.resize(numAttentionHeads);

                for (int i = 0; i < numAttentionHeads; i++) {
                    parallelQuery[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelQuery[i]->setName("query" + std::to_string(i));

                    parallelKey[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelKey[i]->setName("key" + std::to_string(i));

                    parallelValue[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelValue[i]->setName("value" + std::to_string(i));

                    dropouts[i].reset(NN::Dropout(attentionProbsDropoutProb));
                }

                this->allHeadSize = numAttentionHeads * attentionHeadSize;

                dropout.reset(NN::Dropout(attentionProbsDropoutProb));

                // multiple backends
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

                registerModel({dropout});
            }

            std::vector<Express::VARP> ParallelSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
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
                        parallelKey[i]->loadParameters({_Clone(keyBiasParams[i], true), _Clone(keyWeightParams[i], true)});
                        parallelQuery[i]->loadParameters({_Clone(queryBiasParams[i], true), _Clone(queryWeightParams[i], true)});
                        parallelValue[i]->loadParameters({_Clone(valueBiasParams[i], true), _Clone(valueWeightParams[i], true)});
                    }

                    auto exe = Executor::getGlobalExecutor();
                    auto availableBackends = exe->getAvailableBackends();
                    if (availableBackends.size() > 1) {
                        auto concatKeyBiasParams = _Concat(std::vector<VARP>(keyBiasParams.begin(), keyBiasParams.begin() + firstPart), 1);
                        auto concatKeyWeightParams = _Concat(std::vector<VARP>(keyWeightParams.begin(), keyWeightParams.begin() + firstPart), 0);
                        concatKey->loadParameters({_Clone(concatKeyBiasParams, true), _Clone(concatKeyWeightParams, true)});

                        auto concatQueryBiasParams = _Concat(std::vector<VARP>(queryBiasParams.begin(), queryBiasParams.begin() + firstPart), 1);
                        auto concatQueryWeightParams = _Concat(std::vector<VARP>(queryWeightParams.begin(), queryWeightParams.begin() + firstPart), 0);
                        concatQuery->loadParameters({_Clone(concatQueryBiasParams, true), _Clone(concatQueryWeightParams, true)});

                        auto concatValueBiasParams = _Concat(std::vector<VARP>(valueBiasParams.begin(), valueBiasParams.begin() + firstPart), 1);
                        auto concatValueWeightParams = _Concat(std::vector<VARP>(valueWeightParams.begin(), valueWeightParams.begin() + firstPart), 0);
                        concatValue->loadParameters({_Clone(concatValueBiasParams, true), _Clone(concatValueWeightParams, true)});
                    }
                });

                using namespace Express;
                VARP x = inputs[0];
                VARP attentionMask = inputs[1];

                // squeeze the attentionMask
                attentionMask = _Squeeze(attentionMask, {1});

                std::vector<VARP> allReduced;

                auto exe = Executor::getGlobalExecutor();
                auto availableBackends = exe->getAvailableBackends();

                // TODO: Only support two backends now
                if (availableBackends.size() > 1) {
                    MNN_PRINT("Using multiple backends\n");
                    LOGI("Using multiple backends\n");
                    std::vector<VARP> multipleAllReduced;
                    std::vector<std::pair<MNNForwardType, int>> types;

                    // concat part
                    auto concatXClone = _Clone(x, true); // 在这里会调用readMap
                    auto concatQueryLayer = concatQuery->forward(concatXClone);
                    auto concatKeyLayer = concatKey->forward(concatXClone);
                    auto concatValueLayer = concatValue->forward(concatXClone);

                    auto concatAttentionScores = _MatMul(concatQueryLayer, _Transpose(concatKeyLayer, {0, 2, 1}));
                    concatAttentionScores = concatAttentionScores + _Clone(attentionMask, true);

                    // Normalize the attention scores to probabilities.
                    auto concatAttentionProbs = _Softmax(concatAttentionScores, -1);

                    concatAttentionProbs = dropouts[0]->forward(concatAttentionProbs);

                    auto concatContextLayer = _MatMul(concatAttentionProbs, concatValueLayer);
                    multipleAllReduced.emplace_back(concatContextLayer);

                    for (auto& backendType : availableBackends) {
                        if (backendType.first != MNN_FORWARD_CPU) {
                            MNN_PRINT("Parallel Using first backend %d with %d attention heads on it\n", backendType.first, firstPart);
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
                        attentionScores = attentionScores + _Clone(attentionMask, true);

                        // Normalize the attention scores to probabilities.
                        auto attentionProbs = _Softmax(attentionScores, -1);

                        attentionProbs = dropouts[i]->forward(attentionProbs);

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
                    LOGI("Parallel Computing using one backend\n");
                    for (int i = 0; i < numAttentionHeads; ++i) {
                        auto xClone = _Clone(x, true); // 在这里会调用readMap
                        auto queryLayer = parallelKey[i]->forward(xClone);
                        auto keyLayer = parallelKey[i]->forward(xClone);
                        auto valueLayer = parallelValue[i]->forward(xClone);

                        auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                        attentionScores = attentionScores + _Clone(attentionMask, true);

                        // Normalize the attention scores to probabilities.
                        auto attentionProbs = _Softmax(attentionScores, -1);

                        attentionProbs = dropouts[i]->forward(attentionProbs);

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

            BERTIntermediate::BERTIntermediate(int hiddenSize, int intermediateSize) {
                dense.reset(NN::Linear(hiddenSize, intermediateSize, true));
                registerModel({dense});
            }

            std::vector<Express::VARP> BERTIntermediate::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];

                auto hiddenStates = dense->forward(x);
                // auto ptr = dense->parameters()[0]->readMap<float>();
                // Calculate Gelu with Phi function
                auto geluPhi = _Const(0.5f) * (_Const(1.0f) + _Erf(hiddenStates / _Const(sqrt(2.0))));
                hiddenStates = hiddenStates * geluPhi;
                // hiddenStates = _Gelu(hiddenStates);

                return {hiddenStates};
            }


            BERTOutput::BERTOutput(int hiddenSize, int intermediateSize, float dropoutProb) {
                dense.reset(NN::Linear(intermediateSize, hiddenSize, true));
                layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-12));
                dropout.reset(NN::Dropout(dropoutProb));

                registerModel({dense, layerNorm, dropout});
            }

            std::vector<Express::VARP> BERTOutput::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                VARP inputTensor = inputs[1];

                auto hiddenStates = dense->forward(x);
                hiddenStates = dropout->forward(hiddenStates);
                hiddenStates = layerNorm->forward(hiddenStates + inputTensor);

                return {hiddenStates};
            }

            BERTSelfOutput::BERTSelfOutput(int hiddenSize, float dropoutProb) {
                dense.reset(NN::Linear(hiddenSize, hiddenSize, true));
                layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-12));
                dropout.reset(NN::Dropout(dropoutProb));

                registerModel({dense, layerNorm, dropout});
            }

            std::vector<Express::VARP> BERTSelfOutput::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                VARP inputTensor = inputs[1];

                auto hiddenStates = dense->forward(x);
                hiddenStates = dropout->forward(hiddenStates);
                hiddenStates = layerNorm->forward(hiddenStates + inputTensor);

                return {hiddenStates};
            }

            BERTAttention::BERTAttention(int hiddenSize, int numAttentionHeads, float dropoutProb, bool forParallel) {
                this->forParallel = forParallel;

                output.reset(new BERTSelfOutput(hiddenSize, dropoutProb));

                if (forParallel) {
                    LOGI("Using multi-processor scheduling ...\n");
                    std::vector<Confidant::ProcessorInfo> allocationStrategy = Confidant::MultiProcessorScheduler::mpsPtr->getAllocationStrategy().second;

                    paraSelf.reset(new BERTParallelSelfAttention(hiddenSize, numAttentionHeads, dropoutProb, allocationStrategy));
                    registerModel({paraSelf, output});
                } else {
                    self.reset(new BERTSelfAttention(hiddenSize, numAttentionHeads, dropoutProb, forParallel));
                    registerModel({self, output});
                }
            }

            std::vector<Express::VARP> BERTAttention::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP inputTensor = inputs[0];
                VARP attentionMask = inputs[1];

                if (forParallel) {
                    auto selfOutput = paraSelf->onForward({inputTensor, attentionMask});
                    auto attentionOutput = output->onForward({selfOutput[0], inputTensor});
                    return {attentionOutput};
                }

                // Timer _timer;
                auto selfOutput = self->onForward({inputTensor, attentionMask});
                // auto ptr = selfOutput[0]->readMap<float>();
                // MNN_PRINT("Self Time: %f\n", (float)_timer.durationInUs() / 1000.0f);
                auto attentionOutput = output->onForward({selfOutput[0], inputTensor});
                return {attentionOutput};
            }

            BERTLayer::BERTLayer(int numAttentionHeads, int hiddenSize, int intermediateSize, float dropoutProb, bool forParallel) {
                attention.reset(new BERTAttention(hiddenSize, numAttentionHeads, dropoutProb, forParallel));
                // attention.reset(new BERTAttention(hiddenSize, numAttentionHeads, dropoutProb, forParallel));
                intermediate.reset(new BERTIntermediate(hiddenSize, intermediateSize));
                output.reset(new BERTOutput(hiddenSize, intermediateSize, dropoutProb));

                registerModel({attention, intermediate, output});
            }

            std::vector<Express::VARP> BERTLayer::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = inputs[0];
                VARP attentionMask = inputs[1];

                Timer _attnTime;
                auto attentionOutput = attention->onForward({hiddenStates, attentionMask});
                // auto attnPtr = attentionOutput[0]->readMap<float>();
                // LOGI("BERTLayer_Encoder Time: %f\n", (float)_attnTime.durationInUs() / 1000.0f);

                Timer _intermediateTime;
                auto intermediateOutput = intermediate->onForward({attentionOutput[0]});
                // auto interPtr = intermediateOutput[0]->readMap<float>();
                // LOGI("BERTLayer_Intermediate Time: %f\n", (float)_intermediateTime.durationInUs() / 1000.0f);

                Timer _outputTime;
                auto layerOutput = output->onForward({intermediateOutput[0], attentionOutput[0]});
                // auto outPtr = layerOutput[0]->readMap<float>();
                // LOGI("BERTLayer_Output Time: %f\n", (float)_outputTime.durationInUs() / 1000.0f);

                return {layerOutput[0]};
            }

            // More generic parallel self-attn
            BERTParallelSelfAttention::BERTParallelSelfAttention(int hiddenSize, int numAttentionHeads,
                                                                 float attentionProbsDropoutProb, std::vector<Confidant::ProcessorInfo> allocationStrategy) {
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
                    dropouts.emplace_back(NN::Dropout(attentionProbsDropoutProb));

                    allocatedAttnHeads += curNumAttentionHead;
                }

                MNN_ASSERT(this->numAttentionHeads == allocatedAttnHeads);

                registerModel(parallelKey);
                registerModel(parallelQuery);
                registerModel(parallelValue);
            }

            std::vector<Express::VARP> BERTParallelSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
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

                        auto curComputeWay = processorInfo.computeWay;

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
                VARP attentionMask = inputs[1];

                // squeeze the attentionMask
                attentionMask = _Squeeze(attentionMask, {1});

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
                    attentionScores = attentionScores + _Clone(attentionMask, true);

                    // Normalize the attention scores to probabilities.
                    auto attentionProbs = _Softmax(attentionScores, -1);

                    attentionProbs = dropouts[cnt]->forward(attentionProbs);

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
                    int curNumAttentionHead = allReduced[i]->getInfo()->dim.back() / attentionHeadSize; // 64 is the hiddenSize / numAttentionHeads
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