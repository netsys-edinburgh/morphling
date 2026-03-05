//
// Created by yue on 2024/3/12.
//

#include "Phi2Layer.h"
#include "log.h"
#include "Initializer.hpp"
#include <memory>

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;

            PhiLayer::PhiLayer(PhiArgs &args) {
                this->numHeads = args.nHeads;
                this->headDim = args.hiddenSize / args.nHeads;
                selfAttention.reset(new PhiSelfAttention(args));
                mlp.reset(new PhiFeedForward(args.hiddenSize, args.intermediateSize));

//                attentionNorm.reset(new MLayerNorm(args.hiddenSize, args.normEps));
//                ffnNorm.reset(new MLayerNorm(args.hiddenSize, args.normEps));
                input_layernorm.reset(new MLayerNorm({args.hiddenSize}, true, args.normEps));
//                ffnNorm.reset(new MLayerNorm({args.hiddenSize}, true,args.normEps));
                resid_dropout.reset(NN::Dropout(args.pdrop));

                registerModel({selfAttention, mlp, input_layernorm, resid_dropout});
            }

            std::vector<Express::VARP>
            PhiLayer::onForward(const std::vector<Express::VARP> &inputs) {
                VARP x = inputs[0];
                auto normX = input_layernorm->forward(x);
                VARP h;
                if (inputs.size() > 2) {
                    h = x + selfAttention->onForward({normX, inputs[1], inputs[2]})[0];
                } else {
                    h = x + selfAttention->onForward({normX, inputs[1]})[0];
                }
                auto out = h + mlp->onForward({resid_dropout->forward(h)})[0];
                return {out};
            }

            PhiSelfAttention::PhiSelfAttention(PhiArgs &args) {
                this->numKVHeads = args.nKVHeads == -1 ? args.nHeads : args.nKVHeads;
                this->numHeads = args.nHeads;
                this->hiddenSize = args.hiddenSize;
                this->headDim = args.hiddenSize / args.nHeads;

                if (args.loraArgs.r>0 && args.loraArgs.enableLoRA[0]){
                    this->query = std::make_shared<PhiLoRALayer>(args,hiddenSize,numHeads * headDim);
                    this->key = std::make_shared<PhiLoRALayer>(args,hiddenSize,this->numKVHeads * headDim);
                    this->value = std::make_shared<PhiLoRALayer>(args,hiddenSize,this->numKVHeads * headDim);
                } else{
                    this->query.reset(NN::Linear(hiddenSize, numHeads * headDim, true));
                    this->key.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, true));
                    this->value.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, true));
                }

                out.reset(NN::Linear(numHeads * headDim, hiddenSize, true));

                cacheKey = _Const(0.0f,
                                  {args.maxBatchSize, args.maxSeqLen, this->numKVHeads, headDim},
                                  NCHW);
                cacheValue = _Const(0.0f,
                                    {args.maxBatchSize, args.maxSeqLen, this->numKVHeads, headDim},
                                    NCHW);
                registerModel({query, key, value, out});
            }

            std::vector<Express::VARP>
            PhiSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                std::vector<int> inputDim = x->getInfo()->dim;
                VARP freqsComplex = inputs[1];

                auto mixedQueryLayer = query->forward(x);
                auto mixedKeyLayer = key->forward(x);
                auto mixedValueLayer = value->forward(x);

                auto shape = mixedKeyLayer->getInfo()->dim;
                auto queryLayer = _Reshape(mixedQueryLayer,
                                           {shape[0], shape[1], numHeads, headDim});
                auto keyLayer = _Reshape(mixedKeyLayer, {shape[0], shape[1], numKVHeads, headDim});
                auto valueLayer = _Reshape(mixedValueLayer,
                                           {shape[0], shape[1], numKVHeads, headDim});

                queryLayer = applyRotaryEmbedding(queryLayer, freqsComplex);
                keyLayer = applyRotaryEmbedding(keyLayer, freqsComplex);

                if (inputs.size() > 2) {
                    // startPos is used for generation task
                    VARP startPosVar = inputs[2];

                    int startPos = *(startPosVar->readMap<int>());
                    // save the keyLayer and valueLayer into the cacheKey and cacheValue
                    auto cacheInfo = cacheKey->getInfo();
                    auto curInfo = keyLayer->getInfo();

                    int seqLen = keyLayer->getInfo()->dim[1];
                    auto writeBegin = _Input({4}, NCHW);
                    auto writeEnd = _Input({4}, NCHW);
                    auto writeStride = _Input({4}, NCHW);
                    const int beginData[] = {0, startPos, 0, 0};
                    memcpy(writeBegin->writeMap<int>(), beginData, 4 * sizeof(int));
                    const int endData[] = {inputDim[0], startPos + seqLen, cacheInfo->dim[2],
                                           cacheInfo->dim[3]}; // here inputDim[0] is used because the input batch size may not match the batch size of cache
                    memcpy(writeEnd->writeMap<int>(), endData, 4 * sizeof(int));
                    const int strideData[] = {1, 1, 1, 1};
                    memcpy(writeStride->writeMap<int>(), strideData, 4 * sizeof(int));

                    this->cacheKey = _StridedSliceWrite(cacheKey, writeBegin, writeEnd, writeStride,
                                                        keyLayer, 0, 0, 0, 0, 0);
                    this->cacheValue = _StridedSliceWrite(cacheValue, writeBegin, writeEnd,
                                                          writeStride, valueLayer, 0, 0, 0, 0, 0);

                    auto readBegin = _Input({4}, NCHW);
                    auto readSizes = _Input({4}, NCHW);
                    const int readBeginData[] = {0, 0, 0, 0};
                    memcpy(readBegin->writeMap<int>(), readBeginData, 4 * sizeof(int));
                    const int readSizesData[] = {curInfo->dim[0], startPos + seqLen,
                                                 curInfo->dim[2], curInfo->dim[3]};
                    memcpy(readSizes->writeMap<int>(), readSizesData, 4 * sizeof(int));
                    keyLayer = _Slice(cacheKey, readBegin, readSizes);
                    valueLayer = _Slice(cacheValue, readBegin, readSizes);
                }

                // (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim) 扩充到 nRepeat
                int nRepeat = numHeads / numKVHeads;
                keyLayer = repeatKV(keyLayer, nRepeat);
                valueLayer = repeatKV(valueLayer, nRepeat);

                // (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
                queryLayer = _Transpose(queryLayer, {0, 2, 1, 3});
                // (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
                keyLayer = _Transpose(keyLayer, {0, 2, 1, 3});
                valueLayer = _Transpose(valueLayer, {0, 2, 1, 3});

                // (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
                auto attentionScores = _MatMul(queryLayer, keyLayer, false, true);
                attentionScores = _Divide(attentionScores, _Const((float) sqrt(headDim), {}, NCHW));
                auto attentionProbs = _Softmax(attentionScores, -1);

                auto contextLayer = _MatMul(attentionProbs, valueLayer);

                contextLayer = _Permute(contextLayer, {0, 2, 1, 3});
                auto newContextLayerShape = contextLayer->getInfo()->dim;
                contextLayer = _Reshape(contextLayer,
                                        {newContextLayerShape[0], newContextLayerShape[1],
                                         newContextLayerShape[2] * newContextLayerShape[3]});

                Variable::prepareCompute({contextLayer});
                auto output = out->forward(contextLayer);
                return {output};
            }

            PhiFeedForward::PhiFeedForward(int hiddenSize, int intermediateSize) {
                fc1.reset(NN::Linear(hiddenSize, intermediateSize, true));
                fc2.reset(NN::Linear(intermediateSize, hiddenSize, true));
                registerModel({fc1, fc2});
            }

            std::vector<Express::VARP> PhiFeedForward::onForward(const std::vector<Express::VARP> &inputs) {
                VARP x = inputs[0];
                auto output = fc1->forward(x);
                output = output * _Const(0.5f) * (_Const(1.0f) + _Tanh(_Const(0.7978845608028654f) * (output + _Const(0.044715f) * _Pow(output, _Const(3.0f)))));
                output = fc2->forward(output);

                return {output};
            }

            // More generic parallel self-attn
            Phi2ParallelSelfAttention::Phi2ParallelSelfAttention(PhiArgs &args, std::vector<Confidant::ProcessorInfo> allocationStrategy) {
                this->numKVHeads = args.nKVHeads == -1 ? args.nHeads : args.nKVHeads;
                this->numHeads = args.nHeads;
                this->hiddenSize = args.hiddenSize;
                this->headDim = args.hiddenSize / args.nHeads;
                this->allocationStrategy = allocationStrategy;

                // enable lora
//                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[0]) {
//                    this->query = std::make_shared<LLaMALoRALayer>(args, hiddenSize, numHeads * headDim);
//                } else {
//                    this->query.reset(NN::Linear(hiddenSize, numHeads * headDim, false));
//                }
//                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[1]) {
//                    this->key = std::make_shared<LLaMALoRALayer>(args, hiddenSize, this->numKVHeads * headDim);
//                } else {
//                    this->key.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));
//                }
//                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[2]) {
//                    this->value = std::make_shared<LLaMALoRALayer>(args, hiddenSize, this->numKVHeads * headDim);
//                } else {
//                    this->value.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));
//                }

                this->query.reset(NN::Linear(hiddenSize, numHeads * headDim, false));
                this->key.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));
                this->value.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));

                out.reset(NN::Linear(numHeads * headDim, hiddenSize, false));

                int allocatedAttnHeads = 0;
                for (auto& processorInfo : allocationStrategy) {
                    int curNumAttentionHead = processorInfo.numAttentionHead;
                    if (curNumAttentionHead == 0) {
                        continue;
                    }
                    auto curComputeWay = processorInfo.computeWay;

                    // TODO: currently we assume KVHeads == AttentionHeads
                    if (curComputeWay == Confidant::ONE_ATTN_HEAD) {
//                        if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[0]) {
//                            parallelKey.emplace_back(std::make_shared<LLaMALoRALayer>(args, hiddenSize, curNumAttentionHead * headDim));
//                        } else {
//                            parallelKey.emplace_back(NN::Linear(hiddenSize, headDim * curNumAttentionHead, false));
//                        }
//                        if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[1]) {
//                            parallelQuery.emplace_back(std::make_shared<LLaMALoRALayer>(args, hiddenSize, curNumAttentionHead * headDim));
//                        } else {
//                            parallelQuery.emplace_back(NN::Linear(hiddenSize, headDim * curNumAttentionHead, false));
//                        }
//                        if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[2]) {
//                            parallelValue.emplace_back(std::make_shared<LLaMALoRALayer>(args, hiddenSize, curNumAttentionHead * headDim));
//                        } else {
//                            parallelValue.emplace_back(NN::Linear(hiddenSize, headDim * curNumAttentionHead, false));
//                        }
                        parallelKey.emplace_back(NN::Linear(hiddenSize, headDim * curNumAttentionHead, false));
                        parallelQuery.emplace_back(NN::Linear(hiddenSize, headDim * curNumAttentionHead, false));
                        parallelValue.emplace_back(NN::Linear(hiddenSize, headDim * curNumAttentionHead, false));
                    } else {
                        // other ways
                        for (int i = 0; i < curNumAttentionHead; i++) {
//                            if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[0]) {
//                                parallelKey.emplace_back(std::make_shared<LLaMALoRALayer>(args, hiddenSize, headDim));
//                            } else {
//                                parallelKey.emplace_back(NN::Linear(hiddenSize, headDim, false));
//                            }
//                            if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[1]) {
//                                parallelQuery.emplace_back(std::make_shared<LLaMALoRALayer>(args, hiddenSize, headDim));
//                            } else {
//                                parallelQuery.emplace_back(NN::Linear(hiddenSize, headDim, false));
//                            }
//                            if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[2]) {
//                                parallelValue.emplace_back(std::make_shared<LLaMALoRALayer>(args, hiddenSize, headDim));
//                            } else {
//                                parallelValue.emplace_back(NN::Linear(hiddenSize, headDim, false));
//                            }
                            parallelKey.emplace_back(NN::Linear(hiddenSize, headDim, false));
                            parallelQuery.emplace_back(NN::Linear(hiddenSize, headDim, false));
                            parallelValue.emplace_back(NN::Linear(hiddenSize, headDim, false));
                        }
                    }
                    allocatedAttnHeads += curNumAttentionHead;
                }

                MNN_ASSERT(this->numHeads == allocatedAttnHeads);

                registerModel(parallelKey);
                registerModel(parallelQuery);
                registerModel(parallelValue);
                registerModel({out});
            }

            std::vector<Express::VARP> Phi2ParallelSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                std::call_once(mOnceFlag, [&]() {
                    // load params to parallelKey, parallelQuery, parallelValue
                    auto keyParams = key->parameters();
                    auto queryParams = query->parameters();
                    auto valueParams = value->parameters();

                    auto queryWeightParams = _Split(queryParams[0], {this->numHeads}, 0);
                    auto keyWeightParams = _Split(keyParams[0], {this->numHeads}, 0);
                    auto valueWeightParams = _Split(valueParams[0], {this->numHeads}, 0);

                    int loadedAttnHeads = 0;
                    int cnt = 0;
                    for (auto& processorInfo : allocationStrategy) {
                        int curNumAttentionHead = processorInfo.numAttentionHead;
                        if (curNumAttentionHead == 0) {
                            continue;
                        }

                        auto curComputeWay = processorInfo.computeWay;

                        if (curComputeWay == Confidant::ONE_ATTN_HEAD && curNumAttentionHead > 1) {
                            auto concatKeyWeightParams = _Concat(std::vector<VARP>(keyWeightParams.begin() + loadedAttnHeads, keyWeightParams.begin() + loadedAttnHeads + curNumAttentionHead), 0);
                            parallelKey[cnt]->loadParameters({_Clone(concatKeyWeightParams, true)});

                            auto concatQueryWeightParams = _Concat(std::vector<VARP>(queryWeightParams.begin() + loadedAttnHeads, queryWeightParams.begin() + loadedAttnHeads + curNumAttentionHead), 0);
                            parallelQuery[cnt]->loadParameters({_Clone(concatQueryWeightParams, true)});

                            auto concatValueWeightParams = _Concat(std::vector<VARP>(valueWeightParams.begin() + loadedAttnHeads, valueWeightParams.begin() + loadedAttnHeads + curNumAttentionHead), 0);
                            parallelValue[cnt]->loadParameters({_Clone(concatValueWeightParams, true)});
                            cnt++;
                        } else {
                            // other ways
                            for (int i = 0; i < curNumAttentionHead; i++) {
                                parallelKey[cnt]->loadParameters({_Clone(keyWeightParams[loadedAttnHeads + i], true)});
                                parallelQuery[cnt]->loadParameters({_Clone(queryWeightParams[loadedAttnHeads + i], true)});
                                parallelValue[cnt]->loadParameters({_Clone(valueWeightParams[loadedAttnHeads + i], true)});
                                cnt++;
                            }
                        }

                        loadedAttnHeads += curNumAttentionHead;
                    }
                });

                using namespace Express;
                VARP x = inputs[0];
                VARP freqsComplex = inputs[1];

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

                    if (curComputeWay == Confidant::ONE_ATTN_HEAD) {
                        auto xClone = _Clone(x, true); // 在这里会调用readMap
                        auto freqsComplexClone = _Clone(freqsComplex, true);

                        auto mixedQueryLayer = parallelQuery[cnt]->forward(xClone);
                        auto mixedKeyLayer = parallelKey[cnt]->forward(xClone);
                        auto mixedValueLayer = parallelValue[cnt]->forward(xClone);

                        auto shape = mixedKeyLayer->getInfo()->dim;
                        auto queryLayer = _Reshape(mixedQueryLayer, {shape[0], shape[1], curNumAttentionHead, headDim});
                        auto keyLayer = _Reshape(mixedKeyLayer, {shape[0], shape[1], curNumAttentionHead, headDim});
                        auto valueLayer = _Reshape(mixedValueLayer, {shape[0], shape[1], curNumAttentionHead, headDim});

                        queryLayer = applyRotaryEmbedding(queryLayer, freqsComplexClone);
                        keyLayer = applyRotaryEmbedding(keyLayer, freqsComplexClone);

                        int nRepeat = 1;
                        keyLayer = repeatKV(keyLayer, nRepeat);
                        valueLayer = repeatKV(valueLayer, nRepeat);

                        // (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
                        queryLayer = _Transpose(queryLayer, {0, 2, 1, 3});
                        // (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
                        keyLayer = _Transpose(keyLayer, {0, 2, 1, 3});
                        valueLayer = _Transpose(valueLayer, {0, 2, 1, 3});

                        // (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
                        auto attentionScores = _MatMul(queryLayer, keyLayer, false, true);
                        attentionScores = _Divide(attentionScores, _Const((float) sqrt(headDim), {}, NCHW));
                        auto attentionProbs = _Softmax(attentionScores, -1);

                        auto contextLayer = _MatMul(attentionProbs, valueLayer);

                        allReduced.emplace_back(contextLayer);

                        if (availableBackends.find({curType, curNumThread}) != availableBackends.end()) {
                            types.emplace_back(curType, curNumThread);
                        } else {
                            MNN_PRINT("Backend %d with %d threads not available, falling back to CPU\n", curType, curNumThread);
                            types.emplace_back(MNN_FORWARD_CPU, 1);
                        }

                        cnt++;
                    } else if (curComputeWay == Confidant::SEP_PARALLEL_ATTN_HEAD) {
                        // compute the attn head separately and parallelly
                        for (int i = 0; i < curNumAttentionHead; i++) {
                            auto xClone = _Clone(x, true);
                            auto freqsComplexClone = _Clone(freqsComplex, true);

                            auto queryLayer = parallelKey[cnt]->forward(xClone);
                            auto keyLayer = parallelKey[cnt]->forward(xClone);
                            auto valueLayer = parallelValue[cnt]->forward(xClone);

                            auto singleHeadDim = queryLayer->getInfo()->dim;
                            queryLayer = _Reshape(queryLayer,{singleHeadDim[0], singleHeadDim[1], 1, singleHeadDim[2]});
                            keyLayer = _Reshape(keyLayer,{singleHeadDim[0], singleHeadDim[1], 1, singleHeadDim[2]});
                            valueLayer = _Reshape(valueLayer,{singleHeadDim[0], singleHeadDim[1], 1, singleHeadDim[2]});

                            queryLayer = applyRotaryEmbedding(queryLayer, freqsComplexClone);
                            keyLayer = applyRotaryEmbedding(keyLayer, freqsComplexClone);

                            queryLayer = _Transpose(queryLayer, {0, 2, 1, 3});
                            keyLayer = _Transpose(keyLayer, {0, 2, 1, 3});
                            valueLayer = _Transpose(valueLayer, {0, 2, 1, 3});

                            auto attentionScores = _MatMul(queryLayer, keyLayer, false, true);
                            attentionScores = _Divide(attentionScores, _Const((float) sqrt(headDim), {}, NCHW));
                            auto attentionProbs = _Softmax(attentionScores, -1);

                            auto contextLayer = _MatMul(attentionProbs, valueLayer);
                            allReduced.emplace_back(contextLayer);

                            if (availableBackends.find({curType, curNumThread}) != availableBackends.end()) {
                                types.emplace_back(curType, curNumThread);
                            } else {
                                LOGI("Backend %d with %d threads not available, falling back to CPU\n", curType, curNumThread);
                                types.emplace_back(MNN_FORWARD_CPU, 1);
                            }

                            cnt++;
                        }
                    } else {
                        // compute the attn head separately and sequentially
                        std::vector<VARP> contextLayers;
                        for (int i = 0; i < curNumAttentionHead; i++) {
                            auto xClone = _Clone(x, true);
                            auto freqsComplexClone = _Clone(freqsComplex, true);

                            auto queryLayer = parallelKey[cnt]->forward(xClone);
                            auto keyLayer = parallelKey[cnt]->forward(xClone);
                            auto valueLayer = parallelValue[cnt]->forward(xClone);

                            auto singleHeadDim = queryLayer->getInfo()->dim;
                            queryLayer = _Reshape(queryLayer,{singleHeadDim[0], singleHeadDim[1], 1, singleHeadDim[2]});
                            keyLayer = _Reshape(keyLayer,{singleHeadDim[0], singleHeadDim[1], 1, singleHeadDim[2]});
                            valueLayer = _Reshape(valueLayer,{singleHeadDim[0], singleHeadDim[1], 1, singleHeadDim[2]});

                            queryLayer = applyRotaryEmbedding(queryLayer, freqsComplexClone);
                            keyLayer = applyRotaryEmbedding(keyLayer, freqsComplexClone);

                            queryLayer = _Transpose(queryLayer, {0, 2, 1, 3});
                            keyLayer = _Transpose(keyLayer, {0, 2, 1, 3});
                            valueLayer = _Transpose(valueLayer, {0, 2, 1, 3});

                            auto attentionScores = _MatMul(queryLayer, keyLayer, false, true);
                            attentionScores = _Divide(attentionScores, _Const((float) sqrt(headDim), {}, NCHW));
                            auto attentionProbs = _Softmax(attentionScores, -1);

                            auto contextLayer = _MatMul(attentionProbs, valueLayer);

                            contextLayers.emplace_back(contextLayer);
                            cnt++;
                        }

                        auto concatContextLayer = _Concat(contextLayers, -1);
                        allReduced.emplace_back(_Unsqueeze(concatContextLayer, {1}));

                        if (availableBackends.find({curType, curNumThread}) != availableBackends.end()) {
                            types.emplace_back(curType, curNumThread);
                        } else {
                            MNN_PRINT("Backend %d with %d threads not available, falling back to CPU\n", curType, curNumThread);
                            types.emplace_back(MNN_FORWARD_CPU, 1);
                        }
                    }
                }
                Variable::prepareComputeParallel(allReduced, true, types);

                // split allReduced for concatenation
                std::vector<VARP> splitedAllReduced;
                for (int i = 0; i < allReduced.size(); ++i) {
                    int curNumAttentionHead = allReduced[i]->getInfo()->dim[1];
                    if (curNumAttentionHead == 1) {
                        splitedAllReduced.emplace_back(allReduced[i]);
                        continue;
                    }
                    auto splitedContextLayer = _Split(allReduced[i], {curNumAttentionHead}, 1);
                    splitedAllReduced.insert(splitedAllReduced.end(), splitedContextLayer.begin(), splitedContextLayer.end());
                }

                auto contextLayers = _Concat(splitedAllReduced, 1);

                contextLayers = _Permute(contextLayers, {0, 2, 1, 3});
                auto newContextLayerShape = contextLayers->getInfo()->dim;
                contextLayers = _Reshape(contextLayers, {newContextLayerShape[0], newContextLayerShape[1], newContextLayerShape[2] * newContextLayerShape[3]});

                auto output = out->forward(contextLayers);

                return {output};
            }

            PhiLoRALayer::PhiLoRALayer(PhiArgs &args, int inFeatures, int outFeatures) {
//                MNN_ASSERT(args.loraArgs.r > 0);
                this->alpha = 1.0f;
                this->r = 8;
//                this->loraDropout = 0.0f;

                std::shared_ptr<Initializer> initializer;
                initializer.reset(Initializer::MSRA());
                this->weight = initializer->createConstVar({outFeatures, inFeatures}, NCHW);

                this->loraA = initializer->createConstVar({this->r, inFeatures}, NCHW);
                this->loraB = _Const(0.0f, {outFeatures, this->r}, NCHW);
                loraDropout.reset(NN::Dropout(0.0f));
                addParameter(weight);
            }

            std::vector<Express::VARP> PhiLoRALayer::onForward(const std::vector<Express::VARP> &inputs) {
                VARP x = inputs[0];
                auto output = _MatMul(inputs[0], weight, false, true);

                auto afterA = _MatMul(loraDropout->forward(x), loraA, false, true);
                // weight: 1 * r * seqLen, loraB: dim * r
                auto loraBShape = loraB->getInfo()->dim;
                auto transposedA = _Transpose(afterA, {0, 2, 1});
                auto transposedAShape = transposedA->getInfo()->dim;

                auto afterB = _Conv(_Reshape(loraB, {loraBShape[0], loraBShape[1], 1, 1}), _Const(0.0f, {loraBShape[0], loraBShape[1], 1, 1}), _Reshape(transposedA, {transposedAShape[0], transposedAShape[1], transposedAShape[2], 1}));
                afterB = _Squeeze(afterB, {3});

                output = output + _Multiply(_Transpose(afterB, {0, 2, 1}), _Const(alpha / (float)r));
                return {output};
            }
        }
    }
}


