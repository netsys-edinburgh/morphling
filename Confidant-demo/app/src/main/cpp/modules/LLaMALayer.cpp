//
// Created by Yuhao Chen on 2023/10/17.
//

#include "LLaMALayer.h"

#include <memory>
#include "Initializer.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;
            VARP precomputeThetaPosFrequencies(int headDim, int seqLen, float theta) {
                if (headDim % 2 != 0) {
                    MNN_ERROR("headDim must be divisible by 2");
                    return nullptr;
                }

                auto thetaNumerator = _Range(_Const(0.0f, {}, NCHW), _Const((float) headDim, {}, NCHW), _Const(2.0f, {}, NCHW));
                auto thetaVar = _Divide(_Const(1.0f), _Pow(_Const(theta), _Divide(thetaNumerator, _Const((float) headDim))));

                auto m = _Range(_Const(0.0f, {}, NCHW), _Const((float) seqLen, {}, NCHW), _Const(1.0f, {}, NCHW));

                // Outer product
                auto freqs = _Reshape(m, {m->getInfo()->size, 1}) * _Reshape(thetaVar, {1, thetaVar->getInfo()->size}); // 4, 2

                // freqs complex
                auto abs = _Const(1.0f, freqs->getInfo()->dim, NCHW);
                auto realPart = abs * _Cos(freqs);
                auto imagPart = abs * _Sin(freqs);
                auto freqsComplex = _Concat({_Unsqueeze(realPart, {0}), _Unsqueeze(imagPart, {0})}, 0);

                return freqsComplex;
            }

            VARP repeatKV(VARP x, int nRepeat) {
                std::vector<int> dims = x->getInfo()->dim;
                if (nRepeat == 1) {
                    return x;
                }
                const int broadCastShapeData[] = {dims[0], dims[1], dims[2], nRepeat, dims[3]};
                auto broadCastShape = _Const(broadCastShapeData, {5}, NHWC, halide_type_of<int>());
                auto output = _BroadcastTo(_Reshape(x, {dims[0], dims[1], dims[2], 1, dims[3]}), broadCastShape);
                output = _Reshape(x, {dims[0], dims[1], dims[2] * nRepeat, dims[3]});
                return output;
            }

            VARP applyRotaryEmbedding(VARP x, VARP freqsComplex) {
                auto dim = x->getInfo()->dim;
                auto xComplex = _Split(_Reshape(x, {dim[0], dim[1], dim[2], dim[3] / 2, 2}), {2}, -1);
                xComplex[0] = _Squeeze(xComplex[0], {-1}); // [B, Seq_Len, H, Head_Dim/2, 1] => [B, Seq_Len, H, Head_Dim/2]
                xComplex[1] = _Squeeze(xComplex[1], {-1});

                freqsComplex = _Unsqueeze(freqsComplex, {2});
                auto freqsComplexArr = _Split(freqsComplex, {2}, 0); // freqsComplex: [2, seqLen, headDim]

                // perform the complex multiplication
                auto xRotatedReal = xComplex[0] * freqsComplexArr[0] - xComplex[1] * freqsComplexArr[1];
                // auto realPtr = xRotatedReal->readMap<float>();
                auto xRotatedImag = xComplex[0] * freqsComplexArr[1] + xComplex[1] * freqsComplexArr[0];
                // auto imagPtr = xRotatedImag->readMap<float>();
                auto xRotated = _Concat({_Unsqueeze(xRotatedReal, {-1}), _Unsqueeze(xRotatedImag, {-1})}, -1);
                xRotated = _Reshape(xRotated, dim);

                return xRotated;
            }

            LLaMALoRALayer::LLaMALoRALayer(LLaMAArgs &args, int inFeatures, int outFeatures) {
                // In LoRA, the pretrained weights (W) are decomposed into two matrices (A and B) such that W = AB
                // W: dim * dim, A: dim * r, B: r * dim
                // We use LoRALayer to replace the NN::Linear in q, k, v
                MNN_ASSERT(args.loraArgs.r > 0);
                this->alpha = args.loraArgs.alpha;
                this->r = args.loraArgs.r;

                // init weight
                std::shared_ptr<Initializer> initializer;
                initializer.reset(Initializer::MSRA());
                this->weight = initializer->createConstVar({outFeatures, inFeatures}, NCHW);

                // init loraA and loraB
                // loraA = _Const(0.0f, {args.loraArgs.r, inFeatures}, NCHW);
                this->loraA = initializer->createConstVar({args.loraArgs.r, inFeatures}, NCHW);
                this->loraB = _Const(0.0f, {outFeatures, args.loraArgs.r}, NCHW);
                // this->loraB = initializer->createConstVar({outFeatures, args.loraArgs.r}, NCHW);
                loraDropout.reset(NN::Dropout(args.loraArgs.dropout));
                addParameter(weight);
            }

            std::vector<Express::VARP> LLaMALoRALayer::onForward(const std::vector<Express::VARP> &inputs) {
                // auto output = _MatMul(input, weight, false, true);
                // here merged computing is temporarily not considered
                // The output of LoRALayer should be the same as the output of NN::Linear
                // Input x: batch_size * seq_len * dim (1 * 1 * 4096)
                // Output: batch_size * seq_len * dim (1 * 1 * 4096)

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

            LLaMALayer::LLaMALayer(LLaMAArgs& args) {
                this->numHeads = args.nHeads;
                this->headDim = args.hiddenSize / args.nHeads;
                selfAttention = std::make_shared<LLaMASelfAttention>(args);
                feedForward = std::make_shared<LLaMAFeedForward>(args.hiddenSize, args.multipleOf, args.ffnDimMultiplier);

                attentionNorm = std::make_shared<MRMSNorm>(args.hiddenSize, args.normEps);
                ffnNorm = std::make_shared<MRMSNorm>(args.hiddenSize, args.normEps);

                registerModel({selfAttention, feedForward, attentionNorm, ffnNorm});
            }

            std::vector<Express::VARP> LLaMALayer::onForward(const std::vector<Express::VARP> &inputs) {
                VARP x = inputs[0];
                auto normX = attentionNorm->forward(x);
                VARP h;
                if (inputs.size() > 2) {
                    h = x + selfAttention->onForward({normX, inputs[1], inputs[2]})[0];
                } else {
                    h = x + selfAttention->onForward({normX, inputs[1]})[0];
                }
                auto out = h + feedForward->onForward({ffnNorm->forward(h)})[0];
                return {out};
            }

            LLaMASelfAttention::LLaMASelfAttention(LLaMAArgs& args) {
                this->numKVHeads = args.nKVHeads == -1 ? args.nHeads : args.nKVHeads;
                this->numHeads = args.nHeads;
                this->hiddenSize = args.hiddenSize;
                this->headDim = args.hiddenSize / args.nHeads;

                // enable lora
                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[0]) {
                    this->query = std::make_shared<LLaMALoRALayer>(args, hiddenSize, numHeads * headDim);
                } else {
                    this->query.reset(NN::Linear(hiddenSize, numHeads * headDim, false));
                }
                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[1]) {
                    this->key = std::make_shared<LLaMALoRALayer>(args, hiddenSize, this->numKVHeads * headDim);
                } else {
                    this->key.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));

                }
                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[2]) {
                    this->value = std::make_shared<LLaMALoRALayer>(args, hiddenSize, this->numKVHeads * headDim);
                } else {
                    this->value.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));
                }

                out.reset(NN::Linear(numHeads * headDim, hiddenSize, false));

                cacheKey = _Const(0.0f, {args.maxBatchSize, args.maxSeqLen, this->numKVHeads, headDim}, NCHW);
                cacheValue = _Const(0.0f, {args.maxBatchSize, args.maxSeqLen, this->numKVHeads, headDim}, NCHW);
                registerModel({query, key, value, out});
            }

            std::vector<Express::VARP> LLaMASelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                std::vector<int> inputDim = x->getInfo()->dim;
                VARP freqsComplex = inputs[1];

                auto mixedQueryLayer = query->forward(x);
                auto mixedKeyLayer = key->forward(x);
                auto mixedValueLayer = value->forward(x);

                auto shape = mixedKeyLayer->getInfo()->dim;
                auto queryLayer = _Reshape(mixedQueryLayer, {shape[0], shape[1], numHeads, headDim});
                auto keyLayer = _Reshape(mixedKeyLayer, {shape[0], shape[1], numKVHeads, headDim});
                auto valueLayer = _Reshape(mixedValueLayer, {shape[0], shape[1], numKVHeads, headDim});

                queryLayer = applyRotaryEmbedding(queryLayer, freqsComplex);
                keyLayer = applyRotaryEmbedding(keyLayer, freqsComplex);

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
                contextLayer = _Reshape(contextLayer, {newContextLayerShape[0], newContextLayerShape[1], newContextLayerShape[2] * newContextLayerShape[3]});

                // Variable::prepareCompute({contextLayer});
                auto output = out->forward(contextLayer);
                return { output };
            }

            LLaMAParallelSelfAttention::LLaMAParallelSelfAttention(LLaMAArgs &args, std::vector<Confidant::ProcessorInfo> allocationStrategy) {
                this->numKVHeads = args.nKVHeads == -1 ? args.nHeads : args.nKVHeads;
                this->numHeads = args.nHeads;
                this->hiddenSize = args.hiddenSize;
                this->headDim = args.hiddenSize / args.nHeads;
                this->allocationStrategy = allocationStrategy;

                // enable lora
                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[0]) {
                    this->query = std::make_shared<LLaMALoRALayer>(args, hiddenSize, numHeads * headDim);
                } else {
                    this->query.reset(NN::Linear(hiddenSize, numHeads * headDim, false));
                }
                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[1]) {
                    this->key = std::make_shared<LLaMALoRALayer>(args, hiddenSize, this->numKVHeads * headDim);
                } else {
                    this->key.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));
                }
                if (args.loraArgs.r > 0 && args.loraArgs.enableLoRA[2]) {
                    this->value = std::make_shared<LLaMALoRALayer>(args, hiddenSize, this->numKVHeads * headDim);
                } else {
                    this->value.reset(NN::Linear(hiddenSize, this->numKVHeads * headDim, false));
                }

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
//                        for (int i = 0; i < curNumAttentionHead; i++) {
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
//                        }
                        parallelKey.emplace_back(NN::Linear(hiddenSize, headDim, false));
                        parallelQuery.emplace_back(NN::Linear(hiddenSize, headDim, false));
                        parallelValue.emplace_back(NN::Linear(hiddenSize, headDim, false));
                    }
                    allocatedAttnHeads += curNumAttentionHead;
                }

                MNN_ASSERT(this->numHeads == allocatedAttnHeads);

                registerModel(parallelKey);
                registerModel(parallelQuery);
                registerModel(parallelValue);
                registerModel({out});
            }

            std::vector<Express::VARP>
            LLaMAParallelSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
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
                                MNN_PRINT("Backend %d with %d threads not available, falling back to CPU\n", curType, curNumThread);
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

            LLaMAFeedForward::LLaMAFeedForward(int hiddenSize, int multipleOf, int ffnDimMultiplier) {
                int hiddenDim = hiddenSize * 4;
                hiddenDim = 2 * hiddenDim / 3;
                if (ffnDimMultiplier != -1) {
                    hiddenDim = hiddenSize * ffnDimMultiplier;
                }
                // Round the hidden_dim to the nearest multiple of the multipleOf parameter
                hiddenDim = multipleOf * ((hiddenDim + multipleOf - 1) / multipleOf);

                hiddenDim = 14336; // intermediate size for llama3

                w1.reset(NN::Linear(hiddenSize, hiddenDim, false));
                w2.reset(NN::Linear(hiddenDim, hiddenSize, false));
                w3.reset(NN::Linear(hiddenSize, hiddenDim, false));

                registerModel({w1, w2, w3});
            }

            std::vector<Express::VARP> LLaMAFeedForward::onForward(const std::vector<Express::VARP> &inputs) {
                VARP x = inputs[0];
                auto w1X = w1->forward(x);
                // (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
                auto swish = _Multiply(w1X, _Sigmoid(w1X));

                // (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
                auto xV = w3->forward(x);
                // (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
                auto output = swish * xV;
                // (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
                output = w2->forward(output);

                return {output};
            }
        }
    }
}
