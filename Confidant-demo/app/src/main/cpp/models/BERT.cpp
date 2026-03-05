//
// Created by Yuhao Chen on 2023/6/12.
//
#include "BERT.h"
#include "BERTLayer.h"
#include "MNN/AutoTime.hpp"
#include <memory>
#include "Loss.hpp"
#include "log.h"
#include "commonStates.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            /*
             * Measure the computation time of BERT self-attention mechanism using different number of attention heads
             */
            std::unordered_map<MNNForwardType, std::vector<float>> BERTProfileProcessors(int batchSize, std::map<std::string, double> &modelArgs) {
                auto exe = Executor::getGlobalExecutor();
                MNN::BackendConfig config;
                config.precision = MNN::BackendConfig::Precision_High;

//                std::vector<std::pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_VULKAN, 1}, {MNN_FORWARD_CPU, 1}};
                std::vector<std::pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_CPU, 1}};
                std::vector<std::pair<MNNForwardType, int> > availableTypes;

                for (auto& type : types) {
                    exe->setGlobalExecutorConfig(type.first, config, type.second);
                }

                // check available types
                auto availableBackends = exe->getAvailableBackends();

                int hiddenSize = (int) modelArgs["hidden_size"];
                int numAttentionHeads = (int) modelArgs["num_attention_heads"];
                int seqLen = 128;
                int attentionHeadSize = hiddenSize / numAttentionHeads;
                int repeatTime = 5;

                std::shared_ptr<Module> query, key, value, dropout;
                std::unordered_map<MNNForwardType, std::vector<float>> profilingResult;

                for (auto& type :types) {
                    if (availableBackends.find(type) == availableBackends.end()) {
                        // if the backend is not available, set all to inf
                        profilingResult[type.first] = std::vector<float>(numAttentionHeads, std::numeric_limits<float>::infinity());
                        continue;
                    }

                    exe->setGlobalExecutorConfig(type.first, config, type.second);

                    for (int curHead = 1; curHead <= numAttentionHeads; curHead++) {
                        float timeSum = 0.0f;
                        for (int k = 0; k < repeatTime; k++) {
                            key.reset(NN::Linear(hiddenSize, attentionHeadSize * curHead, true));
                            value.reset(NN::Linear(hiddenSize, attentionHeadSize * curHead, true));
                            query.reset(NN::Linear(hiddenSize, attentionHeadSize * curHead, true));
                            dropout.reset(NN::Dropout(0));

                            auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                            auto attentionMask = _Const(1.0f, {batchSize, 1, seqLen}, NCHW);

                            auto queryLayer = query->forward(x);
                            auto keyLayer = key->forward(x);
                            auto valueLayer = value->forward(x);

                            auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                            attentionScores = _Divide(attentionScores, _Const((float) sqrt(attentionHeadSize), {}, NCHW));
                            attentionScores = attentionScores + _Clone(attentionMask, true);

                            // Normalize the attention scores to probabilities.
                            auto attentionProbs = _Softmax(attentionScores, -1);

                            attentionProbs = dropout->forward(attentionProbs);

                            auto contextLayer = _MatMul(attentionProbs, valueLayer);
                            // AUTOTIME;
                            MNN::Timer _100Time;
                            auto ptr = contextLayer->readMap<float>();
                            float curTime = (float)_100Time.durationInUs() / 1000.0f;
                            timeSum += curTime;
                            LOGI("BERTProfileProcessors():Type %d, BatchSize: %d, SeqLen: %d, HiddenSize: %d, AttentionHeads: %d, Time: %f\n",
                                 type.first, batchSize, seqLen, hiddenSize, curHead, curTime);
                            _100Time.reset();
                        }
                        profilingResult[type.first].push_back(timeSum / (1.0f * repeatTime));
                    }
                }

                // compare the OpenCL and Vulkan and choose one of them
                if (profilingResult.find(MNN_FORWARD_OPENCL) != profilingResult.end() && profilingResult.find(MNN_FORWARD_VULKAN) != profilingResult.end()) {
                    if (profilingResult[MNN_FORWARD_OPENCL][0] < profilingResult[MNN_FORWARD_VULKAN][0]) {
                        profilingResult.erase(MNN_FORWARD_VULKAN);
                    } else {
                        profilingResult.erase(MNN_FORWARD_OPENCL);
                    }
                }

                // here CPU should be set again such that the main processor used will be CPU
                exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);

                return profilingResult;
            }

            float BERTProfileBlock(std::map<std::string, double> &modelArgs, int numBlocks) {
                int batchSize = Confidant::CommonStates::getBatchSize();
                int hiddenSize = (int) modelArgs["hidden_size"];
                int numAttentionHeads = (int) modelArgs["num_attention_heads"];
                int intermediateSize = (int) modelArgs["intermediate_size"];
                float hiddenDropout = (float) modelArgs["hidden_dropout_prob"];
                int forParallel = (bool) modelArgs["for_parallel"];
                int seqLen = 128;

                auto block = std::make_shared<BERTLayer>(numAttentionHeads, hiddenSize,
                                            intermediateSize, hiddenDropout,
                                            forParallel);

                int repeatTime = 5;
                float totalTime = 0.0f;
                for (int k = 0; k < repeatTime; k++) {
                    auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                    auto attentionMask = _Const(1.0f, {batchSize, 1, 1, seqLen}, NCHW);

                    MNN::Timer _100Time;
                    auto output = block->onForward({x, attentionMask});
                    for (int i = 0; i < numBlocks - 1; i++) {
                        output = block->onForward({output[0], attentionMask});
                    }
                    auto ptr = output[0]->readMap<float>();
                    totalTime += (float)_100Time.durationInUs() / 1000.0f;
                }

                return totalTime / (1.0f * repeatTime);
            }

            std::vector<VARP> computeBERTForClassificationLoss(std::vector<VARP> &logits,
                                                               std::vector<VARP> &labels) {
                // attention mask is in logits[1]
                assert(logits.size() == 2);

                // calculate the attention mask idx
                std::vector<int> maskVecIdx;
                auto logit = logits[0];

                auto attentionMask = logits[1];
                auto attentionMaskPtr = attentionMask->readMap<float>();
                std::vector<int> dims = attentionMask->getInfo()->dim;
                for (int i = 0; i < dims[0]; i++) {
                    for (int j = 0; j < dims[1]; j++) {
                        if (attentionMaskPtr[i * dims[1] + j] != 0.0) {
                            maskVecIdx.push_back(i * dims[1] + j);
                        }
                    }
                }

                auto activeLoss = _Const(maskVecIdx.data(), {(int)maskVecIdx.size()}, NCHW, halide_type_of<int>());
                auto activeLogits = _GatherV2(_Reshape(logit, {-1, logit->getInfo()->dim[2]}), activeLoss, _Scalar<int>(0));
                auto activeLabels = _GatherV2(_Reshape(labels[0], {-1}), activeLoss, _Scalar<int>(0));
                auto newActiveLabels = _OneHot(_Cast<int32_t>(activeLabels), _Scalar<int>(9), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

                int ignoreIndex = -100;
                auto ignoredMask = _Cast<float>(_Unsqueeze(_NotEqual(activeLabels, _Scalar<int>(ignoreIndex)), {1}));
                newActiveLabels = newActiveLabels * ignoredMask;

                auto loss    = _CrossEntropy(_Softmax(activeLogits), newActiveLabels);
                return {loss};
            }

            BERTPooler::BERTPooler(int hiddenSize) {
                dense.reset(NN::Linear(hiddenSize, hiddenSize, true));
                registerModel({dense});
            }

            std::vector<Express::VARP>
            BERTPooler::onForward(const std::vector<Express::VARP> &inputs) {
                // We "pool" the model by simply taking the hidden state corresponding
                // to the first token.
                using namespace Express;
                VARP hiddenStates = inputs[0];

                int startSlice[] = {0, 0, 0};
                int sizeSlice[] = {-1, 1, -1};

                auto firstTokenTensor = _Slice(hiddenStates,
                                               _Const(startSlice, {3}, NCHW, halide_type_of<int>()),
                                               _Const(sizeSlice, {3}, NCHW,
                                                      halide_type_of<int>()));
                auto pooledOutput = dense->forward(firstTokenTensor);
                pooledOutput = _Tanh(pooledOutput);

                return {pooledOutput};
            }


            BERTEncoder::BERTEncoder(int numHiddenLayers, int numAttentionHeads, int hiddenSize,
                                     int intermediateSize, float dropoutProb, bool forParallel) {
                for (int i = 0; i < numHiddenLayers; i++) {
                    layer.push_back(std::make_shared<BERTLayer>(numAttentionHeads, hiddenSize,
                                                                intermediateSize, dropoutProb,
                                                                forParallel));
                }

                registerModel(layer);
            }

            std::vector<Express::VARP>
            BERTEncoder::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = inputs[0];
                VARP attentionMask = inputs[1];

                // TODO: outputAllEncodedLayers is ignored here
                for (int i = 0; i < layer.size(); i++) {
                    hiddenStates = layer[i]->onForward({hiddenStates, attentionMask})[0];
                }

                return {hiddenStates};
            }

            BERT::BERT(int vocabSize, int hidden, int nLayers, int attnHeads, int intermediateSize,
                       float dropout, bool forParallel) {
                this->embedding = std::make_shared<BERTEmbedding>(vocabSize, hidden, 512, 2,
                                                                  dropout);
                this->encoder = std::make_shared<BERTEncoder>(nLayers, attnHeads, hidden,
                                                              intermediateSize, dropout,
                                                              forParallel);
                this->pooler = std::make_shared<BERTPooler>(hidden);

                registerModel({embedding, encoder, pooler});
            }

            std::vector<Express::VARP> BERT::onForward(const std::vector<Express::VARP> &inputs) {
                // {inputsIds, tokenTypeIds, attentionMask, outputAllEncodedLayers}
                VARP inputIds = inputs[0];
                VARP tokenTypeIds;
                VARP attentionMask;

                if (inputs.size() < 2) {
                    // attentionMask is None
                    // We assume that the attention mask is None only for the sub-model on the central node
                    std::vector<float> maskVec;
                    auto inputPtr = inputs[0]->readMap<int>();
                    std::vector<int> dims = inputs[0]->getInfo()->dim;
                    for (int i = 0; i < dims[0]; i++) {
                        for (int j = 0; j < dims[1]; j++) {
                            if (inputPtr[i * dims[1] + j] != 0) {
                                maskVec.push_back(1.0);
                            } else {
                                maskVec.push_back(0.0);
                            }
                        }
                    }
                    attentionMask = _Const(maskVec.data(), inputs[0]->getInfo()->dim, NCHW, halide_type_of<float>());
                } else {
                    attentionMask = inputs[1];
                }

                if (inputs.size() < 3) {
                    // tokenTypeIds is None
                    std::vector<int> allZeros(inputIds->getInfo()->size, 0);
                    tokenTypeIds = _Const(allZeros.data(), inputIds->getInfo()->dim, NCHW,
                                          halide_type_of<int>());
                } else {
                    tokenTypeIds = inputs[2];
                }

                auto extendedAttentionMask = _Unsqueeze(attentionMask, {1, 2});
                extendedAttentionMask =(_Scalar<float>(1.0) - extendedAttentionMask) * _Scalar<float>(-10000.0);

                auto embeddingOutput = embedding->onForward({inputIds, tokenTypeIds});
                // auto ptr = embeddingOutput[0]->readMap<float>();
                auto encoderOutput = encoder->onForward(
                        {embeddingOutput[0], extendedAttentionMask});
                // auto encoderPtr = encoderOutput[0]->readMap<float>();
                auto poolerOutput = pooler->onForward({encoderOutput[0]});
                // auto poolerPtr = poolerOutput[0]->readMap<float>();
                return {encoderOutput[0], attentionMask};
            }

            BERTForClassification::BERTForClassification(int vocabSize, int hidden, int nLayers,
                                                         int attnHeads,
                                                         int intermediateSize,
                                                         float attentionDropout,
                                                         float hiddenDropout, int numClasses,
                                                         bool forParallel) {
                this->nLayers = nLayers;
                this->hiddenSize = hidden;
                this->bert = std::make_shared<BERT>(vocabSize, hidden, nLayers, attnHeads,
                                                    intermediateSize, attentionDropout,
                                                    forParallel);
                this->dropout.reset(NN::Dropout(hiddenDropout));
                this->classifier.reset(NN::Linear(hidden, numClasses, true));
                registerModel({bert, dropout, classifier});
            }

            std::vector<Express::VARP>
            BERTForClassification::onForward(const std::vector<Express::VARP> &inputs) {
                auto sequenceOutput = bert->onForward(inputs);
                auto output = dropout->onForward(sequenceOutput);
                output = classifier->onForward({output[0]});
                return {output[0], sequenceOutput[1]};
            }

            void BERTForClassification::loadParam(std::string &weightsBasePath, bool isTrainable) {
                LOGI("Loading BERT weights from %s", weightsBasePath.c_str());
                for (int i = 0; i < this->nLayers + 2; i++) {
                    std::string path = weightsBasePath + "bert_" + std::to_string(i) + ".mnn";
                    auto params = Variable::load(path.c_str());

                    if (i == 0) {
                        bert->embedding->loadParameters(params);
                    } else if (i == this->nLayers + 1) {
                        for (auto &para: params) {
                            para.fix(VARP::TRAINABLE);
                        }
                        bert->pooler->loadParameters(params);
                    } else {
                        for (auto &para: params) {
                            para.fix(VARP::TRAINABLE);
                        }
                        bert->encoder->layer[i - 1]->loadParameters(params);
                    }
                }
            }

            std::vector<VARP> BERTForClassification::getLoss(std::vector<VARP> &logits, std::vector<VARP> &labels) {
                return computeBERTForClassificationLoss(logits, labels);
            }

            float BERTForClassification::getOutputDataSizeByIdx(int idx, int batchSize, int seqLen) {
                // 4 bytes for float
                return batchSize * seqLen * hiddenSize * 1.0 * 4;
            }

            float BERTForClassification::getModelTimeByIdx(int idx, int batchSize, int seqLen) {
                if (idx == 0) {
                    // embedding
                    VARP x = _Input({batchSize, seqLen}, NCHW, halide_type_of<int>());
                    std::vector<int> allZeros(x->getInfo()->size, 0);
                    VARP tokenTypeIds = _Const(allZeros.data(), x->getInfo()->dim, NCHW,
                                          halide_type_of<int>());

                    MNN::Timer _100Time;
                    auto embeddingOutput = bert->embedding->onForward({x, tokenTypeIds});
                    auto embeddingOutputPtr = embeddingOutput[0]->readMap<float>();
                    auto time = _100Time.durationInUs() / 1000.0f;
                    return time;
                }

                // encoder
                auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                auto attentionMask = _Const(1.0f, {batchSize, 1, 1, seqLen}, NCHW);
                MNN::Timer _100Time;
                auto encoderOutput = bert->encoder->layer[0]->onForward({x, attentionMask});
                auto encoderOutputPtr = encoderOutput[0]->readMap<float>();
                auto time = _100Time.durationInUs() / 1000.0f;
                return time;
            }

            SubBERTForClassification::SubBERTForClassification(int start, int end,
                                                               std::map<std::string, double> &args) {
                this->totalLayers = (int) args["total_layer"];
                int numClasses = (int) args["n_class"];
                int hidden = (int) args["hidden_size"];
                int vocabSize = (int) args["vocab_size"];
                this->numHiddenLayers = (int) args["num_hidden_layers"];

                float attentionDropout = (float) args["attention_dropout_prob"];
                float hiddenDropout = (float) args["hidden_dropout_prob"];

                int numAttentionHeads = (int) args["num_attention_heads"];
                int intermediateSize = (int) args["intermediate_size"];

                int forParallel = (bool) args["for_parallel"];

                encoders = std::vector<std::shared_ptr<Express::Module> >();

                if (end == -1) {
                    end = totalLayers - 1;
                }

                if (start == 0) {
                    this->embedding = std::make_shared<BERTEmbedding>(vocabSize, hidden, 512, 2,
                                                                      attentionDropout);
                    registerModel({this->embedding});
                    layers.emplace_back(this->embedding);
                }

                this->startLayer = start;
                this->endLayer = end;

                int curLayer = 1;
                for (int i = 0; i < numHiddenLayers; i++) {
                    if (curLayer >= start && curLayer <= end) {
                        auto encoder = std::make_shared<BERTLayer>(numAttentionHeads, hidden,
                                                                   intermediateSize,attentionDropout, forParallel);
                        encoders.push_back(encoder);
                        layers.emplace_back(encoder);
                    }
                    curLayer++;
                }

                if (!encoders.empty()) {
                    registerModel(encoders);
                }

                int poolerLayerIdx = numHiddenLayers + 1;
                if (poolerLayerIdx >= start && poolerLayerIdx <= end) {
                    this->pooler = std::make_shared<BERTPooler>(hidden);
                    registerModel({this->pooler});
                    layers.emplace_back(this->pooler);
                }

                if (end == totalLayers - 1) {
                    this->dropout.reset(NN::Dropout(hiddenDropout));
                    this->classifier.reset(NN::Linear(hidden, numClasses, true));
                    registerModel({this->dropout, this->classifier});
                    layers.emplace_back(this->dropout);
                    layers.emplace_back(this->classifier);
                }
            }

            std::vector<Express::VARP>
            SubBERTForClassification::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                VARP attentionMask;

                if (inputs.size() < 2) {
                    // attentionMask is None
                    // We assume that the attention mask is None only for the sub-model on the central node
                    std::vector<float> maskVec;
                    auto inputPtr = inputs[0]->readMap<int>();
                    std::vector<int> dims = inputs[0]->getInfo()->dim;
                    for (int i = 0; i < dims[0]; i++) {
                        for (int j = 0; j < dims[1]; j++) {
                            if (inputPtr[i * dims[1] + j] != 0) {
                                maskVec.push_back(1.0);
                            } else {
                                maskVec.push_back(0.0);
                            }
                        }
                    }
                    attentionMask = _Const(maskVec.data(), inputs[0]->getInfo()->dim, NCHW, halide_type_of<float>());
                } else {
                    attentionMask = inputs[1];
                }

                if (embedding != nullptr) {
                    VARP tokenTypeIds;

                    if (inputs.size() < 3) {
                        // tokenTypeIds is None
                        std::vector<int> allZeros(x->getInfo()->size, 0);
                        tokenTypeIds = _Const(allZeros.data(), x->getInfo()->dim, NCHW,
                                              halide_type_of<int>());
                    } else {
                        tokenTypeIds = inputs[2];
                    }
                    x = embedding->onForward({x, tokenTypeIds})[0];
                }

                if (!encoders.empty()) {
                    auto extendedAttentionMask = _Unsqueeze(attentionMask, {1, 2});
                    extendedAttentionMask = (_Scalar<float>(1.0) - extendedAttentionMask) * _Scalar<float>(-10000.0);

                    for (const auto &encoder: encoders) {
                        x = encoder->onForward({x, extendedAttentionMask})[0];
                    }
                }

                if (classifier != nullptr) {
                    x = dropout->onForward({x})[0];
                    x = classifier->onForward({x})[0];
                }

                return {x, attentionMask};
            }

            void SubBERTForClassification::loadParamByLayer(int layer, std::string &weightsBasePath,
                                                            int startLayer, bool isTrainable) {
                if (layer >= this->totalLayers - 1) {
                    // No weights for classifier
                    return;
                }

                if (startLayer == 0) {
                    // Since the startLayer in Bert refers to the start of the encoder decoders, we need to add 1 to the startLayer
                    startLayer = 1;
                }
                std::string path = weightsBasePath + "bert_" + std::to_string(layer) + ".mnn";
                auto params = Variable::load(path.c_str());

                if (layer == 0) {
                    embedding->loadParameters(params);
                } else if (layer == this->numHiddenLayers + 2) {
                    classifier->loadParameters(params);
                } else if (layer == this->numHiddenLayers + 1) {
                    // pooler
                    for (auto &para: params) {
                        para.fix(VARP::TRAINABLE);
                    }
                    pooler->loadParameters(params);
                } else {
                    for (auto &para: params) {
                        para.fix(VARP::TRAINABLE);
                    }
                    encoders[layer - startLayer]->loadParameters(params);
                }
            }

            std::vector<VARP> SubBERTForClassification::getParamsByLayer(int layer, bool isTrainable) {
                int originLayer = this->startLayer + layer;
                std::vector<VARP> params;
                if (originLayer == 0) {
                    params = embedding->parameters();
                } else if (originLayer == this->numHiddenLayers + 2) {
                    params = classifier->parameters();
                } else if (originLayer == this->numHiddenLayers + 1) {
                    params = pooler->parameters();
                } else {
                    if (startLayer >= 1) {
                        params = encoders[layer]->parameters();
                    } else {
                        // embedding exist
                        params = encoders[layer - 1]->parameters();
                    }
                }

                if (!isTrainable) {
                    return params;
                }

                std::vector<VARP> trainableParams;
                for (auto &p: params) {
                    if (nullptr == p.get()) {
                        continue;
                    }
                    if (p->expr().first->get() != nullptr) {
                        continue;
                    }
                    if (p->expr().first->inputType() == Express::VARP::TRAINABLE) {
                        trainableParams.push_back(p);
                    }
                }
                return trainableParams;
            }

            std::vector<VARP> SubBERTForClassification::getLoss(std::vector<VARP> &logits,
                                                                std::vector<VARP> &labels) {
                return computeBERTForClassificationLoss(logits, labels);
            }
        }
    }
}