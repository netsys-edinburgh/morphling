//
// Created by gsw on 2023/11/12.
//
#include "GPT2.h"
#include "GPT2Layer.h"
#include "MNN/AutoTime.hpp"
#include "MLayerNorm.h"
#include <memory>
#include "log.h"
#include "commonStates.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            /*
             * Measure the computation time of GPT2 self-attention mechanism using different number of attention heads
             */
            std::unordered_map<MNNForwardType, std::vector<float>> GPT2ProfileProcessors(int batchSize, std::map<std::string, double> &modelArgs) {
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

                            auto queryLayer = query->forward(x);
                            auto keyLayer = key->forward(x);
                            auto valueLayer = value->forward(x);

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

                            auto contextLayer = _MatMul(attentionProbs, valueLayer);
                            // AUTOTIME;
                            MNN::Timer _100Time;
                            auto ptr = contextLayer->readMap<float>();
                            float curTime = (float)_100Time.durationInUs() / 1000.0f;
                            timeSum += curTime;
                            LOGI("GPT2ProfileProcessors():Type %d, BatchSize: %d, SeqLen: %d, HiddenSize: %d, AttentionHeads: %d, Time: %f\n",
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

            float GPT2ProfileBlock(std::map<std::string, double> &modelArgs, int numBlocks) {
                int batchSize = Confidant::CommonStates::getBatchSize();
                int hiddenSize = (int) modelArgs["hidden_size"];
                int numAttentionHeads = (int) modelArgs["num_attention_heads"];
                int intermediateSize = (int) modelArgs["intermediate_size"];
                int forParallel = (bool) modelArgs["for_parallel"];
                int seqLen = 128;

                auto block =  std::make_shared<GPT2Layer>(numAttentionHeads, hiddenSize, intermediateSize, forParallel);

                int repeatTime = 5;
                float totalTime = 0.0f;
                for (int k = 0; k < repeatTime; k++) {
                    auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);

                    MNN::Timer _100Time;
                    auto output = block->onForward({x});

                    for (int i = 0; i < numBlocks - 1; i++) {
                        output = block->onForward({output[0]});
                    }

                    auto ptr = output[0]->readMap<float>();
                    totalTime += (float)_100Time.durationInUs() / 1000.0f;
                }

                return totalTime / (1.0f * repeatTime);
            }

            std::vector<VARP> computeGPT2QALoss(std::vector<VARP> &logits, std::vector<VARP> &labels) {
                auto logit = logits[0];

                auto startPositions = labels[0];
                auto endPositions = labels[1];
                auto logitsSize = logit->getInfo()->dim;
                int seqLen = logitsSize[1];

                auto logitsArr = _Split(logit, {2}, 2);
                auto startLogits = _Squeeze(logitsArr[0], {-1});
                auto endLogits = _Squeeze(logitsArr[1], {-1});

                if (startPositions->getInfo()->dim.size() > 1) {
                    startPositions = _Squeeze(startPositions, {-1});
                }

                if (endPositions->getInfo()->dim.size() > 1) {
                    endPositions = _Squeeze(endPositions, {-1});
                }

                // TODO: clamp_ is ignored here
                int ignoreIndex = -1;

                auto activeStartLabel = _OneHot(_Cast<int32_t>(startPositions), _Scalar<int>(seqLen), _Scalar<float>(1.0f), _Scalar<float>(0.0f));
                auto activeEndLabel = _OneHot(_Cast<int32_t>(endPositions), _Scalar<int>(seqLen), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

                auto startLoss = _CrossEntropy(_Softmax(startLogits), activeStartLabel);
                // auto startLossPtr = startLoss->readMap<float>();
                auto endLoss = _CrossEntropy(_Softmax(endLogits), activeEndLabel);
                auto loss = (startLoss + endLoss) * _Scalar(0.5f);

                return {loss};
            }

            // embeddings + 12layers +n layerNorm
            GPT2Encoder::GPT2Encoder(int vocabSize, int maxPositionEmbeddings, int numHiddenLayers,
                                     int numAttentionHeads, int hiddenSize, int intermediateSize, bool forParallel) {
                for (int i = 0; i < numHiddenLayers; i++) {
                    layers.push_back(
                            std::make_shared<GPT2Layer>(numAttentionHeads, hiddenSize, intermediateSize, forParallel));
                }
                layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-5));//12  ->5
                Embeddings = std::make_shared<GPT2Embedding>(vocabSize, hiddenSize, maxPositionEmbeddings);//12  ->5


                registerModel(layers);//12层block
                registerModel({layerNorm});//13层norm
            }

            std::vector<Express::VARP> GPT2Encoder::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = Embeddings->forward(inputs[0]);
                for (int i = 0; i < layers.size(); i++) {//normal 12 24 36 48
                    hiddenStates = layers[i]->onForward({hiddenStates })[0];
                }

                hiddenStates = layerNorm->onForward({hiddenStates})[0];

                return {hiddenStates};
            }
            //12 *encoders
            GPT2layers::GPT2layers(int numHiddenLayers, int numAttentionHeads, int hiddenSize, int intermediateSize, bool forParallel) {
                for (int i = 0; i < numHiddenLayers; i++) {
                    layer.push_back(
                            std::make_shared<GPT2Layer>(numAttentionHeads, hiddenSize, intermediateSize, forParallel));
                }
                registerModel(layer);//12层block
            }

            std::vector<Express::VARP> GPT2layers::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = inputs[0];

                for (int i = 0; i < layer.size(); i++) {//normal 12 24 36 48
                    hiddenStates = layer[i]->onForward({hiddenStates})[0];
                }
                return {hiddenStates};
            }

            //embeddings + 12layers +n layerNorm [+ dense 769->50257]
            GPT2::GPT2(GPT2Args& args) {
                transformer.reset(new GPT2Encoder(args.vocabSize, args.maxPositionEmbeddings,  args.numHiddenLayers,  args.numAttentionHeads,
                                                  args.hiddenSize,  args.intermediateSize, args.forParallel));

                dense.reset(NN::Linear(args.hiddenSize, args.vocabSize, false));
                registerModel({transformer, dense});

            };

            std::vector<Express::VARP> GPT2::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = transformer->onForward({inputs[0]})[0];
                hiddenStates = dense->onForward({hiddenStates})[0];

                return {hiddenStates};
            }

            // train test for model GPT@ dataset == QA
            // embeddings + 12layers +n layerNorm [+ dense 769->2]
            GPT2QAModel::GPT2QAModel(GPT2Args& args) {
                hiddenSize = args.hiddenSize;
                transformer = std::make_shared<GPT2Encoder>(args.vocabSize, args.maxPositionEmbeddings,  args.numHiddenLayers,  args.numAttentionHeads,
                                                  args.hiddenSize,  args.intermediateSize, args.forParallel);
                dense.reset(NN::Linear(args.hiddenSize, QA_dim, true));
                registerModel({transformer, dense});
            };

            void GPT2QAModel::loadParam(std::string &weightsBasePath, bool isTrainable) {
                LOGI("Loading GPT2 weights from %s", weightsBasePath.c_str());
                int encoderNum = this->transformer->layers.size();
                for (int i = 0; i < encoderNum + 2; i++) {
                    if (i == 0) {
                        std::string path = weightsBasePath + "gpt2_embeddings.mnn";
                        auto params = Variable::load(path.c_str());
                        if (params.empty()) {
                            LOGI("Load GPT2 embeddings failed");
                            continue;
                        }

                        transformer->Embeddings->loadParameters(params);
                    } else if (i == encoderNum + 1) {
                        std::string path = weightsBasePath + "gpt2_norm.mnn";
                        auto params = Variable::load(path.c_str());

                        if (params.empty()) {
                            LOGI("Load GPT2 norm failed");
                            continue;
                        }
                        for (auto &para: params) {
                            para.fix(VARP::TRAINABLE);
                        }
                        transformer->layerNorm->loadParameters(params);
                    } else {
                        std::string path = weightsBasePath + "gpt2_layer_" + std::to_string(i - 1) + ".mnn";
                        auto params = Variable::load(path.c_str());

                        if (params.empty()) {
                            LOGI("Load GPT2 layer (encoder) %d failed", i);
                            continue;
                        }

                        for (auto &para: params) {
                            para.fix(VARP::TRAINABLE);
                        }
                        transformer->layers[i - 1]->loadParameters(params);
                    }
                }
            }

            std::vector<VARP> GPT2QAModel::getLoss(std::vector<VARP> &logits, std::vector<VARP> &labels) {
                return computeGPT2QALoss(logits, labels);
            }

            std::vector<Express::VARP> GPT2QAModel::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = transformer->onForward({inputs[0]})[0];
                hiddenStates = dense->onForward({hiddenStates})[0];
                return {hiddenStates};
            }

            SubGPT2::SubGPT2(int start, int end, std::map<std::string, double> &modelArgs) {
                this->numHiddenLayers = (int)modelArgs["num_hidden_layers"];
                this->totalLayers = (int)modelArgs["total_layer"];

                GPT2Args gpt2Args;
                gpt2Args.vocabSize = (int)modelArgs["vocab_size"];
                gpt2Args.maxPositionEmbeddings = (int)modelArgs["max_position_embeddings"];
                gpt2Args.numHiddenLayers = (int)modelArgs["num_hidden_layers"];
                gpt2Args.numAttentionHeads = (int)modelArgs["num_attention_heads"];
                gpt2Args.hiddenSize = (int)modelArgs["hidden_size"];
                gpt2Args.intermediateSize = (int)modelArgs["intermediate_size"];
                gpt2Args.dropoutProb = (float)modelArgs["hidden_dropout_prob"];
                gpt2Args.forParallel = (bool)modelArgs["for_parallel"];

                if (end == -1) {
                    end = this->totalLayers - 1;
                }


                if (start == 0) {
                    this->embedding = std::make_shared<GPT2Embedding>(gpt2Args.vocabSize, gpt2Args.hiddenSize, gpt2Args.maxPositionEmbeddings, 0.0);
                    registerModel({this->embedding});
                    this->layers.push_back(this->embedding);
                }

                int curLayer = 1;
                for (int i = 0; i < this->numHiddenLayers; i++) {
                    if(curLayer >= start && curLayer <= end){
                        auto decoder = std::make_shared<GPT2Layer>(gpt2Args.numAttentionHeads,
                                                                   gpt2Args.hiddenSize, gpt2Args.intermediateSize, gpt2Args.forParallel);
                        this->decoders.push_back(decoder);
                        this->layers.push_back(decoder);
                    }
                    curLayer++;
                }

                if (!decoders.empty()) {
                    registerModel(decoders);
                }

                if (end >= this->totalLayers-2) {
                    this->norm = std::make_shared<MLayerNorm>(std::vector<int>{gpt2Args.hiddenSize}, true, 1e-5);
                    registerModel({this->norm});
                    this->layers.push_back(this->norm);
                }

                if (end == this->totalLayers-1) {
                    this->dense.reset(NN::Linear(gpt2Args.hiddenSize, 2, true));
                    registerModel({this->dense});
                    this->layers.push_back(this->dense);
                }
            }

            float GPT2QAModel::getOutputDataSizeByIdx(int idx, int batchSize, int seqLen) {
                if (idx == transformer->layers.size() + 2) {
                    // final output
                    return batchSize * seqLen * 2 * 1.0 * 4;
                }

                return batchSize * seqLen * hiddenSize * 1.0 * 4;
            }

            float GPT2QAModel::getModelTimeByIdx(int idx, int batchSize, int seqLen) {
                if (idx == 0) {
                    // embedding
                    std::vector<int> xData(seqLen, 1);
                    // std::vector<int> xData = {19895, 5712, 20221, 11, 262, 1524, 468, 257, 7835, 2095, 13, 1629, 404, 262, 8774, 11819, 338, 3869, 29500, 318, 257, 10861, 15207, 286, 262, 5283, 5335, 13, 34528, 287, 2166, 286, 262, 8774, 11819, 290, 6476, 340, 11, 318, 257, 15317, 15207, 286, 1951, 351, 5101, 510, 49309, 351, 262, 8177, 366, 37522, 578, 1215, 2185, 16543, 2516, 1911, 7406, 284, 262, 8774, 11819, 318, 262, 32520, 3970, 286, 262, 17380, 8894, 13, 34528, 2157, 262, 37792, 3970, 318, 262, 10299, 33955, 11, 257, 37919, 1295, 286, 11443, 290, 14580, 13, 632, 318, 257, 30069, 286, 262, 7128, 33955, 379, 406, 454, 8906, 11, 4881, 810, 262, 5283, 5335, 1128, 7241, 306, 4120, 284, 9281, 6206, 324, 5857, 311, 12944, 343, 516, 287, 1248, 3365, 13, 1629, 262, 886, 286, 262, 1388, 3708, 357, 392, 287, 257, 1277, 1627, 326, 20417, 832, 513, 25827, 290, 262, 3561, 31390, 828, 318, 257, 2829, 11, 3660, 7815, 15207, 286, 5335, 13};
                    VARP x = _Const(xData.data(), {batchSize, seqLen}, NCHW, halide_type_of<int>());
                    auto ptr = x->readMap<int>();
                    MNN::Timer _100Time;
                    auto embeddingOutput = transformer->Embeddings->onForward({x});
                    auto embeddingOutputPtr = embeddingOutput[0]->readMap<float>();
                    auto time = _100Time.durationInUs() / 1000.0f;
                    return time;
                }

                if (idx == transformer->layers.size() + 1) {
                    // norm
                    VARP x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                    MNN::Timer _100Time;
                    auto output = transformer->layerNorm->onForward({x});
                    auto outputPtr = output[0]->readMap<float>();
                    auto time = _100Time.durationInUs() / 1000.0f;
                    return time;
                }
                if (idx == transformer->layers.size() + 2) {
                    // dense
                    VARP x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                    MNN::Timer _100Time;
                    auto output = dense->onForward({x});
                    auto outputPtr = output[0]->readMap<float>();
                    auto time = _100Time.durationInUs() / 1000.0f;
                    return time;
                }

                // encoder
                auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);
                MNN::Timer _100Time;
                auto encoderOutput = transformer->layers[0]->onForward({x});
                auto encoderOutputPtr = encoderOutput[0]->readMap<float>();
                auto time = _100Time.durationInUs() / 1000.0f;
                return time;
            }

            std::vector<Express::VARP>
            SubGPT2::onForward(const std::vector<Express::VARP> &inputs) {
                VARP h = inputs[0];

                if (this->embedding != nullptr){
                    h = this->embedding->onForward({h})[0];
                }

                for (const auto & decoder : this->decoders){
                    h = decoder->onForward({h})[0];
                }

                if (this->norm != nullptr){
                    h = this->norm->onForward({h})[0];
                }

                if (this->dense != nullptr){
                    h = this->dense->onForward({h})[0];
                }

                return {h};
            }

            void SubGPT2::loadParamByLayer(int layer, std::string &weightsBasePath, int startLayer, bool isTrainable) {
                if(layer > this->totalLayers-1 || layer < 0){
                    MNN_ERROR("layer index out of range");
                    return;
                }
//
                if (startLayer == 0) {
                    LOGI("SubGPT2: Setting embedding to fixed\n");
                    std::string gpt2EmbeddingParamsPath = weightsBasePath + "gpt2_embedding.mnn";
                    auto gpt2EmbeddingParams = Variable::load(gpt2EmbeddingParamsPath.c_str());
                    this->embedding->loadParameters(gpt2EmbeddingParams);
                }
//
                startLayer = (startLayer==0) ? 1: startLayer;

                if(layer == 0) {
                    std::string gpt2EmbeddingParamsPath = weightsBasePath + "gpt2_embedding.mnn";
                    auto gpt2EmbeddingParams = Variable::load(gpt2EmbeddingParamsPath.c_str());
                    this->embedding->loadParameters(gpt2EmbeddingParams);
//                } else if (layer-startLayer < this->numHiddenLayers) {
//                    std::string gpt2DecoderParamsPath = weightsBasePath + "gpt2_decoder_" +
//                                                        std::to_string(layer - startLayer) + ".mnn";
//                    auto gpt2DecoderParams = Variable::load(gpt2DecoderParamsPath.c_str());
//                    this->decoders[layer - startLayer]->loadParameters(gpt2DecoderParams);
//                } else if (layer == this->totalLayers-2) {
//                    std::string gpt2NormParamsPath = weightsBasePath + "gpt2_norm.mnn";
//                    auto gpt2NormParams = Variable::load(gpt2NormParamsPath.c_str());
//                    this->norm->loadParameters(gpt2NormParams);
                }
            }

            std::vector<VARP> SubGPT2::getParamsByLayer(int layer, bool isTrainable) {
                if (layer<0 || layer>this->totalLayers-1){
                    MNN_ERROR("layer index out of range");
                    return {};
                }

                return this->layers[layer]->parameters();
            }

            std::vector<VARP>
            SubGPT2::getLoss(std::vector<VARP> &logits, std::vector<VARP> &labels) {
                return computeGPT2QALoss(logits, labels);
            }

        }
    }
}