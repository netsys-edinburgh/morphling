//
// Created by Yuhao Chen on 2023/10/16.
//
#include "LLaMA.h"
#include <random>
#include "log.h"
#include "Phi2Layer.h"
#include "MNN/AutoTime.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            /*
             * Measure the computation time of LLamA self-attention mechanism using different number of attention heads
             */
            std::unordered_map<MNNForwardType, std::vector<float>> LLaMAProfileProcessors(int batchSize, std::map<std::string, double> &modelArgs) {
                auto exe = Executor::getGlobalExecutor();
                MNN::BackendConfig config;
                config.precision = MNN::BackendConfig::Precision_High;

                std::vector<std::pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_CPU, 1}};
                std::vector<std::pair<MNNForwardType, int> > availableTypes;

                for (auto& type : types) {
                    exe->setGlobalExecutorConfig(type.first, config, type.second);
                }

                // check available types
                auto availableBackends = exe->getAvailableBackends();

                int hiddenSize = (int) modelArgs["hidden_size"];
                int numAttentionHeads = (int) modelArgs["num_attention_heads"];
                int headDim = hiddenSize / numAttentionHeads;
                int numKVHeads = 32;
                int seqLen = 128;
                int attentionHeadSize = hiddenSize / numAttentionHeads;
                int repeatTime = 1;

                std::shared_ptr<Module> query, key, value, dropout;
                std::unordered_map<MNNForwardType, std::vector<float>> profilingResult;

                auto freqsComplex = precomputeThetaPosFrequencies(hiddenSize / numAttentionHeads, seqLen * 2);
                auto freqsComplexShape = freqsComplex->getInfo()->dim;

                for (auto& type :types) {
                    if (availableBackends.find(type) == availableBackends.end()) {
                        // if the backend is not available, set all to inf
                        profilingResult[type.first] = std::vector<float>(numAttentionHeads, std::numeric_limits<float>::infinity());
                        continue;
                    }

                    exe->setGlobalExecutorConfig(type.first, config, type.second);

                    for (int curHead = 1; curHead <= numAttentionHeads; curHead++) {
                        float timeSum = 0.0f;
                        exe->gc(Executor::FULL);
                        for (int k = 0; k < repeatTime; k++) {
                            key.reset(NN::Linear(hiddenSize, headDim * curHead, true));
                            value.reset(NN::Linear(hiddenSize, headDim * curHead, true));
                            query.reset(NN::Linear(hiddenSize, headDim * curHead, true));
                            dropout.reset(NN::Dropout(0));

                            auto x = _Const(1.0f, {batchSize, seqLen, hiddenSize}, NCHW);

                            auto sliceBegin  = _Input({3}, NCHW);
                            auto sliceSize    = _Input({3}, NCHW);
                            const int beginData[] = {0, 0, 0};
                            memcpy(sliceBegin->writeMap<int>(), beginData, 4 * sizeof(int));
                            const int sizeData[] = {2, seqLen, freqsComplexShape[2]};
                            memcpy(sliceSize->writeMap<int>(), sizeData, 4 * sizeof(int));
                            auto freqs = _Slice(freqsComplex, sliceBegin, sliceSize);

                            auto mixedQueryLayer = query->forward(x);
                            auto mixedKeyLayer = key->forward(x);
                            auto mixedValueLayer = value->forward(x);

                            auto shape = mixedKeyLayer->getInfo()->dim;
                            auto queryLayer = _Reshape(mixedQueryLayer,
                                                       {shape[0], shape[1], curHead, headDim});
                            auto keyLayer = _Reshape(mixedKeyLayer, {shape[0], shape[1], curHead, headDim});
                            auto valueLayer = _Reshape(mixedValueLayer,
                                                       {shape[0], shape[1], curHead, headDim});

                            queryLayer = applyRotaryEmbedding(queryLayer, freqs);
                            keyLayer = applyRotaryEmbedding(keyLayer, freqs);

                            int nRepeat = numAttentionHeads / numKVHeads;
                            keyLayer = repeatKV(keyLayer, nRepeat);
                            valueLayer = repeatKV(valueLayer, nRepeat);

                            queryLayer = _Transpose(queryLayer, {0, 2, 1, 3});
                            keyLayer = _Transpose(keyLayer, {0, 2, 1, 3});
                            valueLayer = _Transpose(valueLayer, {0, 2, 1, 3});

                            // (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
                            auto attentionScores = _MatMul(queryLayer, keyLayer, false, true);
                            attentionScores = _Divide(attentionScores, _Const((float) sqrt(headDim), {}, NCHW));
                            auto attentionProbs = _Softmax(attentionScores, -1);

                            auto contextLayer = _MatMul(attentionProbs, valueLayer);

                            // AUTOTIME;
                            MNN::Timer _100Time;
                            auto ptr = contextLayer->readMap<float>();
                            // MNN_PRINT("Ptr: %f\n", ptr[0]);
                            auto curTime = (float)_100Time.durationInUs() / 1000.0f;
                            timeSum += curTime;
                            LOGI("LLaMAProfileProcessors():Type %d, BatchSize: %d, SeqLen: %d, HiddenSize: %d, AttentionHeads: %d, Time: %f\n",
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

            VARP sampleTopP(VARP probs, float topP) {
                auto probsShape = probs->getInfo()->dim;
                int vocabSize = probsShape[1];

                int selectedKNum = (int)(vocabSize * topP);
                auto KNumVar = _Input({1}, NCHW, halide_type_of<int>());
                auto KNumPtr = KNumVar->writeMap<int>();
                KNumPtr[0] = selectedKNum;

                auto probsSort = _TopKV2(probs, KNumVar);
                // redistribute the prob
                probsSort[0] = _Divide(probsSort[0], _ReduceSum(probsSort[0], {1}, true));

                // generate a random index
                std::random_device rd;
                std::mt19937 rng(rd());
                std::uniform_int_distribution<> dis(0, selectedKNum - 1);
                auto randomIndex = _Input({1}, NCHW, halide_type_of<int>());
                auto randomIndexPtr = randomIndex->writeMap<int>();
                randomIndexPtr[0] = dis(rng);

                auto nextToken = _GatherV2(probsSort[1], randomIndex, _Scalar<int>(1));
                return nextToken;
            }

            LLaMA::LLaMA(LLaMAArgs args) {
                this->args = args;

                tokenizer.reset(new SentencePiece());
                tokenizer->load(args.tokenizerPath);

                args.vocabSize = tokenizer->get_vocab_size();
                transformer.reset(new LLaMATransformer(args));
                // TODO: Load pretrained weights here
            }

            void LLaMA::getResponse(std::vector<std::string>& prompts, float temperature, float topP, int maxGenLen) {
                std::vector<std::vector<int>> encodeResults;
                for (auto prompt : prompts) {
                    auto encodeResult = tokenizer->encode(prompt);
                    encodeResult.insert(encodeResult.begin(), 29871);
                    tokenizer->add_bos(encodeResult);
                    tokenizer->add_eos(encodeResult);
                    encodeResults.push_back(encodeResult);
                }

                int batchSize = encodeResults.size();
                MNN_ASSERT(batchSize <= args.maxBatchSize);

                int maxPromptLen = 0;
                for (auto encodeResult : encodeResults) {
                    maxPromptLen = std::max(maxPromptLen, (int)encodeResult.size());
                }
                MNN_ASSERT(maxPromptLen <= args.maxSeqLen);

                int totalLen = std::min(args.maxSeqLen - 1 + maxPromptLen, args.maxSeqLen);
                int padId = -1; // no <pad> in MNN's vocab file
                std::vector<int> padIdVec(batchSize * totalLen, padId);
                VARP tokens = _Const(padIdVec.data(), {batchSize, totalLen}, NCHW, halide_type_of<int>());

                // put encodeResults into tokens
                auto tokensPtr = tokens->writeMap<int>();
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < encodeResults[i].size(); j++) {
                        tokensPtr[i * totalLen + j] = encodeResults[i][j];
                    }
                }

                // we put one token each time
                auto sliceBegin  = _Input({2}, NCHW);
                auto sliceSize    = _Input({2}, NCHW);

                auto nextBegin = _Input({2}, NCHW);
                auto nextEnd = _Input({2}, NCHW);
                auto nextStride = _Input({2}, NCHW);
                auto nextSize = _Input({2}, NCHW);

                auto promptTokensMask = _NotEqual(tokens, _Scalar<int>(padId));
                int eosCnt = 0;

                for (int curPos = 1; curPos < totalLen; curPos++) {
                    const int beginData[] = {0, curPos - 1};
                    memcpy(sliceBegin->writeMap<int>(), beginData, 2 * sizeof(int));
                    const int sizeData[] = {batchSize, 1};
                    memcpy(sliceSize->writeMap<int>(), sizeData, 2 * sizeof(int));

                    auto curToken = _Slice(tokens, sliceBegin, sliceSize);
                    auto logits = transformer->onForward({curToken, _Scalar<int>(curPos)})[0];

                    auto logitsPtr = logits->readMap<float>();
                    MNN_PRINT("logits output %f\n", logitsPtr[0]);

                    logits = _Squeeze(logits, {1}); // [bts, 1, embedding] => [bts, embedding]
                    if (temperature > 0) {
                        auto probs = _Softmax(_Divide(logits, _Scalar<float>(temperature)));
                        logits = sampleTopP(probs, topP);
                    } else {
                        logits = _ArgMax(logits, 1);
                    }

                    const int nextBeginData[] = {0, curPos};
                    memcpy(nextBegin->writeMap<int>(), nextBeginData, 2 * sizeof(int));

                    const int nextEndData[] = {batchSize, curPos + 1};
                    memcpy(nextEnd->writeMap<int>(), nextEndData, 2 * sizeof(int));

                    const int nextStrideData[] = {1, 1};
                    memcpy(nextStride->writeMap<int>(), nextStrideData, 2 * sizeof(int));

                    const int nextSizeData[] = {batchSize, 1};
                    memcpy(nextSize->writeMap<int>(), nextSizeData, 2 * sizeof(int));

                    auto nextCurToken = _Slice(tokens, nextBegin, nextSize);
                    auto nextMask = _Slice(promptTokensMask, nextBegin, nextSize);
                    auto nextMaskPtr = nextMask->readMap<int>();
                    auto nextToken = _Select(_Equal(nextMask, _Scalar<int>(1)),
                                             nextCurToken, logits);
                    auto nextTokenPtr = nextToken->readMap<int>();

                    auto info = nextToken->getInfo();
                    tokens = _StridedSliceWrite(tokens, nextBegin, nextEnd, nextStride, nextToken, 0, 0, 0, 0, 0);
                    // check whether all the tokens are <eos>
                    for (int i = 0; i < info->size; i++) {
                        if (!nextMaskPtr[i] && nextTokenPtr[i] == tokenizer->get_eos_id()) {
                            eosCnt++;
                        }
                        if (eosCnt >= batchSize) {
                            break;
                        }
                    }
                }

                // decode tokens
                auto outputTokensPtr = tokens->readMap<int>();
                std::string outputTexts = "";
                for (int i = 0; i < tokens->getInfo()->size; i++) {
                    std::cout << outputTokensPtr[i] << " ";
                    outputTexts += tokenizer->decode(outputTokensPtr[i]);
                }
                std::cout << outputTexts << std::endl;
            }

            void LLaMA::setLoRAParamsTrainable() {
                MNN_ASSERT(args.loraArgs.r > 0);
                for (int i = 0; i < args.nLayers; i++) {
                    if (args.loraArgs.enableLoRA[0]) {
                        std::shared_ptr<LLaMALoRALayer> curQuery = std::dynamic_pointer_cast<LLaMALoRALayer>(transformer->decoders[i]->selfAttention->query);
                        curQuery->weight.fix(VARP::CONSTANT);
                        curQuery->loraA.fix(VARP::TRAINABLE);
                        curQuery->loraB.fix(VARP::TRAINABLE);
                        curQuery->addParameter(curQuery->loraA);
                        curQuery->addParameter(curQuery->loraB);
                    }
                    if (args.loraArgs.enableLoRA[1]) {
                        std::shared_ptr<LLaMALoRALayer> curKey = std::dynamic_pointer_cast<LLaMALoRALayer>(transformer->decoders[i]->selfAttention->key);
                        curKey->weight.fix(VARP::CONSTANT);
                        curKey->loraA.fix(VARP::TRAINABLE);
                        curKey->loraB.fix(VARP::TRAINABLE);
                        curKey->addParameter(curKey->loraA);
                        curKey->addParameter(curKey->loraB);
                    }
                    if (args.loraArgs.enableLoRA[2]) {
                        std::shared_ptr<LLaMALoRALayer> curValue = std::dynamic_pointer_cast<LLaMALoRALayer>(transformer->decoders[i]->selfAttention->value);
                        curValue->weight.fix(VARP::CONSTANT);
                        curValue->loraA.fix(VARP::TRAINABLE);
                        curValue->loraB.fix(VARP::TRAINABLE);
                        curValue->addParameter(curValue->loraA);
                        curValue->addParameter(curValue->loraB);
                    }
                }
            }

            LLaMATransformer::LLaMATransformer(LLaMAArgs& args) {
                if (vocabSize == -1) {
                    MNN_ERROR("vocabSize must be specified");
                    return ;
                }
                this->vocabSize = args.vocabSize;
                this->nLayers = args.nLayers;
                this->maxSeqLen = args.maxSeqLen;
                embedding.reset(new MEmbedding(args.vocabSize, args.hiddenSize));

                for (int i = 0; i < nLayers; i++) {
                    decoders.emplace_back(new LLaMALayer(args));
                }

                norm.reset(new MRMSNorm(args.hiddenSize, args.normEps));
                output.reset(NN::Linear(args.hiddenSize, args.vocabSize, false));

                // 提前计算旋转编码的频率
                freqsComplex = precomputeThetaPosFrequencies(args.hiddenSize / args.nHeads, args.maxSeqLen * 2);
            }

            std::vector<Express::VARP> LLaMATransformer::onForward(const std::vector<Express::VARP> &inputs) {
                // inputs[0]: tokens, inputs[1]: startPos
                // If inputs.size() == 1, it means a training process
                VARP tokens = inputs[0];
                auto inputShape = tokens->getInfo()->dim;
                int batchSize = inputShape[0], seqLen = inputShape[1];
                if (seqLen > this->maxSeqLen) {
                    MNN_PRINT("Seqlen is too long, maxSeqLen = %d, seqLen = %d\n", this->maxSeqLen, seqLen);
                    MNN_ASSERT(false);
                }

//                if (seqLen > 1) {
//                    MNN_ERROR("LLaMA only supports seqLen = 1");
//                    return {};
//                }

                // (B,seq_len) => (B, Seq_len, Dim)
                // dim = dimension = 4096
                auto h = embedding->forward(tokens);

                if (inputs.size() > 1) {
                    // generate task
                    // Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
                    // 提前计算position
                    int startPos = *(inputs[1]->readMap<int>());
                    auto freqsComplexShape = freqsComplex->getInfo()->dim;
                    auto sliceBegin  = _Input({3}, NCHW);
                    auto sliceSize    = _Input({3}, NCHW);
                    const int beginData[] = {0, startPos, 0};
                    memcpy(sliceBegin->writeMap<int>(), beginData, 4 * sizeof(int));
                    const int sizeData[] = {2, seqLen, freqsComplexShape[2]};
                    memcpy(sliceSize->writeMap<int>(), sizeData, 4 * sizeof(int));
                    // freqsComplex: [2, seqLen, args.hiddenSize / args.nHeads * 2]
                    auto freqs = _Slice(freqsComplex, sliceBegin, sliceSize);

                    for (int i = 0; i < nLayers; i++) {
                        h = decoders[i]->onForward({h, freqs, inputs[1]})[0];
                    }
                } else {
                    // training process
                    // get freqs with length = seqLen
                    auto freqsComplexShape = freqsComplex->getInfo()->dim;
                    auto sliceBegin  = _Input({3}, NCHW);
                    auto sliceSize    = _Input({3}, NCHW);
                    const int beginData[] = {0, 0, 0};
                    memcpy(sliceBegin->writeMap<int>(), beginData, 4 * sizeof(int));
                    const int sizeData[] = {2, seqLen, freqsComplexShape[2]};
                    memcpy(sliceSize->writeMap<int>(), sizeData, 4 * sizeof(int));
                    // freqsComplex: [2, seqLen, args.hiddenSize / args.nHeads * 2]
                    auto freqs = _Slice(freqsComplex, sliceBegin, sliceSize);

                    for (int i = 0; i < nLayers; i++) {
                        h = decoders[i]->onForward({h, freqs})[0];
                    }
                }

                h = norm->forward(h);
                h = output->forward(h);
                return {h};
            }

            void LLaMATransformer::loadParam(std::string &weightsBasePath, bool isTrainable) {
                return ;
            }

            std::vector<VARP> LLaMATransformer::getLoss(std::vector<VARP> &logits, std::vector<VARP> &labels) {
                // TODO: implement getLoss
                auto logit = logits[0];
                auto targets = labels[0];
                auto logitsInfo = logit->getInfo();
                auto logitsBegin = _Input({3}, NCHW);
                auto logitsEnd = _Input({3}, NCHW);
                auto logitsStrides = _Input({3}, NCHW);
                const int logitsBeginData[] = {0, 0, 0};
                memcpy(logitsBegin->writeMap<int>(), logitsBeginData, 3 * sizeof(int));
                const int logitsEndData[] = {logitsInfo->dim[0], logitsInfo->dim[1] - 1, logitsInfo->dim[2]};
                memcpy(logitsEnd->writeMap<int>(), logitsEndData, 3 * sizeof(int));
                const int logitsStridesData[] = {1, 1, 1};
                memcpy(logitsStrides->writeMap<int>(), logitsStridesData, 3 * sizeof(int));

                auto logitsSlice = _StridedSlice(logit, logitsBegin, logitsEnd, logitsStrides, 0, 0, 0, 0, 0);
                auto activeLogits = _Reshape(logitsSlice, {logitsInfo->dim[0] * (logitsInfo->dim[1] - 1), logitsInfo->dim[2]});

                // stridedSlice version
                auto targetsInfo = logit->getInfo();
                auto targetsBegin = _Input({2}, NCHW);
                auto targetsEnd = _Input({2}, NCHW);
                auto targetsStrides = _Input({2}, NCHW);
                const int targetsBeginData[] = {0, 1};
                memcpy(targetsBegin->writeMap<int>(), targetsBeginData, 2 * sizeof(int));
                const int targetsEndData[] = {targetsInfo->dim[0], targetsInfo->dim[1]};
                memcpy(targetsEnd->writeMap<int>(), targetsEndData, 2 * sizeof(int));
                const int targetsStridesData[] = {1, 1};
                memcpy(targetsStrides->writeMap<int>(), targetsStridesData, 2 * sizeof(int));

                auto targetsSlice = _StridedSlice(targets, targetsBegin, targetsEnd, targetsStrides, 0, 0, 0, 0, 0);
                auto activeLabels = _Reshape(targetsSlice, {-1});
                auto newActiveLabels = _OneHot(_Cast<int32_t>(activeLabels), _Scalar<int>(logitsInfo->dim[2]), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

                // calculate loss
                int ignoreIndex = -1;
                auto ignoredMask = _Cast<float>(_Unsqueeze(_NotEqual(activeLabels, _Scalar<int>(ignoreIndex)), {1}));
                newActiveLabels = newActiveLabels * ignoredMask;

                // Softmax is used here to normalize the logits, but not provided in lit-llama
                // Since on and off value should be provided in OneHot, if _Softmax is not used, the loss becomes NaN
                auto loss = _CrossEntropy(_Softmax(activeLogits), newActiveLabels);
                return {loss};
            }

            SubLLaMA::SubLLaMA(int start, int end, std::map<std::string, double> &modelArgs) {
                LLaMAArgs llamaArgs;

                this->numHiddenLayers = (int) modelArgs["num_hidden_layers"];
                this->totalLayers = (int) modelArgs["total_layer"];
                this->vocabSize = (int) modelArgs["vocab_size"];
                this->maxSeqLen = (int) modelArgs["max_seq_len"];

                llamaArgs.nLayers = (int)modelArgs["num_hidden_layers"];
                llamaArgs.vocabSize = (int) modelArgs["vocab_size"];
                llamaArgs.hiddenSize = (int) modelArgs["hidden_size"];
                llamaArgs.normEps = (float) modelArgs["norm_eps"];
                llamaArgs.nHeads = (int) modelArgs["num_attention_heads"];
                llamaArgs.nKVHeads = (int) modelArgs["num_kv_heads"];
                llamaArgs.maxSeqLen = (int) modelArgs["max_seq_len"];
                llamaArgs.maxBatchSize = (int) modelArgs["max_batch_size"];
                llamaArgs.multipleOf = (int) modelArgs["multiple_of"];
                llamaArgs.ffnDimMultiplier = (int) modelArgs["ffn_dim_multiplier"];
                llamaArgs.loraArgs.r = (int) modelArgs["lora_r"];
                llamaArgs.loraArgs.enableLoRA = {(bool)modelArgs["enable_lora_q"], (bool)modelArgs["enable_lora_k"],
                                                 (bool)modelArgs["enable_lora_v"]};
                llamaArgs.loraArgs.alpha = (float) modelArgs["lora_alpha"];
                llamaArgs.loraArgs.dropout = (float) modelArgs["lora_dropout"];

                if(end == -1)
                    end = this->totalLayers-1;

                if (start == 0) {
                    this->embedding = std::shared_ptr<MEmbedding> (new MEmbedding(llamaArgs.vocabSize, llamaArgs.hiddenSize));
                    registerModel({this->embedding});
                    layers.emplace_back(this->embedding);
                }

                int curLayer = 1;
                decoders = std::vector<std::shared_ptr<Module>>();
                for (int i = 0; i < numHiddenLayers; i++) {
                    if (curLayer >= start && curLayer <= end) {
                        auto decoder = std::make_shared<LLaMALayer>(llamaArgs);
                        decoders.push_back(decoder);
                        layers.emplace_back(decoder);
                    }
                    curLayer++;
                }

                if (!decoders.empty()) {
                    registerModel(decoders);
                }

                int outputLayerIndex = llamaArgs.nLayers + 1;
                if (outputLayerIndex >= start && outputLayerIndex <= end) {
                    this->output.reset(NN::Linear(llamaArgs.hiddenSize, llamaArgs.vocabSize, false));
                    registerModel({this->output});
                    layers.emplace_back(this->output);
                }

                if (end == this->totalLayers - 1) {
                    this->norm = std::make_shared<MRMSNorm>(llamaArgs.hiddenSize, llamaArgs.normEps);
                    registerModel({this->norm});
                    layers.emplace_back(this->norm);
                }
//                if(!decoders.empty()){
//                    auto params = decoders[0]->parameters();

                this->freqsComplex = precomputeThetaPosFrequencies(llamaArgs.hiddenSize / llamaArgs.nHeads,
                                                             llamaArgs.maxSeqLen * 2);

            }

            std::vector<Express::VARP>
            SubLLaMA::onForward(const std::vector<Express::VARP> &inputs) {
                // only support for train
                MNN_ASSERT(inputs.size() == 1);
                auto h = inputs[0];
                auto inputShape = h->getInfo()->dim;
                int seqLen = inputShape[1];
                if (seqLen > this->maxSeqLen) {
                    MNN_PRINT("Seqlen is too long, maxSeqLen = %d, seqLen = %d\n", this->maxSeqLen, seqLen);
                }

                if(this->embedding != nullptr){
                    h = this->embedding->forward(h);
                }

                if(!this->decoders.empty()){
                    auto freqsComplexShape = freqsComplex->getInfo()->dim;
                    auto sliceBegin  = _Input({3}, NCHW);
                    auto sliceSize    = _Input({3}, NCHW);
                    const int beginData[] = {0, 0, 0};
                    memcpy(sliceBegin->writeMap<int>(), beginData, 4 * sizeof(int));
                    const int sizeData[] = {2, seqLen, freqsComplexShape[2]};
                    memcpy(sliceSize->writeMap<int>(), sizeData, 4 * sizeof(int));
                    auto freqs = _Slice(freqsComplex, sliceBegin, sliceSize);

                    for (int i = 0; i < this->decoders.size(); i++) {
                        h = this->decoders[i]->onForward({h, freqs})[0];
                    }
                }

                if (this->norm != nullptr) {
                    h = this->norm->forward(h);
                }

                if (this->output != nullptr) {
                    h = this->output->forward(h);
                }

//                auto ptr = h->readMap<float>();

//                auto ptr2 = h->readMap<float>();

                return {h};
            }

            void SubLLaMA::loadParamByLayer(int layer, std::string &weightsBasePath, int startLayer, bool isTrainable) {
                if(layer > this->totalLayers - 1 || layer < 0){
                    MNN_ERROR("layer index out of range");
                    return;
                }
                startLayer = (startLayer==0)? 1: startLayer;
                if(layer == 0){
                    auto embeddingParams = this->embedding->parameters();
                    for (auto& param : embeddingParams) {
                        param.fix(VARP::CONSTANT);
                    }
//                    std::string llamaEmbeddiingParamsPath = weightsBasePath + "embedding.mnn";
//                    auto llamaEmbeddingParams = Variable::load(llamaEmbeddiingParamsPath.c_str());
//                    auto embeddingPtr = llamaEmbeddingParams[0]->readMap<float>();
//                    this->embedding->loadParameters(llamaEmbeddingParams);
//                    auto loadedParams = this->embedding->parameters();
//                    auto loadedParamsPtr = loadedParams[0]->readMap<float>();

                }else if(layer - startLayer <= this->numHiddenLayers){
                    int decoderIndex = layer - startLayer;
                    std::string llamaDecoderParamsPath = weightsBasePath + "layer" +
                                                         std::to_string(decoderIndex) + ".mnn";
                    auto llamaDecoderParamsVar = Variable::load(llamaDecoderParamsPath.c_str());
                    this->decoders[decoderIndex]->loadParameters(llamaDecoderParamsVar);
                }else if(layer == this->totalLayers-2){
                    std::string llamaOutputParamsPath = weightsBasePath + "llama_output.mnn";
                    auto llamaOutputParams = Variable::load(llamaOutputParamsPath.c_str());
                    this->output->loadParameters(llamaOutputParams);
                } else if(layer == this->totalLayers-1){
                    std::string llamaNormParamsPath = weightsBasePath + "llama_norm.mnn";
                    auto llamaNormParams = Variable::load(llamaNormParamsPath.c_str());
                    this->norm->loadParameters(llamaNormParams);
                }

            }
//                if(layer > this->totalLayers - 1 || layer < 0){
//                    MNN_ERROR("layer index out of range");
//                    return;
//                }
//
//                startLayer = (startLayer==0)? 1: startLayer;
//
//                if(layer == 0){
//                    std::string llamaEmbeddiingParamsPath = weightsBasePath + "llama_embeddings.mnn";
//                    auto llamaEmbeddingParams = Variable::load(llamaEmbeddiingParamsPath.c_str());
//                    auto embeddingPtr = llamaEmbeddingParams[0]->readMap<float>();
//                    this->embedding->loadParameters(llamaEmbeddingParams);
//                    auto loadedParams = this->embedding->parameters();
//                    auto loadedParamsPtr = loadedParams[0]->readMap<float>();
//                    int i = 0;
//                }
//                else if(layer - startLayer <= this->numHiddenLayers){
//                    int decoderIndex = layer - startLayer;
//                    std::string llamaDecoderParamsPath = weightsBasePath + "llama_layer_" +
//                                                         std::to_string(decoderIndex) + ".mnn";
//                    auto llamaDecoderParamsVar = Variable::load(llamaDecoderParamsPath.c_str());
//                    this->decoders[decoderIndex]->loadParameters(llamaDecoderParamsVar);
//                }
//                else if(layer == this->totalLayers-2){
//                    std::string llamaOutputParamsPath = weightsBasePath + "llama_output.mnn";
//                    auto llamaOutputParams = Variable::load(llamaOutputParamsPath.c_str());
//                    this->output->loadParameters(llamaOutputParams);
//                } else if(layer == this->totalLayers-1){
//                    std::string llamaNormParamsPath = weightsBasePath + "llama_norm.mnn";
//                    auto llamaNormParams = Variable::load(llamaNormParamsPath.c_str());
//                    this->norm->loadParameters(llamaNormParams);
//                }
//            }
//
            std::vector<VARP> SubLLaMA::getParamsByLayer(int layer, bool isTrainable) {

                if (layer > this->totalLayers - 1 || layer < 0) {
                    MNN_ERROR("layer index out of range");
                    return {};
                }
//
                return this->layers[layer]->parameters();
            }

            std::vector<VARP>
            SubLLaMA::getLoss(std::vector<VARP> &logits, std::vector<VARP> &labels) {
                auto logit = logits[0];
                auto targets = labels[0];
//
//                auto logitsInfo = logit->getInfo();
//                auto logitsBegin = _Input({3}, NCHW);
//                auto logitsSizes = _Input({3}, NCHW);
//                const int logitsBeginData[] = {0, 0, 0};
//                memcpy(logitsBegin->writeMap<int>(), logitsBeginData, 3 * sizeof(int));
//                const int logitsSizesData[] = {logitsInfo->dim[0], logitsInfo->dim[1] - 1, logitsInfo->dim[2]};
//                memcpy(logitsSizes->writeMap<int>(), logitsSizesData, 3 * sizeof(int));
//
//                auto logitsSlice = _Slice(logit, logitsBegin, logitsSizes);
//                auto activeLogits = _Reshape(logitsSlice, {logitsInfo->dim[0] * (logitsInfo->dim[1] - 1), logitsInfo->dim[2]});
//
//                auto targetsInfo = logit->getInfo();
//                auto targetsBegin = _Input({2}, NCHW);
//                auto targetsSizes = _Input({2}, NCHW);
//                const int targetsBeginData[] = {0, 1};
//                memcpy(targetsBegin->writeMap<int>(), targetsBeginData, 2 * sizeof(int));
//                const int targetsSizesData[] = {targetsInfo->dim[0], targetsInfo->dim[1] - 1};
//                memcpy(targetsSizes->writeMap<int>(), targetsSizesData, 2 * sizeof(int));
//
//                auto targetsSlice = _Slice(targets, targetsBegin, targetsSizes);
//                auto activeLabels = _Reshape(targetsSlice, {-1});
//                auto newActiveLabels = _OneHot(_Cast<int32_t>(activeLabels), _Scalar<int>(logitsInfo->dim[2]), _Scalar<float>(1.0f), _Scalar<float>(0.0f));
//
//                // calculate loss
//                int ignoreIndex = -1;
//                auto ignoredMask = _Cast<float>(_Unsqueeze(_NotEqual(activeLabels, _Scalar<int>(ignoreIndex)), {1}));
//                newActiveLabels = newActiveLabels * ignoredMask;
//
//                auto loss = _CrossEntropy(_Softmax(activeLogits), newActiveLabels);
//                auto ptr = loss->readMap<float>();
//                return {loss};
// key idea: shift the targets such that output n predicts token n+1
//    auto logitsInfo = logits->getInfo();
//    auto logitsBegin = _Input({3}, NCHW);
//    auto logitsSizes = _Input({3}, NCHW);
//    const int logitsBeginData[] = {0, 0, 0};
//    memcpy(logitsBegin->writeMap<int>(), logitsBeginData, 3 * sizeof(int));
//    const int logitsSizesData[] = {logitsInfo->dim[0], logitsInfo->dim[1] - 1, logitsInfo->dim[2]};
//    memcpy(logitsSizes->writeMap<int>(), logitsSizesData, 3 * sizeof(int));
//
//    auto logitsSlice = _Slice(logits, logitsBegin, logitsSizes);

                // stridedSlice version
                auto logitsInfo = logit->getInfo();
                auto logitsBegin = _Input({3}, NCHW);
                auto logitsEnd = _Input({3}, NCHW);
                auto logitsStrides = _Input({3}, NCHW);
                const int logitsBeginData[] = {0, 0, 0};
                memcpy(logitsBegin->writeMap<int>(), logitsBeginData, 3 * sizeof(int));
                const int logitsEndData[] = {logitsInfo->dim[0], logitsInfo->dim[1] - 1, logitsInfo->dim[2]};
                memcpy(logitsEnd->writeMap<int>(), logitsEndData, 3 * sizeof(int));
                const int logitsStridesData[] = {1, 1, 1};
                memcpy(logitsStrides->writeMap<int>(), logitsStridesData, 3 * sizeof(int));

                auto logitsSlice = _StridedSlice(logit, logitsBegin, logitsEnd, logitsStrides, 0, 0, 0, 0, 0);
                auto activeLogits = _Reshape(logitsSlice, {logitsInfo->dim[0] * (logitsInfo->dim[1] - 1), logitsInfo->dim[2]});

                // stridedSlice version
                auto targetsInfo = logit->getInfo();
                auto targetsBegin = _Input({2}, NCHW);
                auto targetsEnd = _Input({2}, NCHW);
                auto targetsStrides = _Input({2}, NCHW);
                const int targetsBeginData[] = {0, 1};
                memcpy(targetsBegin->writeMap<int>(), targetsBeginData, 2 * sizeof(int));
                const int targetsEndData[] = {targetsInfo->dim[0], targetsInfo->dim[1]};
                memcpy(targetsEnd->writeMap<int>(), targetsEndData, 2 * sizeof(int));
                const int targetsStridesData[] = {1, 1};
                memcpy(targetsStrides->writeMap<int>(), targetsStridesData, 2 * sizeof(int));

                auto targetsSlice = _StridedSlice(targets, targetsBegin, targetsEnd, targetsStrides, 0, 0, 0, 0, 0);
                auto activeLabels = _Reshape(targetsSlice, {-1});
                auto newActiveLabels = _OneHot(_Cast<int32_t>(activeLabels), _Scalar<int>(logitsInfo->dim[2]), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

                // calculate loss
                int ignoreIndex = -1;
                auto ignoredMask = _Cast<float>(_Unsqueeze(_NotEqual(activeLabels, _Scalar<int>(ignoreIndex)), {1}));
                newActiveLabels = newActiveLabels * ignoredMask;

                // Softmax is used here to normalize the logits, but not provided in lit-llama
                // Since on and off value should be provided in OneHot, if _Softmax is not used, the loss becomes NaN
                auto loss = _CrossEntropy(_Softmax(activeLogits), newActiveLabels);
                return {loss};
            }


        }
    }
}