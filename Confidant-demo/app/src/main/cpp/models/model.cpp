//
// Created by Yuhao Chen on 2023/1/10.
//

#include "model.h"
#include "BERT.h"
#include "GPT2.h"
#include "LLaMA.h"
#include "Phi2.h"
#include "log.h"
#include "ModelArgs.h"

using namespace std;
using namespace MNN;
using namespace MNN::Train::Model;

namespace Confidant {
    std::shared_ptr<SingleModel> ModelZoo::modelPtr = nullptr;
    std::shared_ptr<SubModel> ModelZoo::subModelPtr = nullptr;

    /*
     * Create a model based on the model name and arguments
     */
    void ModelZoo::createModel(std::string &modelName, std::map<std::string, double> &modelArgs) {
        if (modelName == "BERTForClassification") {
            int vocabSize = (int) modelArgs["vocab_size"];
            int numHiddenLayers = (int) modelArgs["num_hidden_layers"];
            int hiddenSize = (int) modelArgs["hidden_size"];
            int intermediateSize = (int) modelArgs["intermediate_size"];
            int numAttentionHeads = (int) modelArgs["num_attention_heads"];
            float attnDropout = (float) modelArgs["attention_dropout_prob"];
            float hiddenDropout = (float) modelArgs["hidden_dropout_prob"];
            bool forParallel = (bool) modelArgs["for_parallel"];

            modelPtr = std::shared_ptr<BERTForClassification>(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers,
                                                                                        numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,
                                                                                        9, forParallel));
        } else if(modelName == "GPT2") {
            GPT2Args gpt2Args;
            gpt2Args.vocabSize = (int) modelArgs["vocab_size"];
            gpt2Args.maxPositionEmbeddings = (int) modelArgs["maxPositionEmbeddings"];
            gpt2Args.numHiddenLayers = (int) modelArgs["num_hidden_layers"];
            gpt2Args.numAttentionHeads = (int) modelArgs["num_attention_heads"];
            gpt2Args.hiddenSize = (int) modelArgs["hidden_size"];
            gpt2Args.intermediateSize = (int) modelArgs["intermediate_size"];
            gpt2Args.dropoutProb = (float) modelArgs["hidden_dropout_prob"];
            gpt2Args.forParallel = (bool) modelArgs["for_parallel"];

            modelPtr = std::shared_ptr<GPT2QAModel>(new GPT2QAModel(gpt2Args));
        } else if(modelName == "LLaMALora" or modelName == "LLaMA") {
            LLaMAArgs llamaArgs;
            llamaArgs.vocabSize = (int) modelArgs["vocab_size"];
            llamaArgs.nLayers = (int) modelArgs["num_hidden_layers"];
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

            modelPtr = std::shared_ptr<LLaMATransformer>(new LLaMATransformer(llamaArgs));
        } else if(modelName == "Phi2Alpaca") {
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
            phiArgs.maxBatchSize = (int)modelArgs["max_batch_size"];
            modelPtr = std::shared_ptr<PhiTransformer>(new PhiTransformer(phiArgs));
        }
    }

    void ModelZoo::createSubModel(std::string &modelName, std::map<std::string, double> &modelArgs, int start, int end) {
        if (modelName == "BERTForClassification") {
            subModelPtr = std::shared_ptr<SubModel>(new SubBERTForClassification(start, end, modelArgs));
        } else if(modelName == "GPT2"){
            subModelPtr = std::shared_ptr<SubModel>(new SubGPT2(start, end, modelArgs));
        } else if(modelName == "LLaMALora" or modelName == "LLaMA"){
            subModelPtr = std::shared_ptr<SubModel>(new SubLLaMA(start, end, modelArgs));
        } else if(modelName == "Phi2Alpaca" or modelName == "Phi2"){
            subModelPtr = std::shared_ptr<SubModel>(new SubPhi2(start, end, modelArgs));
        }
    }
}