//
// Created by Yuhao CHen on 2023/11/24.
//

#include "profiler.h"
#include <MNN/AutoTime.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::Train::Model;

namespace Confidant {
    std::vector<std::shared_ptr<Module>> initEncoderByModelName(std::string& modelName, std::map<std::string, double> &modelArgs, int numEncoders) {
        std::vector<std::shared_ptr<Module> > encoders;
        if (modelName == "BERTForClassification") {
            int vocabSize = (int) modelArgs["vocab_size"];
            int numHiddenLayers = (int) modelArgs["num_hidden_layers"];
            int hiddenSize = (int) modelArgs["hidden_size"];
            int intermediateSize = (int) modelArgs["intermediate_size"];
            int numAttentionHeads = (int) modelArgs["num_attention_heads"];
            float attnDropout = (float) modelArgs["attention_dropout_prob"];
            float hiddenDropout = (float) modelArgs["hidden_dropout_prob"];
            bool forParallel = (bool) modelArgs["for_parallel"];

            for (int i = 0; i < numEncoders; i++) {
                encoders.push_back(std::shared_ptr<BERTLayer>(new BERTLayer(numAttentionHeads, hiddenSize, intermediateSize, hiddenDropout)));
            }
        }

        return encoders;
    }

    std::vector<float> profileEncoders(std::string& modelName, std::map<std::string, double> &modelArgs, int numEncoders) {
        std::vector<float> ccv;

        for (int i = 1; i <= numEncoders; i++) {
            auto encoders = initEncoderByModelName(modelName, modelArgs, i);

            int bts = 8;
            int seqLen = 256;
            int hiddenSize = (int) modelArgs["hidden_size"];

            auto x = _Const(1.0f, {bts, seqLen, hiddenSize}, NCHW);
            auto attentionMask = _Const(1.0f, {bts, 1, 1, seqLen}, NCHW);
            MNN::Timer _100Time;
            for (auto encoder : encoders) {
                x = encoder->onForward({x, attentionMask})[0];

            }
            auto ptr = x->readMap<float>();
            ccv.push_back(_100Time.durationInUs() / 1000.0f);
        }

        return ccv;
    }
}