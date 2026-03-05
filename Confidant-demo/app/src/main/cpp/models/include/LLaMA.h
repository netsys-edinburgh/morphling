//
// Created by Yuhao Chen on 2023/10/16.
//

#ifndef MNN_LLAMA_H
#define MNN_LLAMA_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "Loss.hpp"
#include "MEmbedding.h"
#include "LLaMALayer.h"
#include "ModelArgs.h"
#include "Tokenizer.h"
#include "SubModel.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            std::unordered_map<MNNForwardType, std::vector<float>> LLaMAProfileProcessors(int batchSize, std::map<std::string, double> &modelArgs);
            VARP sampleTopP(VARP probs, float topP);

            class LLaMATransformer : public Confidant::SingleModel {
            public:
                LLaMATransformer(LLaMAArgs& args);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                std::shared_ptr<MEmbedding> embedding;
                std::vector<std::shared_ptr<LLaMALayer>> decoders;
                std::shared_ptr<Module> output;
                std::shared_ptr<MRMSNorm> norm;

                virtual void loadParam(std::string& weightsBasePath, bool isTrainable = false) override;
                virtual std::vector<VARP> getLoss(std::vector<VARP>& logits, std::vector<VARP>& labels) override;

            // void setLoRAParamsTrainable();
            private:
                int vocabSize, nLayers;
                int maxSeqLen;
                VARP freqsComplex;
            };

            class LLaMA {
            public:
                explicit LLaMA(LLaMAArgs args);
                void getResponse(std::vector<std::string>& prompts, float temperature = 0.6f, float topP = 0.9f, int maxGenLen = -1);
                void setLoRAParamsTrainable();
                std::shared_ptr<LLaMATransformer> transformer;
                std::shared_ptr<SentencePiece> tokenizer;
            private:
                LLaMAArgs args;
            };

            class SubLLaMA: public Confidant::SubModel {
            public:
                SubLLaMA(int start, int end, std::map<std::string, double>& modelArgs);

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

                virtual void loadParamByLayer(int layer, std::string& weightsBasePath, int startLayer = 1, bool isTrainable = false) override;

                virtual std::vector<VARP> getParamsByLayer(int layer, bool isTrainable) override;
                virtual std::vector<VARP> getLoss(std::vector<VARP>& logit, std::vector<VARP>& labels) override;

                std::shared_ptr<LLaMATransformer> transformer;
                std::shared_ptr<MEmbedding> embedding;
                std::vector<std::shared_ptr<Module>> decoders;
                std::shared_ptr<Module> output;
                std::shared_ptr<MRMSNorm> norm;

                std::vector<std::shared_ptr<Module>> layers;

            private:
                int numHiddenLayers;
                int totalLayers;
                int vocabSize;
                int maxSeqLen;
                VARP freqsComplex;
                LLaMAArgs args;
            };
        }
    }
}

#endif //MNN_LLAMA_H
