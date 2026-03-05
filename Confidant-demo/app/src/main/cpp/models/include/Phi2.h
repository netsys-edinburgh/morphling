//
// Created by yue on 2024/3/12.
//

#ifndef CONFIDANT_PHI2_H
#define CONFIDANT_PHI2_H
#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "Loss.hpp"
#include "MEmbedding.h"
#include "Phi2Layer.h"
#include "ModelArgs.h"
#include "Tokenizer.h"
#include "SubModel.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            std::unordered_map<MNNForwardType, std::vector<float>> Phi2ProfileProcessors(int batchSize, std::map<std::string, double> &modelArgs);
            VARP sampleTopP(VARP probs, float topP);

            class PhiTransformer : public Confidant::SingleModel{
            public:
                PhiTransformer(PhiArgs& args);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                std::shared_ptr<MEmbedding> embed_tokens;
                std::shared_ptr<Express::Module> embed_dropout;
                std::vector<std::shared_ptr<PhiLayer>> layers;
                std::shared_ptr<Module> final_layernorm;
                std::shared_ptr<Module> output;

                virtual void loadParam(std::string& weightsBasePath, bool isTrainable = false) override;
                virtual std::vector<VARP> getLoss(std::vector<VARP>& logits, std::vector<VARP>& labels) override;

            private:
                int vocabSize, nLayers;
                int maxSeqLen;
                VARP freqsComplex;
            };

            class Phi2 {
            public:
                explicit Phi2(PhiArgs args);
                void getResponse(std::vector<std::string>& prompts, float temperature = 0.6f, float topP = 0.9f, int maxGenLen = -1);
                std::shared_ptr<PhiTransformer> transformer;
                std::shared_ptr<SentencePiece> tokenizer;
            private:
                PhiArgs args;
            };

            class SubPhi2: public Confidant::SubModel {
            public:
                SubPhi2(int start, int end, std::map<std::string,double>& modelArgs);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

                virtual void loadParamByLayer(int layer, std::string& weightsBasePath, int startLayer = 1, bool isTrainable = false) override;

                virtual std::vector<VARP> getParamsByLayer(int layer, bool isTrainable) override;

                virtual std::vector<VARP> getLoss(std::vector<VARP>& logit, std::vector<VARP>& labels) override;

                std::shared_ptr<MEmbedding> embed_tokens;
                std::shared_ptr<Express::Module> embed_dropout;
                std::vector<std::shared_ptr<Module>> decoders;
                std::shared_ptr<MEmbedding> embedding;
                std::shared_ptr<Module> final_layernorm;
                std::shared_ptr<Module> output;

                std::vector<std::shared_ptr<Module>> layers;
            private:
                int numHiddenLayers;
                int totalLayers;
                int vocabSize;
                int maxSeqLen;
                VARP freqsComplex;
            };

        }
    }
}


#endif //CONFIDANT_PHI2_H
