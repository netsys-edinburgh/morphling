//
// Created by gsw on 2023/6/12.
//

#ifndef MNN_GPT2_H
#define MNN_GPT2_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "Loss.hpp"
#include "GPT2Embedding.h"
#include "SubModel.h"
#include "MLayerNorm.h"
#include "ModelArgs.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;
            std::unordered_map<MNNForwardType, std::vector<float>> GPT2ProfileProcessors(int batchSize, std::map<std::string, double> &modelArgs);
            float GPT2ProfileBlock(std::map<std::string, double> &modelArgs, int numBlocks);

            class GPT2Encoder : public Module {
            public:
                GPT2Encoder(int vocabSize, int maxPositionEmbeddings, int numHiddenLayers,
                            int numAttentionHeads, int hiddenSize, int intermediateSize, bool forParallel = false);
                std::shared_ptr<MLayerNorm> layerNorm;
                std::vector<std::shared_ptr<Module>> layers;
                std::shared_ptr<GPT2Embedding> Embeddings;

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

            };
            class GPT2layers : public Module {
            public:

                GPT2layers(int numHiddenLayers,  int numAttentionHeads, int hiddenSize, int intermediateSize, bool forParallel = false);
                std::vector<std::shared_ptr<Module>> layer;

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

            };
            class GPT2 : public Module {
            public:
                GPT2(GPT2Args& args);
                std::shared_ptr<GPT2Encoder> transformer;
                std::shared_ptr<Module> dense;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

            };

            class GPT2QAModel : public Confidant::SingleModel {
            public:
                GPT2QAModel(GPT2Args& args);
                std::shared_ptr<GPT2Encoder> transformer;
                std::shared_ptr<Module> dense;
                int QA_dim = 2;
                int hiddenSize;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                virtual void loadParam(std::string& weightsBasePath, bool isTrainable = false) override;
                virtual std::vector<VARP> getLoss(std::vector<VARP>& logits, std::vector<VARP>& labels) override;

                virtual float getOutputDataSizeByIdx(int idx, int batchSize, int seqLen) override;
                virtual float getModelTimeByIdx(int idx, int batchSize, int seqLen) override;
            };

            class SubGPT2 : public Confidant::SubModel {
            public:
                SubGPT2(int start, int end, std::map<std::string, double>& modelArgs);

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

                virtual void loadParamByLayer(int layer, std::string& weightsBasePath, int startLayer = 1, bool isTrainable = false) override;

                virtual std::vector<VARP> getParamsByLayer(int layer, bool isTrainable) override;

                virtual std::vector<VARP> getLoss(std::vector<VARP>& logit, std::vector<VARP>& labels) override;

                std::shared_ptr<GPT2Embedding> embedding;
                std::vector<std::shared_ptr<Module>> decoders;
                std::shared_ptr<MLayerNorm> norm;
                std::shared_ptr<Module> dense;

                std::vector<std::shared_ptr<Module>> layers;
            private:
                int numHiddenLayers;
                int totalLayers;
            };
        }
    }
}

#endif //MNN_GPT2_H
