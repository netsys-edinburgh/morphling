//
// Created by Yuhao Chen on 2023/6/13.
//

#ifndef MNN_GPT2EMBEDDING_H
#define MNN_GPT2EMBEDDING_H

#include "MNN/expr/Module.hpp"
#include "NN.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;
            class MNN_PUBLIC GPT2Embedding : public Express::Module {
            public:
                std::shared_ptr<Express::Module> wordEmbeddings;
                std::shared_ptr<Express::Module> positionEmbeddings;
                //std::shared_ptr<Express::Module> tokenTypeEmbeddings;
                //std::shared_ptr<Express::Module> layerNorm;
                std::shared_ptr<Express::Module> dropout;
                int hiddenSize;

                GPT2Embedding(int vocabSize, int hiddenSize, int maxPositionEmbeddings, float dropoutProb = 0.0);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };
        }
    }
}

#endif //MNN_GPT2EMBEDDING_H
