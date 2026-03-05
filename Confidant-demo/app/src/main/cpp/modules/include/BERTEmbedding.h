//
// Created by Yuhao Chen on 2023/6/13.
//

#ifndef MNN_BERTEMBEDDING_H
#define MNN_BERTEMBEDDING_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;
            class MNN_PUBLIC BERTEmbedding : public Express::Module {
            public:
                std::shared_ptr<Express::Module> wordEmbeddings;
                std::shared_ptr<Express::Module> positionEmbeddings;
                std::shared_ptr<Express::Module> tokenTypeEmbeddings;
                std::shared_ptr<Express::Module> layerNorm;
                std::shared_ptr<Express::Module> dropout;
                int hiddenSize;

                BERTEmbedding(int vocabSize, int hiddenSize, int maxPositionEmbeddings, int typeVocabSize, float dropoutProb = 0.1);

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };
        }
    }
}

#endif //MNN_BERTEMBEDDING_H
