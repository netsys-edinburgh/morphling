//
// Created by Yuhao Chen on 2023/6/12.
//

#include "MEmbedding.h"
#include "Initializer.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;

            MEmbedding::MEmbedding(int numEmbedding, int embeddingDim, int paddingIdx, float maxNorm, float normType,
                                   bool scaleGradByFreq, bool sparse, bool freeze, VARP weight): numEmbedding(numEmbedding),
                                                                                      embeddingDim(embeddingDim),
                                                                                      maxNorm(maxNorm),
                                                                                      normType(normType),
                                                                                      scaleGradByFreq(scaleGradByFreq),
                                                                                      sparse(sparse) {

                if (paddingIdx != -1) {
                    if (paddingIdx > 0) {
                        if (paddingIdx < numEmbedding) {
                            MNN_PRINT("Padding_idx must be within num_embeddings");
                            return ;
                        }
                    } else if (paddingIdx < 0) {
                        if (paddingIdx >= -numEmbedding) {
                            MNN_PRINT("Padding_idx must be within num_embeddings");
                            return ;
                        }
                        paddingIdx = numEmbedding + paddingIdx;
                    }
                }
                this->paddingIdx = paddingIdx;

                if (weight == nullptr) {
                    std::shared_ptr<Initializer> initializer;
                    initializer.reset(Initializer::gauss());
                    this->weight = initializer->createConstVar({numEmbedding, embeddingDim}, NCHW);

                    if (!freeze) {
//                        this->weight.fix(VARP::TRAINABLE);
                        this->weight.fix(VARP::CONSTANT);
                    }
                } else {
                    this->weight = weight;
                }
                addParameter(this->weight);
            }

            std::vector<Express::VARP> MEmbedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];

                auto y = _GatherV2(weight, x);
                return {y};
            }

        }
    }
}