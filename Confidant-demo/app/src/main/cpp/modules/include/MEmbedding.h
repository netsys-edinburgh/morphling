//
// Created by Yuhao Chen on 2023/6/12.
//

#ifndef MNN_MEMBEDDING_H
#define MNN_MEMBEDDING_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;
            class MNN_PUBLIC MEmbedding : public Express::Module {
            private:
                int numEmbedding;
                int embeddingDim;
                int paddingIdx;
                float maxNorm;
                float normType;
                bool scaleGradByFreq;
                bool sparse;
            public:
                VARP weight;
                MEmbedding(int numEmbedding, int embeddingDim, int paddingIdx = -1, float maxNorm = 0.0,
                           float normType = 2.0, bool scaleGradByFreq = false, bool sparse = false,
                           bool freeze = false, VARP weight = nullptr);

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };
        } // namespace Model
    } // namespace Train
} // namespace MNN

#endif //MNN_MEMBEDDING_H
