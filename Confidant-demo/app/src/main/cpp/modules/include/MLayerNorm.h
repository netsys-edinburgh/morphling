//
// Created by Yuhao Chen on 2023/5/22.
//

#ifndef MLayerNorm_hpp
#define MLayerNorm_hpp

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "Initializer.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;
            class MNN_PUBLIC MLayerNorm : public Express::Module {
            private:
                VARP eps;
                VARP weight;
                VARP bias;
                std::vector<int> normalShape;
                bool elementwiseAffine;
            public:
                // Only image input is considered here, where the type of normalShape is std::vector<int>
                MLayerNorm(std::vector<int> normalShape, bool elementwiseAffine = true, double eps = 1e-5);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

        } // namespace Model
    } // namespace Train
} // namespace MNN

#endif //MNN_MLAYERNORM_H
