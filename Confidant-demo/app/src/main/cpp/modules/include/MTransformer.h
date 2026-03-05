//
// Created by Yuhao Chen on 2023/5/17.
//

#ifndef MTransformer_hpp
#define MTransformer_hpp

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "Initializer.hpp"

namespace MNN {
    class MNN_PUBLIC MTransformer : public Express::Module {
    public:
        MTransformer(int dim, int depth, int heads, int dimHead, int mlpDim, float dropout = 0.0);

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::vector<std::pair<std::shared_ptr<Module>, std::shared_ptr<Module>>> layers;
    };
}

#endif //MTransformer_hpp
