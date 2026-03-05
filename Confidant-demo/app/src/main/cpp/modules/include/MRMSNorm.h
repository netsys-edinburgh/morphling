//
// Created by Yuhao Chen on 2023/10/16.
//

#ifndef MNN_MRMSNORM_HPP
#define MNN_MRMSNORM_HPP

#include <MNN/expr/Module.hpp>
#include "NN.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;
            class MNN_PUBLIC MRMSNorm : public Express::Module {
            private:
                VARP weight;
                VARP eps;
            public:
                // bool affine = true, bool trackRunningStats = true？
                MRMSNorm(int dim, float eps = 1e-6);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                VARP _Norm(VARP x);
            };
        }
    }
}

#endif //MNN_MRMSNORM_HPP
