//
// Created by 陈宇豪 on 2023/8/9.
//

#ifndef MNN_ADAMW_H
#define MNN_ADAMW_H

#include <set>
#include <string>
#include <vector>
#include "ParameterOptimizer.hpp"
#include "SGD.hpp"

namespace MNN {
namespace Train {
        class MNN_PUBLIC ADAMW : public SGD {
        public:
            ADAMW(std::shared_ptr<Express::Module> module);
            virtual ~ ADAMW() = default;

            virtual Express::VARP onComputeUpdateValue(Express::VARP param, Express::VARP grad) override;

            float getMomentum2();

            void setMomentum2(float momentum2);

            float getEps();

            void setEps(float eps);

            float getWeightDecay();

            void setWeightDecay(float decay);

            private:
            float mMomentum2 = 0.999; // default 0.999
            float mEps       = 1e-8;
            float mWeightDecay = 0.01;
            std::map<MNN::Express::VARP, MNN::Express::VARP> mHistory2;
        };
    } // namespace Train
} // namespace MNN


#endif //MNN_ADAMW_H
