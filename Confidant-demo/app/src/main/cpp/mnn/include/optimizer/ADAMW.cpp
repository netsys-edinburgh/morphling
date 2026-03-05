//
// Created by 陈宇豪 on 2023/8/9.
//

#include "ADAMW.hpp"
#include "OpGrad.hpp"

using namespace MNN::Express;

namespace MNN {
    namespace Train {
        ADAMW::ADAMW(std::shared_ptr<Module> module) : SGD(module) {
            auto train = ParameterOptimizer::trainable();
            for (auto p : train) {
                mHistory2[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
            }
        }

        void ADAMW::setMomentum2(float momentum2) {
            mMomentum2 = momentum2;
        }

        void ADAMW::setEps(float eps) {
            mEps = eps;
        }

        float ADAMW::getMomentum2() {
            return mMomentum2;
        }

        float ADAMW::getEps() {
            return mEps;
        }

        void ADAMW::setWeightDecay(float decay) {
            mWeightDecay = decay;
        }

        float ADAMW::getWeightDecay() {
            return mWeightDecay;
        }

        Express::VARP ADAMW::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
            auto lr    = _Const(mLearningRate, {}, NCHW);
            auto step  = _Const(currentStep(), {}, NCHW);
            auto beta1 = _Const(mMomentum, {}, NCHW);
            auto beta2 = _Const(mMomentum2, {}, NCHW);
            auto eps   = _Const(mEps, {}, NCHW);
            auto weightDecay = _Const(mWeightDecay, {}, NCHW);
            // auto m = mHistory[param];
            // auto v = mHistory2[param];

            auto correction = _Sqrt(_Const(1.0f, {}, NCHW) - _Pow(beta2, step)) / (_Const(1.0f, {}, NCHW) - _Pow(beta1, step));

            // update of m
            mHistory[param] = beta1 * mHistory[param] + (_Const(1.0f, {}, NCHW) - beta1) * grad;
            mHistory[param].fix(Express::VARP::CONSTANT);

            // update of v
            mHistory2[param] = beta2 * mHistory2[param] + (_Const(1.0f, {}, NCHW) - beta2) * _Square(grad);
            mHistory2[param].fix(Express::VARP::CONSTANT);

            auto updateValue = lr * correction * (mHistory[param] / (_Sqrt(mHistory2[param]) + eps) + weightDecay * param);
            updateValue.fix(Express::VARP::CONSTANT);

            return updateValue;
        }

    } // namespace Train
} // namespace MNN