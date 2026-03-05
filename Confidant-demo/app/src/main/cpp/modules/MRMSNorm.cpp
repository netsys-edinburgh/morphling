//
// Created by Yuhao Chen on 2023/10/16.
//

#include "MRMSNorm.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;

            MRMSNorm::MRMSNorm(int dim, float eps) {
                this->eps = _Const(eps, {}, NCHW);;
                weight = _Const(1.0f, {dim}, NCHW);
                addParameter(this->weight);
            }

            VARP MRMSNorm::_Norm(VARP x) {
                auto mean = _ReduceMean(x, {-1}, true);
                // _Divide(x, _Sqrt(_ReduceMean(_Pow(x, _Const(2.0f, {}, NCHW)), {-1}, true) + eps));
                return x * _Rsqrt(_ReduceMean(_Pow(x, _Const(2.0f, {}, NCHW)), {-1}, true) + eps);
            }

            std::vector<Express::VARP> MRMSNorm::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                auto ptr = _Norm(x)->readMap<float>();
                auto weightPtr = weight->readMap<float>();
                auto y = weight * _Norm(x);
                auto ptr2 = y->readMap<float>();
                return {y};
            }
        }
    }
}