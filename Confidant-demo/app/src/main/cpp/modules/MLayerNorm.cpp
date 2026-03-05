//
// Created by Yuhao Chen on 2023/5/22.
//

#include "MLayerNorm.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;

            MLayerNorm::MLayerNorm(std::vector<int> normalShape, bool elementwiseAffine, double eps): elementwiseAffine(elementwiseAffine), normalShape(normalShape) {
                this->eps = _Const(eps, {}, NCHW);
                if (elementwiseAffine) {
                    // TODO: order uncertain
                    weight = _TrainableParam(1.0f, normalShape, NCHW);
                    bias = _TrainableParam(0.0f, normalShape, NCHW);

                    addParameter(weight);
                    addParameter(bias);
                }
            }

            std::vector<Express::VARP> MLayerNorm::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                // TODO: The dimension value

                std::vector<int> dims;
                for (int i = 0; i < normalShape.size(); i++) {
                    int dim = -1 - i;
                    // int dim = normalShape.size() - i;
                    dims.push_back(dim);
                }
                reverse(dims.begin(), dims.end());

                auto mean = _ReduceMean(x, dims, true);
                auto var = _ReduceMean(_Square(_Subtract(x, mean)), dims, true);
                auto std = _Sqrt(var + eps);

                auto y = (x - mean) / std;

                if (elementwiseAffine) {
                    y = weight * y;
                    y = y + bias;
                }

                return {y};
            }
        }
    }
}