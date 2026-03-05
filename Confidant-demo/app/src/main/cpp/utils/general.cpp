//
// Created by Yuhao Chen on 2023/1/10.
//

#include "general.h"
#include "log.h"
#include <android/bitmap.h>
#include "MNN/expr/NeuralNetWorkOp.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    std::vector<VARP> cloneParams(std::vector<VARP>& params) {
        int n = params.size();
        std::vector<VARP> copiedParams(n);
        for (int i = 0; i < n; i++) {
            copiedParams[i] = _Clone(params[i], true);
//            auto info = params[i]->getInfo();
//            auto ptr = params[i]->readMap<void>();  // 计算并且存入 cache？
//            if (nullptr == ptr) {
//                MNN_ERROR("Compute error in SGD\n");
//                return {};
//            }
//            auto newVar = _Const(ptr, info->dim, info->order, info->type);
//            copiedParams[i]= newVar;
        }
        return copiedParams;
    }

    void copyParameters(std::vector<VARP>& src, std::vector<VARP>& dst) {
        int n = src.size();
        if (src.empty() || src.size() != dst.size()) {
            LOGE("Error trainable parameters, empty or parameter size not match \n");
            return ;
        }

        // Copied from MNN source code
        for (int i = 0; i < n; ++i) {
            if (nullptr != dst[i].get()) {
                // Check Origin parameter's size
                auto dstInfo = dst[i]->getInfo();
                auto srcInfo = src[i]->getInfo();
                if (dstInfo->dim.size() != srcInfo->dim.size() || dstInfo->order != srcInfo->order) {
                    LOGE("Error parameters %d, dim size or order not match \n", i);
                    return ;
                }
                if (dstInfo->size != srcInfo->size || dstInfo->type != srcInfo->type) {
                    LOGE("Error parameters %d, size or type not match \n", i);
                    return ;
                }
            }
            Variable::replace(dst[i], src[i]);
        }
    }
}
