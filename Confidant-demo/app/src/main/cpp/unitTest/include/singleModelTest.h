//
// Created by Yuhao Chen on 2024/2/28.
//

#ifndef CONFIDANT_SINGLEMODELTEST_H
#define CONFIDANT_SINGLEMODELTEST_H
#include "ParameterOptimizer.hpp"

using namespace MNN;
using namespace MNN::Train;

namespace Confidant {
    void BERTSelfAttnTest();
    void LLaMASelfAttnTest();
    void smallChunkBERTSelfAttnTest();
    void smallChunkBERTTest();
    void smallChunkBERTNewImplTest();
    void singleBERTParallelTest();
    void singleBERTTest();
    void singleGPT2Test();
}


#endif //CONFIDANT_SINGLEMODELTEST_H
