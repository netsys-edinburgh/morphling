//
// Created by Yuhao Chen on 2023/10/17.
//

#ifndef MNN_LLAMALAYER_H
#define MNN_LLAMALAYER_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "MRMSNorm.h"
#include "ModelArgs.h"
#include "multiProcessorScheduler.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            VARP precomputeThetaPosFrequencies(int headDim, int seqLen, float theta = 10000.0f);

            // MNN Implementation of LoRA
            class LLaMALoRALayer : public Module {
            private:
                int hiddenSize;
                float alpha;
                int r;
            public:
                LLaMALoRALayer(LLaMAArgs& args, int inFeatures, int outFeatures);
                VARP loraA, loraB;
                VARP weight;
                std::shared_ptr<Module> loraDropout;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class LLaMASelfAttention : public Module {
            private:
                int numHeads;
                int hiddenSize;
                int numKVHeads;
                int headDim;
                std::map<std::tuple<int, int, int>, VARP> cacheK, cacheV;
            public:
                LLaMASelfAttention(LLaMAArgs& args);
                std::shared_ptr<Module> query, key, value, out;
                VARP cacheKey, cacheValue;

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                // bool forParallel;
            };

            class LLaMAParallelSelfAttention : public Module {
            private:
                int numHeads;
                int hiddenSize;
                int numKVHeads;
                int headDim;

                std::once_flag mOnceFlag;
                std::vector<Confidant::ProcessorInfo> allocationStrategy;
            public:
                LLaMAParallelSelfAttention(LLaMAArgs &args, std::vector<Confidant::ProcessorInfo> allocationStrategy = {});
                std::shared_ptr<Module> query, key, value, out;

                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };


//            class LLaMAParallelSelfAttention : public Module {
//            private:
//                int numHeads;
//                int hiddenSize;
//                int numKVHeads;
//                int headDim;
//                int firstPart;
//                std::once_flag mOnceFlag;
//                std::map<std::tuple<int, int, int>, VARP> cacheK, cacheV;
//            public:
//                LLaMAParallelSelfAttention(LLaMAArgs& args, int firstPart=6);
//                std::shared_ptr<Module> query, key, value, out;
//
//                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
//                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue;
//                std::shared_ptr<Module> concatQuery, concatKey, concatValue;
//            };



            class LLaMAFeedForward : public Module {
            public:
                LLaMAFeedForward(int hiddenSize, int multipleOf, int ffnDimMultiplier = -1);
                std::shared_ptr<Module> w1, w2, w3;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class MNN_PUBLIC LLaMALayer : public Express::Module {
            private:
                int headDim;
                int numHeads;
            public:
                std::shared_ptr<LLaMASelfAttention> selfAttention;
                std::shared_ptr<LLaMAFeedForward> feedForward;
                // Normalization BEFORE the attention/feedforward block
                std::shared_ptr<MRMSNorm> attentionNorm, ffnNorm;
                LLaMALayer(LLaMAArgs& args);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };
        }
    }
}

#endif //MNN_LLAMALAYER_H
