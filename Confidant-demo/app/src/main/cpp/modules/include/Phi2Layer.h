//
// Created by yue on 2024/3/12.
//

#ifndef FTPIPEHD_MNN_PHI2LAYER_H
#define FTPIPEHD_MNN_PHI2LAYER_H


#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "ModelArgs.h"
#include "MLayerNorm.h"
#include "multiProcessorScheduler.h"


namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;
//            VARP precomputeThetaPosFrequencies(int headDim, int seqLen, float theta = 10000.0f);
            VARP applyRotaryEmbedding(VARP x, VARP freqsComplex);
            VARP repeatKV(VARP x, int nRepeat);

            class PhiLoRALayer : public Module{
            private:
                int hiddenSize;
                float alpha;
                int r;
            public:
                PhiLoRALayer(PhiArgs& args, int inFeatures,int outFeatures);
                VARP loraA, loraB;
                VARP weight;
                std::shared_ptr<Module> loraDropout;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class PhiSelfAttention : public Module {
            private:
                int numHeads;
                int hiddenSize;
                int numKVHeads;
                int headDim;
                std::map<std::tuple<int, int, int>, VARP> cacheK, cacheV;
            public:
                PhiSelfAttention(PhiArgs& args);
                std::shared_ptr<Module> query, key, value, out;
                VARP cacheKey, cacheValue;

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue, parallelDropout;
                // bool forParallel;
            };

            class Phi2ParallelSelfAttention : public Module {
            private:
                int numHeads;
                int hiddenSize;
                int numKVHeads;
                int headDim;

                std::once_flag mOnceFlag;
                std::vector<Confidant::ProcessorInfo> allocationStrategy;
            public:
                Phi2ParallelSelfAttention(PhiArgs &args, std::vector<Confidant::ProcessorInfo> allocationStrategy = {});
                std::shared_ptr<Module> query, key, value, out;

                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class PhiFeedForward : public Module {
            public:
                PhiFeedForward(int hiddenSize, int intermediateSize);
                std::shared_ptr<Module> fc1,fc2;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class MNN_PUBLIC PhiLayer : public Express::Module {
            private:
                int headDim;
                int numHeads;
            public:
                std::shared_ptr<PhiSelfAttention> selfAttention;
                std::shared_ptr<PhiFeedForward> mlp;
                // Normalization BEFORE the attention/feedforward block
                std::shared_ptr<Module> input_layernorm;
                std::shared_ptr<Express::Module> resid_dropout;

                PhiLayer(PhiArgs& args);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

        }
    }
}


#endif //FTPIPEHD_MNN_PHI2LAYER_H
