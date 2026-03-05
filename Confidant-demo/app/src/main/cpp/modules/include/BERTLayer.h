//
// Created by Yuhao Chen on 2023/6/16.
//

#ifndef MNN_BERTLAYER_H
#define MNN_BERTLAYER_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "multiProcessorScheduler.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            class BERTSelfOutput : public Module {
            public:
                BERTSelfOutput(int hiddenSize, float dropoutProb = 0.1);
                std::shared_ptr<Module> dense;
                std::shared_ptr<Module> layerNorm;
                std::shared_ptr<Module> dropout;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class BERTSelfAttention : public Module {
            private:
                int numAttentionHeads;
                int attentionHeadSize;
                int allHeadSize;
            public:
                BERTSelfAttention(int hiddenSize, int numAttentionHeads, float attentionProbsDropoutProb = 0.1, bool forParallel = false);
                std::shared_ptr<Module> query, key, value;
                std::shared_ptr<Module> dropout;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                bool forParallel;
            };

            class BERTParallelSelfAttention : public Module {
            private:
                int numAttentionHeads;
                int attentionHeadSize;
                int allHeadSize;
                std::once_flag mOnceFlag;
                std::vector<Confidant::ProcessorInfo> allocationStrategy;
            public:
                BERTParallelSelfAttention(int hiddenSize, int numAttentionHeads, float attentionProbsDropoutProb = 0.1, std::vector<Confidant::ProcessorInfo> allocationStrategy = {});
                std::shared_ptr<Module> query, key, value;
                std::shared_ptr<Module> dropout;

                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue, dropouts;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class ParallelSelfAttention : public Module {
            private:
                int numAttentionHeads;
                int attentionHeadSize;
                int allHeadSize;
                std::once_flag mOnceFlag;
                int firstPart;
            public:
                ParallelSelfAttention(int hiddenSize, int numAttentionHeads, float attentionProbsDropoutProb = 0.1, int firstPart = 6);
                std::shared_ptr<Module> query, key, value;
                std::shared_ptr<Module> dropout;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue, dropouts;
                // for multiple backends
                std::shared_ptr<Module> concatQuery, concatKey, concatValue;
            };

            class BERTAttention : public Module {
            public:
                BERTAttention(int hiddenSize, int numAttentionHeads, float dropoutProb = 0.1, bool forParallel = false);
                std::shared_ptr<BERTSelfAttention> self;
                std::shared_ptr<BERTParallelSelfAttention> paraSelf;
                std::shared_ptr<BERTSelfOutput> output;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                bool forParallel;
            };

            class BERTIntermediate : public Module {
            public:
                BERTIntermediate(int hiddenSize, int intermediateSize);
                std::shared_ptr<Module> dense;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class BERTOutput : public Module {
            public:
                BERTOutput(int hiddenSize, int intermediateSize, float dropoutProb = 0.1);
                std::shared_ptr<Module> dense;
                std::shared_ptr<Module> layerNorm;
                std::shared_ptr<Module> dropout;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class MNN_PUBLIC BERTLayer : public Express::Module {
            public:
                std::shared_ptr<BERTAttention> attention;
                std::shared_ptr<BERTIntermediate> intermediate;
                std::shared_ptr<BERTOutput> output;

                BERTLayer(int numAttentionHeads, int hiddenSize, int intermediateSize, float dropoutProb = 0.1, bool forParallel = false);

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };
        }
    }
}

#endif //MNN_BERTLAYER_H
