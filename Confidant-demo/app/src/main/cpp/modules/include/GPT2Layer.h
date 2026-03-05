//
// Created by Yuhao Chen on 2023/6/13.
//

#ifndef MNN_GPT2LAYER_H
#define MNN_GPT2LAYER_H

#include "MNN/expr/Module.hpp"
#include "NN.hpp"
#include "multiProcessorScheduler.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            class GPT2SelfAttention : public Module {
            private:
                int numAttentionHeads;
                int attentionHeadSize;
                int allHeadSize;
            public:
                GPT2SelfAttention(int hiddenSize, int numAttentionHeads);
                std::shared_ptr<Module> query, key, value;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class GPT2ParallelSelfAttention : public Module {
            private:
                int numAttentionHeads;
                int attentionHeadSize;
                int allHeadSize;
                int firstPart;
                std::once_flag mOnceFlag;
            public:
                GPT2ParallelSelfAttention(int hiddenSize, int numAttentionHeads, int firstPart=0);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                std::shared_ptr<Module> query, key, value;
                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue;
                std::shared_ptr<Module> concatQuery, concatKey, concatValue;
            };

            class GPT2GenericParallelSelfAttention : public Module {
            private:
                int numAttentionHeads;
                int attentionHeadSize;
                int allHeadSize;
                std::once_flag mOnceFlag;
                std::vector<Confidant::ProcessorInfo> allocationStrategy;
            public:
                GPT2GenericParallelSelfAttention(int hiddenSize, int numAttentionHeads, std::vector<Confidant::ProcessorInfo> allocationStrategy = {});
                std::shared_ptr<Module> query, key, value;

                std::vector<std::shared_ptr<Module>> parallelQuery, parallelKey, parallelValue;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            //包括自注意力（self）和自注意力输出（output）。这个类组合了自注意力机制的计算和输出。
            class GPT2Attention : public Module {
            public:
                GPT2Attention(int hiddenSize, int numAttentionHeads, bool forParallel);
                std::shared_ptr<Module> self;
                std::shared_ptr<Module> layerNorm;
                std::shared_ptr<Module> dense;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class GPT2Intermediate : public Module {
            public:
                GPT2Intermediate(int hiddenSize, int intermediateSize);
                std::shared_ptr<Module> dense1;
                std::shared_ptr<Module> dense2;//add by gsw
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class GPT2Output : public Module {
            public:
                GPT2Output(int hiddenSize, int intermediateSize );
                std::shared_ptr<Module> layerNorm;
                std::shared_ptr<GPT2Intermediate> intermediate;//add by gsw
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

// 它包括自注意力层（attention）、中间层（intermediate）和输出层（output）
            class MNN_PUBLIC GPT2Layer : public Express::Module {
            public:
                std::shared_ptr<GPT2Attention> attention;
                std::shared_ptr<GPT2Output> output;

                GPT2Layer(int numAttentionHeads, int hiddenSize, int intermediateSize, bool forParallel= false);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

        }
    }
}

#endif //MNN_GPT2LAYER_H
