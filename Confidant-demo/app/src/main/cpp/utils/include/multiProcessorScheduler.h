//
// Created by lanbai on 2023/10/30.
//

#ifndef CONFIDANT_MULTIPROCESSORSCHEDULER_H
#define CONFIDANT_MULTIPROCESSORSCHEDULER_H

#include <unordered_map>
#include <set>
#include <MNN/expr/Module.hpp>

namespace Confidant {
    bool check(float time, std::unordered_map<int, std::vector<std::pair<int, float>>>& backendProfiling, std::set<int>& backendSet);

    typedef enum {
        ONE_ATTN_HEAD = 0, // compute attention heads as one large head
        SEP_SEQ_ATTN_HEAD = 1, // compute attention heads separately and sequentially
        SEP_PARALLEL_ATTN_HEAD = 2 // compute attention heads separately and parallelly
    } AttentionHeadComputeWay;

    struct ProcessorInfo {
        MNNForwardType type;
        int numThread; // used for type
        int numAttentionHead;
        AttentionHeadComputeWay computeWay;

        // constructor
        ProcessorInfo(MNNForwardType type, int numThread, int numAttentionHead, AttentionHeadComputeWay computeWay) {
            this->type = type;
            this->numThread = numThread;
            this->numAttentionHead = numAttentionHead;
            this->computeWay = computeWay;
        }

        explicit ProcessorInfo(int numAttentionHead) {
            this->type = MNN_FORWARD_CPU;
            this->numThread = 1;
            this->numAttentionHead = numAttentionHead;
            this->computeWay = ONE_ATTN_HEAD;
        }

    };

    class MultiProcessorScheduler {
    private:
        int numAttentionHead;
    public:
        std::unordered_map<MNNForwardType, std::vector<float>> profilingResult;
        std::pair<float, std::vector<ProcessorInfo>> allocationStrategy;
        void profileProcessors(std::string& modelName, std::map<std::string, double> &modelArgs);
        bool check(float time);
        void computeAllocationStrategy();
        std::pair<float, std::vector<ProcessorInfo>> getAllocationStrategy();
        void setAllocationStrategy(std::pair<float, std::vector<ProcessorInfo>> allocationStrategy);

        MultiProcessorScheduler() = default;
        static std::shared_ptr<MultiProcessorScheduler> mpsPtr;
    };
}

#endif //FTPIPEHD_MNN_BACKENDSCHEDULER_H
