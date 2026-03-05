//
// Created by lanbai on 2023/10/30.
//

#include "multiProcessorScheduler.h"
#include "MNN/expr/Module.hpp"
#include "NN.hpp"
#include "log.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <utility>
#include "BERTLayer.h"
#include "commonStates.h"
#include "BERT.h"
#include "GPT2.h"
#include "Phi2.h"
#include "LLaMA.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace std;

namespace Confidant {
    std::shared_ptr<MultiProcessorScheduler> MultiProcessorScheduler::mpsPtr = nullptr;

    void MultiProcessorScheduler::computeAllocationStrategy() {
        assert(profilingResult.find(MNN_FORWARD_CPU) != profilingResult.end());
        int n = profilingResult[MNN_FORWARD_CPU].size();
        float minTime = numeric_limits<float>::max();

        for (auto& ret : profilingResult) {
            minTime = std::min(minTime, ret.second[n - 1]);
        }

        float l = 0, r = minTime;
        while (l <= r) {
            float mid = l + (r - l) / 2.0;
            if (check(mid)) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }

        // print the original allocation strategy
        for (int i = 0; i < allocationStrategy.second.size(); i++) {
            LOGI("computeAllocationStrategy(): Processor %d: %d\n", i, allocationStrategy.second[i].numAttentionHead);
        }

        // Correct the allocation strategy since the totalHeads may larger than the numAttentionHeads
        int target = numAttentionHead;
        for (auto& curStrategy : allocationStrategy.second) {
            if (curStrategy.type == MNN_FORWARD_CPU) {
                target -= curStrategy.numAttentionHead;
                continue;
            }

            if (curStrategy.numAttentionHead > target) {
                curStrategy.numAttentionHead = target;
                target = 0;
            } else {
                target -= curStrategy.numAttentionHead;
            }
        }

//        for (auto& curStrategy : allocationStrategy.second) {
//            if (curStrategy.type == MNN_FORWARD_CPU) {
//                curStrategy.numAttentionHead = target;
//                break;
//            }
//        }

        LOGI("The minimum time is %f\n", l);
    }

    bool MultiProcessorScheduler::check(float time) {
        int totalHeads = 0;
        int threshold = 300;
        std::vector<ProcessorInfo> curAllocation;

        for (auto& backendRet : profilingResult) {
            int curHeads = 0;
            float minDiff = 1e9;
            for (int i = 0; i < backendRet.second.size(); i++) {
                int curTime = backendRet.second[i];
                if (abs(curTime - time) < minDiff) {
                    curHeads = i + 1; // attention heads start from 1
                    minDiff = abs(curTime - time);
                }
            }

            // Differences too large
            if (minDiff >= threshold) {
                curHeads = 0;
            }
            curAllocation.emplace_back(ProcessorInfo(backendRet.first, 1, curHeads, ONE_ATTN_HEAD));
            totalHeads += curHeads;
        }

        if (totalHeads >= numAttentionHead) {
            allocationStrategy = {time, curAllocation};
        }
        return totalHeads >= numAttentionHead;
    }

    /*
     * Profile the processors by measuring the computation time using different number of attention heads
     */
    void MultiProcessorScheduler::profileProcessors(std::string& modelName, std::map<std::string, double> &modelArgs) {
        int batchSize = CommonStates::getBatchSize();
        LOGI("Start profiling processors with batch size %d\n", batchSize);
        if (modelName == "BERTForClassification") {
            profilingResult = Model::BERTProfileProcessors(batchSize, modelArgs);
            numAttentionHead = profilingResult[MNN_FORWARD_CPU].size();
        } else if (modelName == "GPT2") {
            profilingResult = Model::GPT2ProfileProcessors(batchSize, modelArgs);
            numAttentionHead = profilingResult[MNN_FORWARD_CPU].size();
        } else if (modelName == "Phi2Alpaca") {
            profilingResult = Model::Phi2ProfileProcessors(batchSize, modelArgs);
            numAttentionHead = profilingResult[MNN_FORWARD_CPU].size();
        } else if (modelName == "LLaMALora") {
            profilingResult = Model::LLaMAProfileProcessors(batchSize, modelArgs);
            numAttentionHead = profilingResult[MNN_FORWARD_CPU].size();
        } else {
            LOGI("profileProcessors(): Unsupported model name\n");
        }
    }

    std::pair<float, std::vector<ProcessorInfo>> MultiProcessorScheduler::getAllocationStrategy() {
        return allocationStrategy;
    }

    void MultiProcessorScheduler::setAllocationStrategy(std::pair<float, std::vector<ProcessorInfo>> _allocationStrategy) {
        this->allocationStrategy = std::move(_allocationStrategy);
    }
}