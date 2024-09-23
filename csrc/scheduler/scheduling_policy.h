#pragma once

#include <tuple>

#include "common/types_and_defs.h"

#define SCHEDULING_POLICY_VALUES(X, EnumType) X(kRoundRobinGemm, EnumType)

DEFINE_ENUM_CLASS(SchedulingPolicyType, SCHEDULING_POLICY_VALUES)

class SchedulingPolicy {
 public:
  SchedulingPolicy() = default;
  virtual ~SchedulingPolicy() = default;

  // schedule for <gpu_id, priority>
  // args: the input arguments for scheduling, need reinterpret_cast to the
  // specific type
  virtual std::tuple<int, int> Schedule(void* args) = 0;
};

class RoundRobinGemmPolicy : public SchedulingPolicy {
 public:
  RoundRobinGemmPolicy(int num_gpus) : num_gpus_(num_gpus) {}
  ~RoundRobinGemmPolicy() = default;

  std::tuple<int, int> Schedule(void* args) override {
    int gpu_id = next_gpu_id_;
    next_gpu_id_ = (next_gpu_id_ + 1) % num_gpus_;
    return {gpu_id, 0};
  }

 private:
  int num_gpus_;
  int next_gpu_id_ = 0;
};