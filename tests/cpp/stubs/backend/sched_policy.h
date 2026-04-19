#pragma once

#include <cstdint>

namespace morphling {
namespace backend {

class PartitionSchedulingPolicy {
 public:
  virtual ~PartitionSchedulingPolicy() {}
};

class RoundRobinSchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  RoundRobinSchedulingPolicy(int64_t, bool) {}
  ~RoundRobinSchedulingPolicy() override {}
};

class GreedySchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  GreedySchedulingPolicy(int64_t, bool) {}
  ~GreedySchedulingPolicy() override {}
};

class LoadBalancedSchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  LoadBalancedSchedulingPolicy(int64_t, bool) {}
  ~LoadBalancedSchedulingPolicy() override {}
};

}  // namespace backend
}  // namespace morphling
