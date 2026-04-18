#pragma once

#include <cstdint>

namespace morphling {
namespace backend {

class PartitionSchedulingPolicy {
 public:
  virtual ~PartitionSchedulingPolicy() = default;
};

class RoundRobinSchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  RoundRobinSchedulingPolicy(int64_t, bool) {}
};

class GreedySchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  GreedySchedulingPolicy(int64_t, bool) {}
};

class LoadBalancedSchedulingPolicy : public PartitionSchedulingPolicy {
 public:
  LoadBalancedSchedulingPolicy(int64_t, bool) {}
};

}  // namespace backend
}  // namespace morphling
