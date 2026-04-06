#pragma once

#include <memory>
#include <string>

namespace morphling {
namespace backend {
class PartitionSchedulingPolicy;
}
}  // namespace morphling

enum class SchedulingPolicyType {
  ROUND_ROBIN = 0,
  GREEDY = 1,
  LOAD_BALANCED = 2
};

enum class DeviceMode {
  BARRIER = 0,  // Wait for N devices before dispatching
  DYNAMIC = 1,  // Elastic — devices can join/leave at any time
};

struct ProxyEnvCfg {
  ProxyEnvCfg();
  ~ProxyEnvCfg();

  std::string listen_ip;
  unsigned listen_port;

  // worker
  int thread;
  int64_t block_size;
  int64_t num_device;
  int max_inflight;
  int wait_matmul_timeout_s = 300;
  int stuck_threshold_ms = 30000;
  bool enable_auto_recovery = true;
  int send_high_water_mark = 4194304;  // 4MB
  int max_batch_per_device = 50;
  std::string pool_mode;  // "gpu", "cpu", or "both"
  std::string loop_strategy;

  // log
  std::string log_path;
  std::string log_level;
  std::string log_type;

  // internal
  int cleanup_wait;
  int64_t tcp_timeout;
  int enable_cli_cache;

  void* instance;

  // Scheduling policy configuration
  SchedulingPolicyType sched_policy_type;
  std::unique_ptr<morphling::backend::PartitionSchedulingPolicy> sched_policy;

  DeviceMode device_mode = DeviceMode::BARRIER;
  int64_t barrier_count = 0;       // 0 = use num_device
  int64_t barrier_timeout_ms = 0;  // 0 = infinite
  int64_t max_queue_size = 1024;

  // Get scheduling policy with type-based reinterpretation
  template <typename T>
  T* GetSchedPolicy() {
    return dynamic_cast<T*>(sched_policy.get());
  }

  int Initialize(const std::string& cfg_file);
};
