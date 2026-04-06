#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>

#include "core/env_cfg.h"

namespace morphling {
namespace backend {

class DispatchGate {
 public:
  DispatchGate(DeviceMode mode, int64_t barrier_count,
               int64_t barrier_timeout_ms, int64_t max_queue_size);

  // BARRIER: blocks until barrier_count devices joined (or timeout).
  //          Returns true if barrier met, false on timeout.
  // DYNAMIC: returns true immediately (caller checks device count separately).
  bool WaitForReady();

  void NotifyDeviceJoined(int64_t device_id);
  void NotifyDeviceLeft(int64_t device_id);

  bool EnqueueWork(std::function<void()> dispatch_fn);
  void FlushQueue();

  bool IsBarrierMet() const;
  size_t GetQueueSize() const;
  DeviceMode GetMode() const { return mode_; }

 private:
  const DeviceMode mode_;
  const int64_t barrier_count_;
  const int64_t barrier_timeout_ms_;
  const int64_t max_queue_size_;

  mutable std::mutex mutex_;
  std::condition_variable barrier_cv_;
  bool barrier_met_ = false;
  int64_t connected_count_ = 0;

  std::deque<std::function<void()>> work_queue_;
  std::condition_variable queue_not_full_cv_;
};

}  // namespace backend
}  // namespace morphling
