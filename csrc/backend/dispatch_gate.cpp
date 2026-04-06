#include "dispatch_gate.h"

#include "muduo_base/logging.h"

namespace morphling {
namespace backend {

DispatchGate::DispatchGate(DeviceMode mode, int64_t barrier_count,
                           int64_t barrier_timeout_ms, int64_t max_queue_size)
    : mode_(mode),
      barrier_count_(barrier_count),
      barrier_timeout_ms_(barrier_timeout_ms),
      max_queue_size_(max_queue_size) {}

bool DispatchGate::WaitForReady() {
  if (mode_ == DeviceMode::DYNAMIC) {
    return true;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  if (barrier_met_) {
    return true;
  }

  auto predicate = [this]() { return barrier_met_; };

  if (barrier_timeout_ms_ <= 0) {
    barrier_cv_.wait(lock, predicate);
    return true;
  }

  return barrier_cv_.wait_for(
      lock, std::chrono::milliseconds(barrier_timeout_ms_), predicate);
}

void DispatchGate::NotifyDeviceJoined(int64_t device_id) {
  std::vector<std::function<void()>> to_flush;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ++connected_count_;

    if (mode_ == DeviceMode::BARRIER && !barrier_met_ &&
        connected_count_ >= barrier_count_) {
      barrier_met_ = true;
      LOG_INFO << "Barrier met: " << connected_count_ << "/" << barrier_count_
               << " devices connected";
      barrier_cv_.notify_all();
    }

    if (mode_ == DeviceMode::DYNAMIC && !work_queue_.empty()) {
      to_flush.assign(std::make_move_iterator(work_queue_.begin()),
                      std::make_move_iterator(work_queue_.end()));
      work_queue_.clear();
      queue_not_full_cv_.notify_all();
    }
  }

  for (auto& fn : to_flush) {
    fn();
  }
}

void DispatchGate::NotifyDeviceLeft(int64_t device_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (connected_count_ > 0) {
    --connected_count_;
  }
  // Barrier is one-shot: do not re-arm on device departure
}

bool DispatchGate::EnqueueWork(std::function<void()> dispatch_fn) {
  std::unique_lock<std::mutex> lock(mutex_);
  queue_not_full_cv_.wait(lock, [this]() {
    return static_cast<int64_t>(work_queue_.size()) < max_queue_size_;
  });
  work_queue_.push_back(std::move(dispatch_fn));
  return true;
}

void DispatchGate::FlushQueue() {
  std::vector<std::function<void()>> to_flush;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    to_flush.assign(std::make_move_iterator(work_queue_.begin()),
                    std::make_move_iterator(work_queue_.end()));
    work_queue_.clear();
    queue_not_full_cv_.notify_all();
  }

  for (auto& fn : to_flush) {
    fn();
  }
}

bool DispatchGate::IsBarrierMet() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return barrier_met_;
}

size_t DispatchGate::GetQueueSize() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return work_queue_.size();
}

}  // namespace backend
}  // namespace morphling
