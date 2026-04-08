#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "../../../csrc/core/env_cfg.h"

namespace morphling {
namespace backend {
class DispatchGate {
 public:
  DispatchGate(DeviceMode mode, int64_t barrier_count,
               int64_t barrier_timeout_ms, int64_t max_queue_size);
  bool WaitForReady();
  void NotifyDeviceJoined(int64_t device_id);
  bool IsBarrierMet() const;

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

TEST(BarrierIntegrationTest, EndToEndBarrierUnblocksAfterTwoDevicesJoin) {
  using morphling::backend::DispatchGate;
  using namespace std::chrono_literals;

  DispatchGate gate(DeviceMode::BARRIER, 2, 2000, 16);

  std::atomic<bool> wait_returned{false};
  std::atomic<bool> wait_result{false};
  std::thread waiter([&]() {
    wait_result.store(gate.WaitForReady());
    wait_returned.store(true);
  });

  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(wait_returned.load());

  gate.NotifyDeviceJoined(1001);
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(wait_returned.load());

  gate.NotifyDeviceJoined(1002);
  waiter.join();

  EXPECT_TRUE(wait_returned.load());
  EXPECT_TRUE(wait_result.load());
  EXPECT_TRUE(gate.IsBarrierMet());
}
