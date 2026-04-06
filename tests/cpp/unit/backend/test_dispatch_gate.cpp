#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "backend/dispatch_gate.h"

namespace {

using morphling::backend::DispatchGate;

class DispatchGateTest : public ::testing::Test {};

TEST_F(DispatchGateTest, BarrierBlocksUntilCountMet) {
  using namespace std::chrono_literals;

  DispatchGate gate(DeviceMode::BARRIER, 2, 2000, 16);
  std::atomic<bool> wait_finished{false};
  std::atomic<bool> wait_result{false};

  std::thread waiter([&]() {
    wait_result.store(gate.WaitForReady());
    wait_finished.store(true);
  });

  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(wait_finished.load());

  gate.NotifyDeviceJoined(1);
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(wait_finished.load());

  gate.NotifyDeviceJoined(2);
  waiter.join();

  EXPECT_TRUE(wait_finished.load());
  EXPECT_TRUE(wait_result.load());
  EXPECT_TRUE(gate.IsBarrierMet());
}

TEST_F(DispatchGateTest, BarrierTimeoutReturnsFalse) {
  DispatchGate gate(DeviceMode::BARRIER, 2, 100, 16);

  auto start = std::chrono::steady_clock::now();
  bool ready = gate.WaitForReady();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  EXPECT_FALSE(ready);
  EXPECT_GE(elapsed.count(), 80);
}

TEST_F(DispatchGateTest, DynamicEnqueueFlush) {
  DispatchGate gate(DeviceMode::DYNAMIC, 0, 0, 16);
  std::atomic<int> flushed{0};

  ASSERT_TRUE(gate.EnqueueWork([&]() { flushed.fetch_add(1); }));
  ASSERT_TRUE(gate.EnqueueWork([&]() { flushed.fetch_add(1); }));
  ASSERT_TRUE(gate.EnqueueWork([&]() { flushed.fetch_add(1); }));

  EXPECT_EQ(gate.GetQueueSize(), 3u);
  gate.NotifyDeviceJoined(100);
  EXPECT_EQ(flushed.load(), 3);
  EXPECT_EQ(gate.GetQueueSize(), 0u);
}

TEST_F(DispatchGateTest, DynamicBackPressure) {
  using namespace std::chrono_literals;

  DispatchGate gate(DeviceMode::DYNAMIC, 0, 0, 2);
  ASSERT_TRUE(gate.EnqueueWork([]() {}));
  ASSERT_TRUE(gate.EnqueueWork([]() {}));

  std::atomic<bool> enqueue_started{false};
  std::atomic<bool> enqueue_finished{false};

  std::thread producer([&]() {
    enqueue_started.store(true);
    gate.EnqueueWork([]() {});
    enqueue_finished.store(true);
  });

  while (!enqueue_started.load()) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(60ms);
  EXPECT_FALSE(enqueue_finished.load());

  gate.FlushQueue();
  producer.join();

  EXPECT_TRUE(enqueue_finished.load());
  EXPECT_EQ(gate.GetQueueSize(), 1u);

  gate.FlushQueue();
  EXPECT_EQ(gate.GetQueueSize(), 0u);
}

TEST_F(DispatchGateTest, BarrierIsOneShot) {
  DispatchGate gate(DeviceMode::BARRIER, 1, 1000, 16);

  gate.NotifyDeviceJoined(7);
  EXPECT_TRUE(gate.IsBarrierMet());

  gate.NotifyDeviceLeft(7);
  EXPECT_TRUE(gate.IsBarrierMet());
  EXPECT_TRUE(gate.WaitForReady());
}

TEST_F(DispatchGateTest, DynamicModeWaitForReadyReturnsTrue) {
  DispatchGate gate(DeviceMode::DYNAMIC, 0, 0, 16);
  EXPECT_TRUE(gate.WaitForReady());
}

TEST_F(DispatchGateTest, GetModeReturnsCorrectMode) {
  DispatchGate barrier_gate(DeviceMode::BARRIER, 1, 0, 16);
  EXPECT_EQ(barrier_gate.GetMode(), DeviceMode::BARRIER);

  DispatchGate dynamic_gate(DeviceMode::DYNAMIC, 0, 0, 16);
  EXPECT_EQ(dynamic_gate.GetMode(), DeviceMode::DYNAMIC);
}

}  // namespace
