#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "backend/partition_tracker.h"

namespace morphling::backend {
namespace {

TEST(PartitionTrackerClaimTest, ConcurrentClaimersClaimEachPartitionOnce) {
  // Given
  auto& tracker = PARTITION_TRACKER;
  tracker.Reset();
  constexpr int kPartitionCount = 100;
  constexpr int kClaimerCount = 8;
  for (int index = 0; index < kPartitionCount; ++index) {
    auto partition = std::make_shared<MatrixPartition>();
    tracker.AddPartition(-1, "partition-" + std::to_string(index), 7,
                         partition);
  }
  std::atomic<bool> start{false};
  std::mutex claimed_mutex;
  std::vector<std::string> claimed_keys;
  std::vector<std::thread> claimers;

  // When
  for (int index = 0; index < kClaimerCount; ++index) {
    claimers.emplace_back([&, index] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto claimed = tracker.ClaimIdlePartitions();
      std::lock_guard<std::mutex> lock(claimed_mutex);
      for (const auto& partition : claimed) {
        EXPECT_TRUE(tracker.ReassignPartitionToDevice(partition->key, index));
        claimed_keys.push_back(partition->key);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& claimer : claimers) {
    claimer.join();
  }

  // Then
  const std::unordered_set<std::string> unique_keys(claimed_keys.begin(),
                                                    claimed_keys.end());
  EXPECT_EQ(claimed_keys.size(), kPartitionCount);
  EXPECT_EQ(unique_keys.size(), kPartitionCount);
  EXPECT_TRUE(tracker.GetIdlePartitions().empty());
  size_t relocated_count = 0;
  for (int index = 0; index < kClaimerCount; ++index) {
    relocated_count += tracker.GetDevicePartitionCount(index);
  }
  EXPECT_EQ(relocated_count, kPartitionCount);
  tracker.Reset();
}

TEST(PartitionTrackerClaimTest, ReassignmentPreservesClaimedPartitionIdentity) {
  // Given
  auto& tracker = PARTITION_TRACKER;
  tracker.Reset();
  auto matrix_partition = std::make_shared<MatrixPartition>();
  tracker.AddPartition(-1, "partition", 9, matrix_partition);
  auto claimed = tracker.ClaimIdlePartitions();
  ASSERT_EQ(claimed.size(), 1);
  const auto claimed_info = claimed.front();

  // When
  ASSERT_TRUE(tracker.ReassignPartitionToDevice("partition", 4));
  ASSERT_TRUE(tracker.ReassignPartitionToDevice("partition", 2));

  // Then
  EXPECT_TRUE(tracker.GetDevicePartitions(-1).empty());
  EXPECT_TRUE(tracker.GetDevicePartitions(4).empty());
  const auto target_partitions = tracker.GetDevicePartitions(2);
  ASSERT_EQ(target_partitions.size(), 1);
  EXPECT_EQ(target_partitions.front(), claimed_info);
  EXPECT_EQ(target_partitions.front()->partition, matrix_partition);
  EXPECT_EQ(target_partitions.front()->owner_device_id, 2);
  EXPECT_EQ(target_partitions.front()->partition->dev_id, 2);
  EXPECT_EQ(target_partitions.front()->state, PartitionState::RUNNING);
  EXPECT_TRUE(tracker.GetIdlePartitions().empty());
  tracker.Reset();
}

TEST(PartitionTrackerClaimTest, RevertedClaimCanBeClaimedOnceAgain) {
  // Given
  auto& tracker = PARTITION_TRACKER;
  tracker.Reset();
  auto matrix_partition = std::make_shared<MatrixPartition>();
  tracker.AddPartition(-1, "partition", 11, matrix_partition);
  auto first_claim = tracker.ClaimIdlePartitions();
  ASSERT_EQ(first_claim.size(), 1);
  ASSERT_TRUE(tracker.ReassignPartitionToDevice("partition", 3));

  // When
  tracker.MarkPartitionIdle("partition");
  auto second_claim = tracker.ClaimIdlePartitions();
  auto third_claim = tracker.ClaimIdlePartitions();

  // Then
  ASSERT_EQ(second_claim.size(), 1);
  EXPECT_EQ(second_claim.front(), first_claim.front());
  EXPECT_TRUE(third_claim.empty());
  EXPECT_TRUE(tracker.GetIdlePartitions().empty());
  tracker.Reset();
}

TEST(PartitionTrackerClaimTest, ReassignmentRejectsUnknownPartition) {
  // Given
  auto& tracker = PARTITION_TRACKER;
  tracker.Reset();

  // When
  const bool reassigned = tracker.ReassignPartitionToDevice("missing", 1);

  // Then
  EXPECT_FALSE(reassigned);
}

}  // namespace
}  // namespace morphling::backend
