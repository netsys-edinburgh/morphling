#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/pytorch_defs.h"
#include "morphling.pb.h"
#include "server_base.h"

// Convenient macro for accessing the DevicePartitionTracker singleton
#define DEVICE_TRACKER morphling::backend::DevicePartitionTracker::GetInstance()

namespace morphling {
namespace backend {

// Partition tracking structure with ownership and OID tracking
struct PartitionInfo {
  std::string key;
  int64_t oid;  // Operation ID to track which MatMul this partition belongs to
  int64_t owner_device_id;  // Device that owns this partition
  bool is_failed;           // True if this partition's owner device has failed
  MatrixPartitionPtr partition;  // Shared pointer to partition data

  PartitionInfo() : oid(-1), owner_device_id(-1), is_failed(false) {}
  PartitionInfo(const std::string& k, int64_t o, int64_t owner,
                MatrixPartitionPtr p)
      : key(k),
        oid(o),
        owner_device_id(owner),
        is_failed(false),
        partition(p) {}
};

// Device liveness information
struct DeviceLiveness {
  int64_t device_id;
  std::string conn_addr;
  bool is_connected;
  std::chrono::steady_clock::time_point last_seen;
  std::chrono::steady_clock::time_point connected_at;
  uint64_t total_partitions_processed;
  uint64_t total_bytes_sent;
  uint64_t total_bytes_received;
  DeviceProfileData profile;

  DeviceLiveness()
      : device_id(-1),
        is_connected(false),
        total_partitions_processed(0),
        total_bytes_sent(0),
        total_bytes_received(0) {}

  std::string DebugString() const;
};

// Device and partition tracker - manages device lifecycle and partition
// assignments Thread-safe singleton component for tracking:
// 1. Device liveness (connections, disconnections, health)
// 2. Partition assignments per device
// 3. Device metrics and statistics
class DevicePartitionTracker {
 public:
  static DevicePartitionTracker& GetInstance();

  // Disable copy and move
  DevicePartitionTracker(const DevicePartitionTracker&) = delete;
  DevicePartitionTracker& operator=(const DevicePartitionTracker&) = delete;
  DevicePartitionTracker(DevicePartitionTracker&&) = delete;
  DevicePartitionTracker& operator=(DevicePartitionTracker&&) = delete;

  // Device lifecycle management
  int64_t RegisterDevice(const std::string& conn_addr,
                         const DeviceProfileData& profile);
  void UnregisterDevice(int64_t device_id);
  void MarkDeviceConnected(int64_t device_id, const std::string& conn_addr);
  void MarkDeviceDisconnected(int64_t device_id);
  void UpdateDeviceLastSeen(int64_t device_id);

  // Device queries
  bool IsDeviceConnected(int64_t device_id) const;
  bool HasDevice(int64_t device_id) const;
  int64_t GetDeviceIdByAddr(const std::string& conn_addr) const;
  std::string GetDeviceAddr(int64_t device_id) const;
  DeviceLiveness GetDeviceLiveness(int64_t device_id) const;
  std::vector<int64_t> GetConnectedDevices() const;
  std::vector<int64_t> GetAllDevices() const;
  size_t GetConnectedDeviceCount() const;
  size_t GetTotalDeviceCount() const;

  // Partition tracking
  void AddPartition(int64_t device_id, const std::string& partition_key,
                    int64_t oid, MatrixPartitionPtr partition);
  void RemovePartition(int64_t device_id, const std::string& partition_key);
  void RemovePartitionByKey(const std::string& partition_key);

  // Mark all partitions owned by a device as failed (ownership removed)
  void MarkPartitionsAsFailed(int64_t device_id);
  std::vector<PartitionInfo> GetDevicePartitions(int64_t device_id) const;
  size_t GetDevicePartitionCount(int64_t device_id) const;
  bool HasPendingPartitions(int64_t device_id) const;

  // Partition redistribution on device failure
  struct FailureRedistribution {
    int64_t failed_device_id;
    int64_t target_device_id;
    std::vector<PartitionInfo> partitions;
    std::unordered_map<int64_t, size_t> oid_counts;  // oid -> count
  };
  FailureRedistribution PrepareDeviceFailureRedistribution(
      int64_t failed_device_id);
  void ApplyFailureRedistribution(const FailureRedistribution& redistribution);
  int64_t FindBestTargetDevice(
      int64_t failed_device_id,
      const std::unordered_set<int64_t>& excluded_devices = {}) const;

  // Statistics
  void RecordPartitionProcessed(int64_t device_id);
  void RecordBytesSent(int64_t device_id, uint64_t bytes);
  void RecordBytesReceived(int64_t device_id, uint64_t bytes);

  // Debug and monitoring
  std::string DebugString() const;
  void DumpState() const;

  // Clear all state (for testing)
  void Reset();

 private:
  DevicePartitionTracker();
  ~DevicePartitionTracker() = default;

  // Internal helpers
  int64_t AllocateDeviceId();

  // State
  mutable std::mutex mutex_;

  // Device ID management
  int64_t next_device_id_;
  std::unordered_map<std::string, int64_t> addr_to_device_id_;
  std::unordered_map<int64_t, std::string> device_id_to_addr_;

  // Device liveness
  std::unordered_map<int64_t, DeviceLiveness> devices_;

  // Partition tracking: device_id -> [PartitionInfo, ...]
  std::unordered_map<int64_t, std::vector<PartitionInfo>> partitions_;

  // Reverse index: partition_key -> device_id (for fast lookup)
  std::unordered_map<std::string, int64_t> partition_to_device_;
};

}  // namespace backend
}  // namespace morphling
