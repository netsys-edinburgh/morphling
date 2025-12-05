#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "morphling.pb.h"
#include "partition_tracker.h"

// Convenient macro for accessing the DevicePartitionTracker singleton
#define DEVICE_TRACKER morphling::backend::DevicePartitionTracker::GetInstance()

namespace morphling {
namespace backend {
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
typedef std::shared_ptr<DeviceLiveness> DeviceLivenessPtr;

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
  //   void MarkDeviceConnected(int64_t device_id, const std::string&
  //   conn_addr); void MarkDeviceDisconnected(int64_t device_id);
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

  // Device liveness and ID management
  std::unordered_set<DeviceLivenessPtr> devices_set_;
  std::unordered_map<int64_t, DeviceLivenessPtr> devices_map_;

  int64_t next_device_id_;
  std::unordered_map<std::string, int64_t> addr_to_device_id_;
  std::unordered_map<int64_t, std::string> device_id_to_addr_;
};

}  // namespace backend
}  // namespace morphling
