#include "device_tracker.h"

#include <algorithm>
#include <sstream>

#include "base/logging.h"

namespace morphling {
namespace backend {

std::string DeviceLiveness::DebugString() const {
  std::ostringstream oss;
  oss << "DeviceLiveness{device_id=" << device_id << ", addr=" << conn_addr
      << ", connected=" << is_connected
      << ", partitions_processed=" << total_partitions_processed
      << ", bytes_sent=" << total_bytes_sent
      << ", bytes_received=" << total_bytes_received << "}";
  return oss.str();
}

DevicePartitionTracker& DevicePartitionTracker::GetInstance() {
  static DevicePartitionTracker instance;
  return instance;
}

DevicePartitionTracker::DevicePartitionTracker() : next_device_id_(0) {}

int64_t DevicePartitionTracker::AllocateDeviceId() { return next_device_id_++; }

int64_t DevicePartitionTracker::RegisterDevice(
    const std::string& conn_addr, const DeviceProfileData& profile) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if device already exists (reconnection case)
  auto it = addr_to_device_id_.find(conn_addr);
  if (it != addr_to_device_id_.end()) {
    int64_t device_id = it->second;
    LOG_INFO << "[DeviceTracker] Reconnection from " << conn_addr
             << ", reusing device_id=" << device_id;

    // Update device state
    auto dev_it = devices_map_.find(device_id);
    if (dev_it != devices_map_.end()) {
      auto& device = dev_it->second;
      device->is_connected = true;
      device->last_seen = std::chrono::steady_clock::now();
      device->connected_at = std::chrono::steady_clock::now();
      device->conn_addr = conn_addr;
      device->profile = profile;
    }

    return device_id;
  }

  // New device - allocate ID
  int64_t device_id = AllocateDeviceId();
  LOG_INFO << "[DeviceTracker] New device registered: " << conn_addr
           << ", assigned device_id=" << device_id;

  // Create device liveness record
  auto liveness = std::make_shared<DeviceLiveness>();
  liveness->device_id = device_id;
  liveness->conn_addr = conn_addr;
  liveness->is_connected = true;
  liveness->last_seen = std::chrono::steady_clock::now();
  liveness->connected_at = std::chrono::steady_clock::now();
  liveness->profile = profile;

  // Update mappings
  addr_to_device_id_[conn_addr] = device_id;
  device_id_to_addr_[device_id] = conn_addr;
  devices_map_[device_id] = liveness;
  devices_set_.insert(liveness);

  return device_id;
}

void DevicePartitionTracker::UnregisterDevice(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot unregister unknown device "
             << device_id;
    return;
  }

  std::string conn_addr = it->second->conn_addr;
  LOG_INFO << "[DeviceTracker] Unregistering device " << device_id << " ("
           << conn_addr << ")";

  // Remove from all mappings
  devices_set_.erase(it->second);
  addr_to_device_id_.erase(conn_addr);
  device_id_to_addr_.erase(device_id);
  devices_map_.erase(device_id);
  device_partitions_.erase(device_id);

  // Clean up reverse partition index
  std::vector<std::string> keys_to_remove;
  for (const auto& [key, dev_id] : partition_to_device_) {
    if (dev_id == device_id) {
      keys_to_remove.push_back(key);
    }
  }
  for (const auto& key : keys_to_remove) {
    partition_to_device_.erase(key);
  }
}

// void DevicePartitionTracker::MarkDeviceConnected(int64_t device_id,
//                                                  const std::string&
//                                                  conn_addr) {
//   std::lock_guard<std::mutex> lock(mutex_);

//   auto it = devices_map_.find(device_id);
//   if (it != devices_map_.end()) {
//     it->second->is_connected = true;
//     it->second->conn_addr = conn_addr;
//     it->second->connected_at = std::chrono::steady_clock::now();
//     it->second->last_seen = std::chrono::steady_clock::now();
//     LOG_INFO << "[DeviceTracker] Device " << device_id << " marked
//     connected";
//   }
// }

// void DevicePartitionTracker::MarkDeviceDisconnected(int64_t device_id) {
//   std::lock_guard<std::mutex> lock(mutex_);

//   auto it = devices_map_.find(device_id);
//   if (it != devices_map_.end()) {
//     it->second->is_connected = false;
//     LOG_INFO << "[DeviceTracker] Device " << device_id
//              << " marked disconnected";
//   }
// }

void DevicePartitionTracker::UpdateDeviceLastSeen(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    it->second->last_seen = std::chrono::steady_clock::now();
  }
}

bool DevicePartitionTracker::IsDeviceConnected(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  return it != devices_map_.end() && it->second->is_connected;
}

bool DevicePartitionTracker::HasDevice(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_map_.find(device_id) != devices_map_.end();
}

int64_t DevicePartitionTracker::GetDeviceIdByAddr(
    const std::string& conn_addr) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = addr_to_device_id_.find(conn_addr);
  return (it != addr_to_device_id_.end()) ? it->second : -1;
}

std::string DevicePartitionTracker::GetDeviceAddr(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_id_to_addr_.find(device_id);
  return (it != device_id_to_addr_.end()) ? it->second : "";
}

DeviceLiveness DevicePartitionTracker::GetDeviceLiveness(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    return *it->second;
  }
  return DeviceLiveness();
}

std::vector<int64_t> DevicePartitionTracker::GetConnectedDevices() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int64_t> connected;
  for (const auto& [device_id, liveness] : devices_map_) {
    if (liveness->is_connected) {
      connected.push_back(device_id);
    }
  }
  return connected;
}

std::vector<int64_t> DevicePartitionTracker::GetAllDevices() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int64_t> all_devices;
  for (const auto& [device_id, _] : devices_map_) {
    all_devices.push_back(device_id);
  }
  return all_devices;
}

size_t DevicePartitionTracker::GetConnectedDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t count = 0;
  for (const auto& [_, liveness] : devices_map_) {
    if (liveness->is_connected) {
      count++;
    }
  }
  return count;
}

size_t DevicePartitionTracker::GetTotalDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_map_.size();
}

void DevicePartitionTracker::AddPartition(int64_t device_id,
                                          const std::string& partition_key,
                                          int64_t oid,
                                          MatrixPartitionPtr partition) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto info =
      std::make_shared<PartitionInfo>(partition_key, oid, device_id, partition);
  device_partitions_[device_id].push_back(info);
  partitions_set_.insert(info);
  partition_map_[partition_key] = info;
  partition_to_device_[partition_key] = device_id;

  LOG_DEBUG << "[DeviceTracker] Added partition " << partition_key
            << " (oid=" << oid << ") to device " << device_id
            << " with ownership";
}

void DevicePartitionTracker::RemovePartition(int64_t device_id,
                                             const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it != device_partitions_.end()) {
    auto& parts = it->second;
    auto part_it = std::find_if(parts.begin(), parts.end(),
                                [&partition_key](const PartitionInfoPtr& p) {
                                  return p->key == partition_key;
                                });

    if (part_it != parts.end()) {
      partitions_set_.erase(*part_it);
      parts.erase(part_it);
      partition_map_.erase(partition_key);
      partition_to_device_.erase(partition_key);
      LOG_DEBUG << "[DeviceTracker] Removed partition " << partition_key
                << " from device " << device_id;

      // Clean up empty partition list
      if (parts.empty()) {
        device_partitions_.erase(it);
      }
    }
  }
}

void DevicePartitionTracker::RemovePartitionByKey(
    const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Find device that has this partition
  auto rev_it = partition_to_device_.find(partition_key);
  if (rev_it == partition_to_device_.end()) {
    LOG_WARN << "[DeviceTracker] Partition " << partition_key
             << " not found in any device";
    return;
  }

  int64_t device_id = rev_it->second;
  partition_to_device_.erase(rev_it);

  // Remove from partition_map_
  auto map_it = partition_map_.find(partition_key);
  if (map_it != partition_map_.end()) {
    partitions_set_.erase(map_it->second);
    partition_map_.erase(map_it);
  }

  // Remove from device's partition list
  auto it = device_partitions_.find(device_id);
  if (it != device_partitions_.end()) {
    auto& parts = it->second;
    auto part_it = std::find_if(parts.begin(), parts.end(),
                                [&partition_key](const PartitionInfoPtr& p) {
                                  return p->key == partition_key;
                                });

    if (part_it != parts.end()) {
      parts.erase(part_it);
      LOG_DEBUG << "[DeviceTracker] Removed partition " << partition_key
                << " from device " << device_id << " (by key lookup)";

      if (parts.empty()) {
        device_partitions_.erase(it);
      }
    }
  }
}

void DevicePartitionTracker::MarkPartitionsAsFailed(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it == device_partitions_.end()) {
    LOG_DEBUG << "[DeviceTracker] No partitions found for device " << device_id;
    return;
  }

  size_t marked_count = 0;
  for (auto& part_info : it->second) {
    // Only mark RUNNING partitions as FAILED (not IDLE or already FINISHED)
    if (part_info->state == PartitionState::RUNNING) {
      part_info->is_failed = true;
      part_info->state = PartitionState::FAILED;
      // Clear ownership - partition is now orphaned
      part_info->owner_device_id = -1;
      marked_count++;
      LOG_DEBUG << "[DeviceTracker] Marked partition " << part_info->key
                << " as FAILED (oid=" << part_info->oid << ")";
    }
  }

  LOG_INFO << "[DeviceTracker] Marked " << marked_count
           << " RUNNING partitions as FAILED for device " << device_id;
}

void DevicePartitionTracker::MarkPartitionRunning(
    const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot mark partition " << partition_key
             << " as RUNNING: not found";
    return;
  }

  it->second->state = PartitionState::RUNNING;
  LOG_DEBUG << "[DeviceTracker] Partition " << partition_key
            << " marked as RUNNING";
}

void DevicePartitionTracker::MarkPartitionFinished(
    const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot mark partition " << partition_key
             << " as FINISHED: not found";
    return;
  }

  it->second->state = PartitionState::FINISHED;
  LOG_DEBUG << "[DeviceTracker] Partition " << partition_key
            << " marked as FINISHED";
}

void DevicePartitionTracker::MarkPartitionFailed(
    const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot mark partition " << partition_key
             << " as FAILED: not found";
    return;
  }

  it->second->state = PartitionState::FAILED;
  it->second->is_failed = true;
  LOG_DEBUG << "[DeviceTracker] Partition " << partition_key
            << " marked as FAILED";
}

void DevicePartitionTracker::MarkPartitionIdle(
    const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partition_map_.find(partition_key);
  if (it == partition_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot mark partition " << partition_key
             << " as IDLE: not found";
    return;
  }

  it->second->state = PartitionState::IDLE;
  it->second->is_failed = false;
  LOG_DEBUG << "[DeviceTracker] Partition " << partition_key
            << " marked as IDLE";
}

void DevicePartitionTracker::MarkDevicePartitionsRunning(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it == device_partitions_.end()) {
    return;
  }

  size_t marked_count = 0;
  for (auto& part : it->second) {
    if (part->state == PartitionState::IDLE) {
      part->state = PartitionState::RUNNING;
      marked_count++;
    }
  }

  LOG_DEBUG << "[DeviceTracker] Marked " << marked_count
            << " IDLE partitions as RUNNING for device " << device_id;
}

std::vector<PartitionInfoPtr> DevicePartitionTracker::GetDevicePartitions(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  if (it != device_partitions_.end()) {
    return it->second;
  }
  return {};
}

size_t DevicePartitionTracker::GetDevicePartitionCount(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  return (it != device_partitions_.end()) ? it->second.size() : 0;
}

bool DevicePartitionTracker::HasPendingPartitions(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_partitions_.find(device_id);
  return it != device_partitions_.end() && !it->second.empty();
}

void DevicePartitionTracker::RedistributeFailedDevicePartitions(
    int64_t failed_device_id, PartitionSchedulingPolicyPtr policy) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Get failed device's partitions
  auto it = device_partitions_.find(failed_device_id);
  if (it == device_partitions_.end() || it->second.empty()) {
    LOG_INFO << "[DeviceTracker] Device " << failed_device_id
             << " has no pending partitions to redistribute";
    return;
  }

  // Get all connected devices (excluding the failed one)
  std::vector<int64_t> available_devices;
  for (const auto& [device_id, liveness] : devices_map_) {
    if (device_id != failed_device_id && liveness->is_connected) {
      available_devices.push_back(device_id);
    }
  }

  if (available_devices.empty()) {
    LOG_ERROR << "[DeviceTracker] No available devices to redistribute "
                 "partitions from device "
              << failed_device_id;
    return;
  }

  // Count FAILED partitions (only redistribute those that were running when
  // device failed)
  std::vector<PartitionInfoPtr> partitions_to_redistribute;
  std::unordered_map<int64_t, size_t> oid_counts;
  for (const auto& part : it->second) {
    if (part->state == PartitionState::FAILED) {
      partitions_to_redistribute.push_back(part);
      oid_counts[part->oid]++;
    }
  }

  if (partitions_to_redistribute.empty()) {
    LOG_INFO << "[DeviceTracker] No FAILED partitions to redistribute "
                "from device "
             << failed_device_id;
    return;
  }

  LOG_INFO << "[DeviceTracker] Redistributing "
           << partitions_to_redistribute.size() << " partitions from device "
           << failed_device_id << " across " << available_devices.size()
           << " devices (" << oid_counts.size() << " OIDs)";

  // Use provided policy or default to round-robin
  if (!policy) {
    policy = std::make_shared<RoundRobinSchedulingPolicy>();
  }

  // Get redistribution mapping from policy
  auto redistribution_map = policy->RedistributePartitions(
      partitions_to_redistribute, available_devices);

  // Apply the redistribution
  for (auto& part : partitions_to_redistribute) {
    auto it = redistribution_map.find(part->key);
    if (it == redistribution_map.end()) {
      LOG_ERROR << "[DeviceTracker] No target device found for partition "
                << part->key;
      continue;
    }

    int64_t target_device_id = it->second;

    // Update ownership and reset state to IDLE
    part->owner_device_id = target_device_id;
    part->is_failed = false;
    part->state = PartitionState::IDLE;

    // Add to target device's partition list
    device_partitions_[target_device_id].push_back(part);

    // Update reverse index
    partition_to_device_[part->key] = target_device_id;

    LOG_DEBUG << "[DeviceTracker] Redistributed partition " << part->key
              << " (oid=" << part->oid << ") from device " << failed_device_id
              << " to device " << target_device_id;
  }

  // Remove failed device's partitions
  device_partitions_.erase(failed_device_id);

  LOG_INFO << "[DeviceTracker] Redistribution complete. Partitions "
              "distributed across "
           << available_devices.size() << " devices";
}

void DevicePartitionTracker::RecordPartitionProcessed(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    it->second->total_partitions_processed++;
  }
}

std::vector<PartitionInfoPtr> DevicePartitionTracker::GetIdlePartitions()
    const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<PartitionInfoPtr> idle_partitions;
  for (const auto& part_info : partitions_set_) {
    if (part_info->state == PartitionState::IDLE) {
      idle_partitions.push_back(part_info);
    }
  }
  return idle_partitions;
}

void DevicePartitionTracker::RecordBytesSent(int64_t device_id,
                                             uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    it->second->total_bytes_sent += bytes;
  }
}

void DevicePartitionTracker::RecordBytesReceived(int64_t device_id,
                                                 uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    it->second->total_bytes_received += bytes;
  }
}

std::string DevicePartitionTracker::DebugString() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::ostringstream oss;
  oss << "DevicePartitionTracker{\n";
  oss << "  Total devices: " << devices_map_.size() << "\n";
  oss << "  Connected devices: ";

  size_t connected = 0;
  for (const auto& [_, liveness] : devices_map_) {
    if (liveness->is_connected) connected++;
  }
  oss << connected << "\n";

  oss << "  Devices:\n";
  for (const auto& [device_id, liveness] : devices_map_) {
    oss << "    " << liveness->DebugString();
    auto it = device_partitions_.find(device_id);
    if (it != device_partitions_.end()) {
      oss << " [" << it->second.size() << " partitions]";
    }
    oss << "\n";
  }

  oss << "}";
  return oss.str();
}

void DevicePartitionTracker::DumpState() const {
  LOG_INFO << "[DeviceTracker] " << DebugString();
}

void DevicePartitionTracker::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG_INFO << "[DeviceTracker] Resetting all state";
  next_device_id_ = 0;
  addr_to_device_id_.clear();
  device_id_to_addr_.clear();
  devices_map_.clear();
  devices_set_.clear();
  device_partitions_.clear();
  partitions_set_.clear();
  partition_map_.clear();
  partition_to_device_.clear();
}

}  // namespace backend
}  // namespace morphling
