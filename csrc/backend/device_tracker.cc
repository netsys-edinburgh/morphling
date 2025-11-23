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
    auto& device = devices_[device_id];
    device.is_connected = true;
    device.last_seen = std::chrono::steady_clock::now();
    device.connected_at = std::chrono::steady_clock::now();
    device.conn_addr = conn_addr;
    device.profile = profile;

    return device_id;
  }

  // New device - allocate ID
  int64_t device_id = AllocateDeviceId();
  LOG_INFO << "[DeviceTracker] New device registered: " << conn_addr
           << ", assigned device_id=" << device_id;

  // Create device liveness record
  DeviceLiveness liveness;
  liveness.device_id = device_id;
  liveness.conn_addr = conn_addr;
  liveness.is_connected = true;
  liveness.last_seen = std::chrono::steady_clock::now();
  liveness.connected_at = std::chrono::steady_clock::now();
  liveness.profile = profile;

  // Update mappings
  addr_to_device_id_[conn_addr] = device_id;
  device_id_to_addr_[device_id] = conn_addr;
  devices_[device_id] = liveness;

  return device_id;
}

void DevicePartitionTracker::UnregisterDevice(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  if (it == devices_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot unregister unknown device "
             << device_id;
    return;
  }

  std::string conn_addr = it->second.conn_addr;
  LOG_INFO << "[DeviceTracker] Unregistering device " << device_id << " ("
           << conn_addr << ")";

  // Remove from all mappings
  addr_to_device_id_.erase(conn_addr);
  device_id_to_addr_.erase(device_id);
  devices_.erase(device_id);
  partitions_.erase(device_id);

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

void DevicePartitionTracker::MarkDeviceConnected(int64_t device_id,
                                                 const std::string& conn_addr) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    it->second.is_connected = true;
    it->second.conn_addr = conn_addr;
    it->second.connected_at = std::chrono::steady_clock::now();
    it->second.last_seen = std::chrono::steady_clock::now();
    LOG_INFO << "[DeviceTracker] Device " << device_id << " marked connected";
  }
}

void DevicePartitionTracker::MarkDeviceDisconnected(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    it->second.is_connected = false;
    LOG_INFO << "[DeviceTracker] Device " << device_id
             << " marked disconnected";
  }
}

void DevicePartitionTracker::UpdateDeviceLastSeen(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    it->second.last_seen = std::chrono::steady_clock::now();
  }
}

bool DevicePartitionTracker::IsDeviceConnected(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  return it != devices_.end() && it->second.is_connected;
}

bool DevicePartitionTracker::HasDevice(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_.find(device_id) != devices_.end();
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

  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    return it->second;
  }
  return DeviceLiveness();
}

std::vector<int64_t> DevicePartitionTracker::GetConnectedDevices() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int64_t> connected;
  for (const auto& [device_id, liveness] : devices_) {
    if (liveness.is_connected) {
      connected.push_back(device_id);
    }
  }
  return connected;
}

std::vector<int64_t> DevicePartitionTracker::GetAllDevices() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int64_t> all_devices;
  for (const auto& [device_id, _] : devices_) {
    all_devices.push_back(device_id);
  }
  return all_devices;
}

size_t DevicePartitionTracker::GetConnectedDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t count = 0;
  for (const auto& [_, liveness] : devices_) {
    if (liveness.is_connected) {
      count++;
    }
  }
  return count;
}

size_t DevicePartitionTracker::GetTotalDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_.size();
}

void DevicePartitionTracker::AddPartition(int64_t device_id,
                                          const std::string& partition_key,
                                          int64_t oid,
                                          MatrixPartitionPtr partition) {
  std::lock_guard<std::mutex> lock(mutex_);

  PartitionInfo info(partition_key, oid, device_id, partition);
  partitions_[device_id].push_back(info);
  partition_to_device_[partition_key] = device_id;

  LOG_DEBUG << "[DeviceTracker] Added partition " << partition_key
            << " (oid=" << oid << ") to device " << device_id
            << " with ownership";
}

void DevicePartitionTracker::RemovePartition(int64_t device_id,
                                             const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partitions_.find(device_id);
  if (it != partitions_.end()) {
    auto& parts = it->second;
    auto part_it = std::find_if(parts.begin(), parts.end(),
                                [&partition_key](const PartitionInfo& p) {
                                  return p.key == partition_key;
                                });

    if (part_it != parts.end()) {
      parts.erase(part_it);
      partition_to_device_.erase(partition_key);
      LOG_DEBUG << "[DeviceTracker] Removed partition " << partition_key
                << " from device " << device_id;

      // Clean up empty partition list
      if (parts.empty()) {
        partitions_.erase(it);
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

  // Remove from device's partition list
  auto it = partitions_.find(device_id);
  if (it != partitions_.end()) {
    auto& parts = it->second;
    auto part_it = std::find_if(parts.begin(), parts.end(),
                                [&partition_key](const PartitionInfo& p) {
                                  return p.key == partition_key;
                                });

    if (part_it != parts.end()) {
      parts.erase(part_it);
      LOG_DEBUG << "[DeviceTracker] Removed partition " << partition_key
                << " from device " << device_id << " (by key lookup)";

      if (parts.empty()) {
        partitions_.erase(it);
      }
    }
  }
}

void DevicePartitionTracker::MarkPartitionsAsFailed(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partitions_.find(device_id);
  if (it == partitions_.end()) {
    LOG_DEBUG << "[DeviceTracker] No partitions found for device " << device_id;
    return;
  }

  size_t marked_count = 0;
  for (auto& part_info : it->second) {
    if (!part_info.is_failed) {
      part_info.is_failed = true;
      // Clear ownership - partition is now orphaned
      part_info.owner_device_id = -1;
      marked_count++;
      LOG_DEBUG << "[DeviceTracker] Marked partition " << part_info.key
                << " as failed (oid=" << part_info.oid << ")";
    }
  }

  LOG_INFO << "[DeviceTracker] Marked " << marked_count
           << " partitions as failed for device " << device_id;
}

std::vector<PartitionInfo> DevicePartitionTracker::GetDevicePartitions(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partitions_.find(device_id);
  if (it != partitions_.end()) {
    return it->second;
  }
  return {};
}

size_t DevicePartitionTracker::GetDevicePartitionCount(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partitions_.find(device_id);
  return (it != partitions_.end()) ? it->second.size() : 0;
}

bool DevicePartitionTracker::HasPendingPartitions(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = partitions_.find(device_id);
  return it != partitions_.end() && !it->second.empty();
}

DevicePartitionTracker::FailureRedistribution
DevicePartitionTracker::PrepareDeviceFailureRedistribution(
    int64_t failed_device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  FailureRedistribution result;
  result.failed_device_id = failed_device_id;
  result.target_device_id = -1;

  // Get failed device's partitions
  auto it = partitions_.find(failed_device_id);
  if (it == partitions_.end() || it->second.empty()) {
    LOG_INFO << "[DeviceTracker] Device " << failed_device_id
             << " has no pending partitions to redistribute";
    return result;
  }

  // Only include partitions that are not already failed
  for (const auto& part : it->second) {
    if (!part.is_failed) {
      result.partitions.push_back(part);
      result.oid_counts[part.oid]++;
    }
  }

  LOG_INFO << "[DeviceTracker] Prepared redistribution for device "
           << failed_device_id << " with " << result.partitions.size()
           << " non-failed partitions across " << result.oid_counts.size()
           << " OIDs";

  return result;
}

void DevicePartitionTracker::ApplyFailureRedistribution(
    const FailureRedistribution& redistribution) {
  std::lock_guard<std::mutex> lock(mutex_);

  int64_t failed_id = redistribution.failed_device_id;
  int64_t target_id = redistribution.target_device_id;

  if (target_id == -1) {
    LOG_ERROR
        << "[DeviceTracker] Cannot apply redistribution: no target device";
    return;
  }

  LOG_INFO << "[DeviceTracker] Applying redistribution: moving "
           << redistribution.partitions.size() << " partitions from device "
           << failed_id << " to device " << target_id;

  // Move partitions to target device and update ownership
  auto& target_parts = partitions_[target_id];
  for (const auto& part : redistribution.partitions) {
    // Create new partition info with updated ownership
    PartitionInfo new_part = part;
    new_part.owner_device_id = target_id;
    new_part.is_failed = false;  // Reset failure flag

    target_parts.push_back(new_part);

    // Update reverse index
    partition_to_device_[part.key] = target_id;

    LOG_DEBUG << "[DeviceTracker] Redistributed partition " << part.key
              << " ownership from device " << failed_id << " to device "
              << target_id;
  }

  // Remove failed device's partitions
  partitions_.erase(failed_id);

  LOG_INFO << "[DeviceTracker] Redistribution complete. Target device "
           << target_id << " now has " << target_parts.size()
           << " total partitions";
}

int64_t DevicePartitionTracker::FindBestTargetDevice(
    int64_t failed_device_id,
    const std::unordered_set<int64_t>& excluded_devices) const {
  std::lock_guard<std::mutex> lock(mutex_);

  int64_t best_device = -1;
  size_t min_partitions = std::numeric_limits<size_t>::max();

  // Find connected device with minimum partitions
  for (const auto& [device_id, liveness] : devices_) {
    // Skip failed device, excluded devices, and disconnected devices
    if (device_id == failed_device_id || !liveness.is_connected ||
        excluded_devices.find(device_id) != excluded_devices.end()) {
      continue;
    }

    size_t partition_count = 0;
    auto it = partitions_.find(device_id);
    if (it != partitions_.end()) {
      partition_count = it->second.size();
    }

    if (partition_count < min_partitions) {
      min_partitions = partition_count;
      best_device = device_id;
    }
  }

  if (best_device != -1) {
    LOG_INFO << "[DeviceTracker] Selected device " << best_device
             << " as target for redistribution (has " << min_partitions
             << " partitions)";
  } else {
    LOG_ERROR << "[DeviceTracker] No suitable target device found for "
                 "redistribution";
  }

  return best_device;
}

void DevicePartitionTracker::RecordPartitionProcessed(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    it->second.total_partitions_processed++;
  }
}

void DevicePartitionTracker::RecordBytesSent(int64_t device_id,
                                             uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    it->second.total_bytes_sent += bytes;
  }
}

void DevicePartitionTracker::RecordBytesReceived(int64_t device_id,
                                                 uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    it->second.total_bytes_received += bytes;
  }
}

std::string DevicePartitionTracker::DebugString() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::ostringstream oss;
  oss << "DevicePartitionTracker{\n";
  oss << "  Total devices: " << devices_.size() << "\n";
  oss << "  Connected devices: ";

  size_t connected = 0;
  for (const auto& [_, liveness] : devices_) {
    if (liveness.is_connected) connected++;
  }
  oss << connected << "\n";

  oss << "  Devices:\n";
  for (const auto& [device_id, liveness] : devices_) {
    oss << "    " << liveness.DebugString();
    auto it = partitions_.find(device_id);
    if (it != partitions_.end()) {
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
  devices_.clear();
  partitions_.clear();
  partition_to_device_.clear();
}

}  // namespace backend
}  // namespace morphling
