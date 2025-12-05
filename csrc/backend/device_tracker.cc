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

void DevicePartitionTracker::RecordPartitionProcessed(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    it->second->total_partitions_processed++;
  }
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

void DevicePartitionTracker::SetDeviceConnection(
    int64_t device_id, const uevent::ConnectionUeventPtr& conn) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (devices_map_.find(device_id) == devices_map_.end()) {
    LOG_WARN << "[DeviceTracker] Cannot set connection for unknown device "
             << device_id;
    return;
  }

  device_conn_[device_id] = conn;
  LOG_DEBUG << "[DeviceTracker] Set connection for device_id " << device_id;
}

uevent::ConnectionUeventPtr DevicePartitionTracker::GetDeviceConnection(
    int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = device_conn_.find(device_id);
  if (it != device_conn_.end()) {
    return it->second;
  }
  return nullptr;
}

void DevicePartitionTracker::RemoveDeviceConnection(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  device_conn_.erase(device_id);
  LOG_DEBUG << "[DeviceTracker] Removed connection for device_id " << device_id;
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
    oss << "    " << liveness->DebugString() << "\n";
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
  device_conn_.clear();
}

}  // namespace backend
}  // namespace morphling
