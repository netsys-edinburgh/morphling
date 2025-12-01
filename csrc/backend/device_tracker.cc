#include "device_tracker.h"

#include <sstream>

#include "base/logging.h"

namespace morphling {
namespace backend {

std::string DeviceLiveness::DebugString() const {
  std::ostringstream oss;
  
  // Calculate elapsed time since stats started
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - stats_start_time);
  double elapsed_seconds = elapsed.count() / 1000.0;
  
  // Calculate throughputs
  double upload_throughput = 0.0;
  double download_throughput = 0.0;
  double total_throughput = 0.0;
  
  if (elapsed_seconds > 0) {
    upload_throughput = total_bytes_sent / elapsed_seconds;
    download_throughput = total_bytes_received / elapsed_seconds;
    total_throughput = upload_throughput + download_throughput;
  }
  
  oss << "DeviceLiveness{device_id=" << device_id << ", addr=" << conn_addr
      << ", connected=" << is_connected
      << ", bytes_sent=" << total_bytes_sent
      << ", bytes_received=" << total_bytes_received
      << ", upload_throughput=" << upload_throughput << " B/s"
      << ", download_throughput=" << download_throughput << " B/s"
      << ", total_throughput=" << total_throughput << " B/s"
      << ", elapsed=" << elapsed_seconds << "s}";
  return oss.str();
}

DevicePartitionTracker& DevicePartitionTracker::GetInstance() {
  static DevicePartitionTracker instance;
  return instance;
}

DevicePartitionTracker::DevicePartitionTracker() : next_device_id_(0) {}

int64_t DevicePartitionTracker::AllocateDeviceId() { return next_device_id_++; }

int64_t DevicePartitionTracker::RegisterDevice(const std::string& conn_addr) {
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

  // Update mappings
  addr_to_device_id_[conn_addr] = device_id;
  device_id_to_addr_[device_id] = conn_addr;
  devices_map_[device_id] = liveness;

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
  addr_to_device_id_.erase(conn_addr);
  device_id_to_addr_.erase(device_id);
  devices_map_.erase(device_id);
}

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

void DevicePartitionTracker::RecordBytesSent(int64_t device_id,
                                             uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second->last_packet_time);
    
    // Calculate throughput for this packet: bytes / elapsed_time
    double elapsed_seconds = (elapsed.count() == 0) ? 0.001 : (elapsed.count() / 1000.0);
    it->second->last_packet_throughput = bytes / elapsed_seconds;
    
    it->second->total_bytes_sent += bytes;
    it->second->total_packets_sent++;
    it->second->last_packet_time = now;
    it->second->last_packet_size = bytes;
  }
}

void DevicePartitionTracker::RecordBytesReceived(int64_t device_id,
                                                 uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it != devices_map_.end()) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second->last_packet_time);
    
    // Calculate throughput for this packet: bytes / elapsed_time
    double elapsed_seconds = (elapsed.count() == 0) ? 0.001 : (elapsed.count() / 1000.0);
    it->second->last_packet_throughput = bytes / elapsed_seconds;
    
    it->second->total_bytes_received += bytes;
    it->second->total_packets_received++;
    it->second->last_packet_time = now;
    it->second->last_packet_size = bytes;
  }
}

double DevicePartitionTracker::GetUploadThroughput(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - it->second->stats_start_time);
  
  if (elapsed.count() == 0) {
    return 0.0;
  }

  // Calculate throughput in bytes per second
  double elapsed_seconds = elapsed.count() / 1000.0;
  double throughput = it->second->total_bytes_sent / elapsed_seconds;
  return throughput;
}

double DevicePartitionTracker::GetDownloadThroughput(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - it->second->stats_start_time);
  
  if (elapsed.count() == 0) {
    return 0.0;
  }

  // Calculate throughput in bytes per second
  double elapsed_seconds = elapsed.count() / 1000.0;
  double throughput = it->second->total_bytes_received / elapsed_seconds;
  return throughput;
}


double DevicePartitionTracker::GetLastPacketThroughput(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  return it->second->last_packet_throughput;
}

double DevicePartitionTracker::GetAveragePacketThroughput(int64_t device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devices_map_.find(device_id);
  if (it == devices_map_.end()) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - it->second->stats_start_time);
  
  if (elapsed.count() == 0) {
    return 0.0;
  }

  uint64_t total_packets = it->second->total_packets_sent + it->second->total_packets_received;
  if (total_packets == 0) {
    return 0.0;
  }

  // Calculate average throughput: total_bytes / elapsed_time
  // This is the same as overall throughput but normalized by packet count for understanding
  double elapsed_seconds = elapsed.count() / 1000.0;
  uint64_t total_bytes = it->second->total_bytes_sent + it->second->total_bytes_received;
  double throughput = total_bytes / elapsed_seconds;
  return throughput;
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
}

}  // namespace backend
}  // namespace morphling
