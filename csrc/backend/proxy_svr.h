#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/env_cfg.h"
#include "common/pytorch_defs.h"
#include "morphling.pb.h"
#include "network/uevent.h"
#include "network/ueventloop_thread.h"
#include "server_base.h"
#include "device_tracker.h"

struct BatchSendRecv {
  size_t sent;
  size_t recv;
  bool ignore;

  BatchSendRecv() : sent(0), recv(0), ignore(false) {}
  void Reset() {
    sent = 0;
    recv = 0;
    ignore = false;
  }
  void IncSent() { ++sent; }
  void IncRecv() { ++recv; }
  void SetIgnore() {
    ignore = true;
    ++sent;
  }
  bool Complete() { return sent == recv; }
  bool HasIgnore() { return ignore; }
};

// Partition tracking structure with OID (Operation ID) tracking
// device_id -> {partition_key, oid}
struct PartitionInfo {
  std::string key;
  int64_t oid;  // Operation ID to track which MatMul this partition belongs to
};

typedef std::unordered_map<int64_t, std::vector<PartitionInfo>> PartitionTrackerWithOid;

// In-flight partition structure - stores complete partition data for recovery
// When a device fails, we need to resend its in-flight partitions to a target device
struct InFlightPartition {
  MatrixPartitionPtr partition;    // The partition object with data
  int64_t original_device_id;      // Which device it was originally sent to
  std::string partition_key;       // Key for tracking
  int64_t oid;                     // Operation ID
};

typedef std::unordered_map<std::string, InFlightPartition> InFlightPartitionMap;

class ProxySvrHandle : public uevent::LoopHandle {
 public:
  ProxySvrHandle(ProxyEnvCfg& ctx, uevent::UeventLoop* loop);

  static void ThreadInit(uevent::UeventLoop* loop);
  static uevent::LoopHandle* CreateMyself(ProxyEnvCfg& ctx,
                                          uevent::UeventLoop* loop) {
    return new ProxySvrHandle(ctx, loop);
  }

 public:
  void RequestCb(const uevent::ConnectionUeventPtr& conn);
  void RequestWriteCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);
  void SendInLoop(const uevent::ConnectionUeventPtr& conn,
                  const MatrixPartitionPtr partition);

 private:
  void HandleMatMul(const void* payload, size_t size);

 private:
  ProxyEnvCfg& ctx_;
  uevent::UeventLoop* loop_;
  std::unordered_map<std::string, uint32_t> conn_inflight_;
  std::deque<std::function<void()>> task_queue_;
  
  // Handshake tracking: maps connection address to handshake state
  // true = handshake complete, false = waiting for device_id
  std::unordered_map<std::string, bool> conn_handshake_complete_;
  std::mutex handshake_mutex_;
};

class ProxySvrImpl : public std::enable_shared_from_this<ProxySvrImpl> {
 public:
  ProxySvrImpl(ProxyEnvCfg& context);
  void Initialize(uevent::UeventLoop* loop);

  void DispatchMatMulAsync(torch::Tensor& mat_a, torch::Tensor& mat_b);
  torch::Tensor WaitMatMul(int oid);
  torch::Tensor& GetOutputMatrix(int oid) { return outputs_[oid]; }

  void IncRspCbCount(int oid, size_t count);
  void DecRspCbCount(int oid, size_t count) { rsp_cb_counts_[oid] += count; }
  
  size_t GetConnectionCount() const { return conn_map_.size(); }

  // Partition tracking methods - now with OID tracking for device failure handling
  void AddPartitionToTracker(int64_t device_id, const std::string& partition_key, int64_t oid);
  void RemovePartitionFromTracker(int64_t device_id, const std::string& partition_key);
  void HandleDeviceFailure(int64_t failed_device_id, int64_t target_device_id);
  const PartitionTrackerWithOid& GetPartitionTracker() const { return partition_tracker_; }

  // Helper function to find a suitable target device for partition redistribution
  // Returns device_id of the target device, or -1 if no suitable device found
  int64_t FindTargetDeviceForFailure(int64_t failed_device_id);

  // Register device_id for a connection (called during handshake)
  void RegisterDeviceId(const std::string& conn_addr, int64_t device_id);

 private:
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);
  void RequestCb(const uevent::ConnectionUeventPtr& conn);
  void RequestWriteCb(const uevent::ConnectionUeventPtr& conn);

  // Rephrase partitions with optional excluded devices for retry scenarios
  void RephrasePartitions(
      std::vector<MatrixPartition>& partitions,
      const std::unordered_set<int64_t>& excluded_devices = {});

 private:
  ProxyEnvCfg& ctx_;
  std::shared_ptr<uevent::ListenerUevent> listener_;

  std::unordered_map<std::string, uevent::ConnectionUeventPtr> conn_map_;
  // Maps connection address to assigned device_id
  std::unordered_map<std::string, int64_t> conn_addr_to_device_id_;
  // Monotonically increasing device ID counter for stable, persistent device IDs
  // Each new device connection gets a unique ID that never changes
  int64_t next_device_id_{0};

  std::atomic_int mm_count_{0};
  std::vector<torch::Tensor> outputs_;
  std::vector<std::atomic_ullong> rsp_cb_counts_;
  std::vector<std::unordered_set<TensorKey>> device_tensors_;
  
  // Partition tracking: device_id -> [(partition_key, oid), ...]
  // Maps each device to the list of partitions assigned to it, with operation IDs
  // This allows us to unblock WaitMatMul when a device fails
  PartitionTrackerWithOid partition_tracker_;
  std::mutex partition_tracker_mutex_;  // Protects partition_tracker_

  // In-flight partitions: partition_key -> complete partition data
  // When a device fails, we need the complete partition data to resend to target device
  // This map stores: partition_key -> {partition object, tensor blocks, device_id, oid}
  InFlightPartitionMap inflight_partitions_;
  std::mutex inflight_partitions_mutex_;  // Protects inflight_partitions_
};

typedef std::shared_ptr<ProxySvrImpl> ProxySvrImplPtr;

class ProxySvr {
 public:
  class Status {
   public:
    enum StatusType {
      kOK = 0,
      kFatal,
      kUnknown,
    };

   public:
    Status() : status_(kOK), err_() {}
    const std::string& err() { return err_; }
    bool OK() { return status_ == kOK; }
    void SetError(StatusType status, const std::string& err) {
      status_ = status;
      if (status_str_.find(status) == status_str_.end()) status_ = kUnknown;
      err_ = status_str_.at(status_) + err_;
    }

   private:
    StatusType status_;
    std::string err_;
    static const std::map<StatusType, std::string> status_str_;
  };

 public:
  ProxySvr();
  void Initialize(const std::string& cfg_file);
  void Start();
  void SetCacheEnabled(bool enabled) { context_.enable_cli_cache = enabled ? 1 : 0; }
  void DispatchMatMulAsync(torch::Tensor& mat_a, torch::Tensor& mat_b) {
    svr_->DispatchMatMulAsync(mat_a, mat_b);
  }
  torch::Tensor WaitMatMul(int oid) { return svr_->WaitMatMul(oid); }
  size_t GetConnectionCount() const { return svr_->GetConnectionCount(); }

 private:
  ProxySvrImplPtr svr_;
  ProxyEnvCfg context_;
  std::shared_ptr<uevent::UeventLoopThread> loop_thread_;
};

