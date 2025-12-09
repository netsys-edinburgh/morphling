#include "proxy_svr.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/generator.h"
#include "common/stats.h"
#include "device_tracker.h"
#include "network/eventloop_libevent.h"
#include "network/listener_libevent.h"
#include "partition_tracker.h"
#include "proto_base.h"
#include "server_base.h"
#include "utils/logging.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <iostream>
#include <set>

#include "base/logging.h"

namespace morphling {
namespace backend {

/*********************************ProxySvrHandle***********************************/

ProxySvrHandle::ProxySvrHandle(ProxyEnvCfg& ctx, UeventLoop* loop)
    : ctx_(ctx), loop_(loop) {
  SRV_STATS->Initialize();
  // Scheduling policy is now in ctx_.sched_policy
}

void ProxySvrHandle::ThreadInit(uevent::UeventLoop* loop) {
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  // handle->RegisterService();
}

void ProxySvrHandle::RequestWriteCb(const uevent::ConnectionUeventPtr& conn) {
  size_t readable = conn->ReadableLength();
  LOG_DEBUG << "RequestWriteCb readable: " << readable;
}

void ProxySvrHandle::SendRegisterRequest(const ConnectionUeventPtr& conn) {
  LOG_INFO << "Sending registration request to "
           << conn->GetPeerAddress().ToString();

  DeviceRegisterRequest request;
  auto buffer = request.Serialize();

  LOG_DEBUG << "Raw registration request data (hex): "
            << BinaryToHex(static_cast<const uint8_t*>(buffer->GetBuffer()),
                           buffer->GetSize());

  // Send the serialized request
  int ret = conn->SendData(buffer->GetBuffer(), buffer->GetSize());
  if (ret < 0) {
    LOG_ERROR << "Failed to send registration request";
    conn->ForceClose();
    return;
  }
}

void ProxySvrHandle::RequestCb(const ConnectionUeventPtr& conn) {
  while (true) {
    size_t readable = conn->ReadableLength();

    int ret = 0;
    uint32_t packsize = 0;
    ret = conn->ReceiveData(&packsize, sizeof(uint32_t));
    if (ret < 0) {
      LOG_ERROR << "ReceiveData packsize err";
      return;
    }
    packsize = ntohl(packsize);
    size_t datasize = packsize + sizeof(packsize);

    LOG_TRACE << "packsize: " << packsize << ", datasize: " << datasize
              << ", readable: " << readable;

    if (readable < datasize) {
      return;
    }

    std::unique_ptr<char[]> data(new char[datasize]);
    char* raw_data = data.get();
    ret = conn->ReceiveData(raw_data, datasize);
    if (ret < 0) {
      LOG_ERROR << "ReceiveData raw_data err";
      return;
    }

    ret = conn->DrainData(datasize);
    if (ret < 0) {
      LOG_ERROR << "DrainData err";
      return;
    }

    // Decode and dispatch message
    DecodeAndDispatch(conn, raw_data, datasize);
  }
}

void ProxySvrHandle::DecodeAndDispatch(const ConnectionUeventPtr& conn,
                                       const void* payload, size_t size) {
  // Step 1: Decode proto message header to get message type
  int32_t message_type = GetMessageType(payload, size);

  if (message_type < 0) {
    LOG_ERROR << "Failed to decode message type";
    return;
  }

  // Step 2: Dispatch to appropriate handler based on message type
  string client_addr = conn->GetPeerAddress().ToString();

  switch (message_type) {
    case morphling::global_api::DEVICE_PROFILE_DATA:
      HandleRegisterResponse(conn, payload, size);
      break;

    case morphling::global_api::COMPUTE_GEMM_DATA:
      // Check if client is connected via tracker
      {
        auto& tracker = DEVICE_TRACKER;
        int64_t device_id = tracker.GetDeviceIdByAddr(client_addr);
        if (device_id == -1 || !tracker.IsDeviceConnected(device_id)) {
          LOG_ERROR << "Client " << client_addr
                    << " not registered or not connected, disconnecting";
          conn->ForceClose();
          return;
        }
      }
      HandleMatMul(conn, payload, size);
      conn_inflight_[client_addr] -= 1;

      if (!task_queue_.empty()) {
        auto task = task_queue_.front();
        task_queue_.pop_front();
        task();
      }
      break;

    default:
      LOG_ERROR << "Unknown message type: " << message_type;
      break;
  }
}

void ProxySvrHandle::HandleRegisterResponse(const ConnectionUeventPtr& conn,
                                            const void* payload, size_t size) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_DEBUG << "Received device profile data from " << client_addr
            << ", size=" << size << ", hex: "
            << BinaryToHex(static_cast<const uint8_t*>(payload), size) << "";

  // Use standard Deserialize interface
  DeviceProfileData profile;
  profile.Deserialize(payload, size);

  // Register device in tracker
  int64_t device_id = DEVICE_TRACKER.RegisterDevice(client_addr, profile);

  // Store connection in tracker
  DEVICE_TRACKER.SetDeviceConnection(device_id, conn);

  // Store device info
  // device_info_[client_addr] = profile;

  LOG_DEBUG << "Client " << client_addr
            << " registered with device_id=" << device_id << ": "
            << profile.DebugString();

  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, this));
}

void ProxySvrHandle::HandleMatMul(const ConnectionUeventPtr& conn,
                                  const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  
  // Record RECEIVE start time (virtual time)
  uint64_t vt_receive_start = VirtualClockNow();

  // Use standard Deserialize interface
  MatrixPartition partition;
  partition.Deserialize(payload, size);

  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << part_key << " RSP Deserialization time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  // Record bytes received (download request from device)
  DEVICE_TRACKER.RecordBytesReceived(partition.dev_id, size);
  
  // Log download throughput after receiving response
  double download_tp = DEVICE_TRACKER.GetDownloadThroughput(partition.dev_id);
  double last_packet_tp = DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp = DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();
  
  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us, end_us);
  
  LOG_INFO << "[HandleMatMul] Device " << partition.dev_id 
           << " - Received: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Download TP: " << download_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";
  
  // Log throughput to file
  DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, "DOWNLOAD",
                                     size, download_tp, start_us, end_us);

  auto [o_ptr, o_size] = partition.mat[0];
  int64_t row_size = o_size / partition.h_dim / sizeof(float);
  int64_t col_size = partition.h_dim;

  uint64_t ul_overhead = CurrentTimeMicros() - partition.timestamp;

  LOG_DEBUG << part_key << " partition: " << partition.DebugString();

  start = std::chrono::high_resolution_clock::now();
  auto output = torch::from_blob(o_ptr, {row_size, col_size},
                                 FLOAT32_TENSOR_OPTIONS(torch::kCPU));
  {
    // std::lock_guard<std::mutex> lock(outputs_mutex_[partition.oid]);
    auto& output_matrix = reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
                              ->GetOutputMatrix(partition.oid);
    IndexPutMatrixBlock(output_matrix, output, partition.row, partition.col,
                        partition.pivot, ctx_.block_size);
  }
  end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << "UpdateMatrixBlock time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  // Record RECEIVE end time (virtual time)
  uint64_t vt_receive_end = VirtualClockNow();
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, "RECEIVE", "END",
                                     vt_receive_start, vt_receive_end);

  // Mark partition as FINISHED and remove from tracker
  PARTITION_TRACKER.MarkPartitionFinished(part_key);
  PARTITION_TRACKER.RemovePartitionByKey(part_key);
  DEVICE_TRACKER.RecordPartitionProcessed(partition.dev_id);

  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->IncRspCbCount(partition.oid, 1);
}

void ProxySvrHandle::SendInLoop(const ConnectionUeventPtr& conn,
                                const MatrixPartitionPtr partition) {
  // check connection valid first
  if (conn->IsClosed()) {
    LOG_ERROR << "Connection to " << conn->GetPeerAddress().ToString()
              << " is not valid. Cannot send partition.";
    return;
  }

  string client_addr = conn->GetPeerAddress().ToString();
  
  task_queue_.push_back([this, conn, partition, client_addr]() {
    // Record SEND start time (virtual time)
    uint64_t vt_send_start = VirtualClockNow();
    
    // Use protobuf serialization instead of binary
    auto buffer = partition->Serialize();
    conn->SendData(buffer->GetBuffer(), buffer->GetSize());
    conn_inflight_[client_addr] += 1;
    
    // Record SEND end time (virtual time)
    uint64_t vt_send_end = VirtualClockNow();
    DEVICE_TRACKER.LogVirtualTimeEvent(partition->dev_id, "SEND", "END",
                                       vt_send_start, vt_send_end);
  });

  if (conn_inflight_[client_addr] >= ctx_.max_inflight) {
    return;
  }

  auto task = task_queue_.front();
  task_queue_.pop_front();
  task();
}

void ProxySvrHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected from " << client_addr;
  conn_inflight_[client_addr] = 0;

  // Send registration request to client
  SendRegisterRequest(conn);
}

void ProxySvrHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected from " << client_addr;
  conn_inflight_.erase(client_addr);
  // device_info_.erase(client_addr);

  std::string conn_addr = conn->GetPeerAddress().ToString();

  // Find device ID by connection address
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(conn_addr);

  LOG_INFO << "[ConnectionClosedCb] Device " << device_id
           << " (addr: " << conn_addr << ") disconnected";

  // Step 1: Check if failed device has pending partitions
  bool has_pending_partitions =
      PARTITION_TRACKER.HasPendingPartitions(device_id);
  size_t pending_count = PARTITION_TRACKER.GetDevicePartitionCount(device_id);

  if (has_pending_partitions) {
    LOG_WARN << "[ConnectionClosedCb] Device " << device_id << " failed with "
             << pending_count << " pending partitions";
  }

  // Step 2: Mark all running partitions as failed
  PARTITION_TRACKER.MarkDevicePartitionsFailed(device_id);

  // Step 3: Remove connection from maps
  // conn_map_.erase(conn_addr);
  if (device_id != -1) {
    DEVICE_TRACKER.RemoveDeviceConnection(device_id);
    DEVICE_TRACKER.UnregisterDevice(device_id);
  }
}

/********************************ProxySvrImpl****************************************/

ProxySvrImpl::ProxySvrImpl(ProxyEnvCfg& ctx)
    : ctx_(ctx), listener_(nullptr), rsp_cb_counts_(5) {
  // Initialize with greedy scheduling policy by default
}

ProxySvrImpl::~ProxySvrImpl() {
  LOG_INFO << "[ProxySvrImpl::~ProxySvrImpl] Shutting down ProxySvrImpl";
  loop_->CancelTimer(failed_partition_check_timer_);
}

void ProxySvrImpl::Initialize(UeventLoop* loop) {
  LOG_INFO << "[ProxySvrImpl::Initialize] Starting server initialization";
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - listen_ip=" << ctx_.listen_ip
           << ", listen_port=" << ctx_.listen_port;
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - num_device="
           << ctx_.num_device << ", thread=" << ctx_.thread;

  // Initialize virtual clock
  base::VirtualClock::instance().Initialize();
  LOG_INFO << "[ProxySvrImpl::Initialize] Virtual clock initialized";

  // Initialize performance logging
  DEVICE_TRACKER.InitPerfLog("./perf.log");
  LOG_INFO << "[ProxySvrImpl::Initialize] Performance logging initialized at ./perf.log";

  loop_ = loop;

  auto create_handle_cb = bind(ProxySvrHandle::CreateMyself, ref(ctx_), _1);
  UsockAddress addr(ctx_.listen_ip, ctx_.listen_port);
  listener_ =
      make_shared<ListenerLibevent>(loop, addr, "proxy_listener", Option());
  listener_->SetCreateLoopHandleCb(create_handle_cb);
  listener_->SetThreadInitCb(ProxySvrHandle::ThreadInit);
  listener_->SetConnectionSuccessCb(
      bind(&ProxySvrImpl::ConnectionSuccessCb, shared_from_this(), _1));
  listener_->SetMessageReadCb(
      bind(&ProxySvrImpl::RequestCb, shared_from_this(), _1));
  listener_->SetConnectionClosedCb(
      bind(&ProxySvrImpl::ConnectionClosedCb, shared_from_this(), _1));
  listener_->SetMessageWriteCb([](const ConnectionUeventPtr& conn) {});
  listener_->SetThreadNum(ctx_.thread);
  listener_->StartPrimaryLoop();

  LOG_INFO << "[ProxySvrImpl::Initialize] ProxySvrImpl listen on:"
           << ctx_.listen_ip << ":" << ctx_.listen_port;

  ctx_.instance = this;

  // Start();
  // InitLogger();

  // no more than 5 MAtMul in parallel
  outputs_ = std::move(std::vector<torch::Tensor>(5));
  // rsp_cb_counts_ = std::move(std::vector<std::atomic_ullong>(5));
  // outputs_mutex_ = std::move(std::vector<std::mutex>(5));
  for (int i = 0; i < 5; i++) {
    outputs_[i] = torch::empty({0, 0});
    rsp_cb_counts_[i] = 0;
  }

  LOG_INFO << "[ProxySvrImpl::Initialize] Server initialization completed. "
              "Waiting for connections...";

  // Start periodic partition health check (every 0.1 seconds)
  // failed_partition_check_timer_ = loop->RunEvery(
  //     0.1, std::bind(&ProxySvrImpl::CheckFailedPartitions, this));
  // idle_partition_redistribute_timer_ =
  //     loop->RunEvery(0.5, std::bind(&ProxySvrImpl::SendIdlePartitions,
  //     this));
  LOG_INFO << "[ProxySvrImpl::Initialize] Started periodic partition health "
              "check (interval=0.1s)";
}

void ProxySvrImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);
  // loop->RunInLoop(bind(&ProxySvrHandle::ConnectionSuccessCb, handle, conn));
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));
}

void ProxySvrImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));

  // Unregister device from tracker
  string client_addr = conn->GetPeerAddress().ToString();
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(client_addr);
  if (device_id != -1) {
    DEVICE_TRACKER.UnregisterDevice(device_id);
  }

  // loop->RunInLoop(bind(&ProxySvrHandle::ConnectionClosedCb, handle, conn));

  // LOG_INFO << "[ConnectionClosedCb] Connection removed. Remaining
  // connections: "
  //          << conn_map_.size();
  // LOG_INFO << "[ConnectionClosedCb] Connection map contents:";
  // for (const auto& conn_pair : conn_map_) {
  //   LOG_INFO << "  - " << conn_pair.first << " -> "
  //            << (conn_pair.second ? "valid" : "null");
  // }

  // // Step 4: Handle partition redistribution if needed
  // if (has_pending_partitions && conn_map_.size() > 0) {
  //   LOG_INFO << "[ConnectionClosedCb] Starting partition redistribution for "
  //               "failed device "
  //            << device_id;
  //   HandleDeviceFailure(device_id);
  //   LOG_INFO << "[ConnectionClosedCb] Partition redistribution completed";
  // } else if (has_pending_partitions && conn_map_.empty()) {
  //   LOG_ERROR << "[ConnectionClosedCb] Device " << device_id << " failed with
  //   "
  //             << pending_count
  //             << " pending partitions but no other devices available!";
  // }
}

void ProxySvrImpl::RequestWriteCb(const uevent::ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->RequestWriteCb(conn);
}

void ProxySvrImpl::RequestCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  loop->RunInLoop(bind(&ProxySvrHandle::RequestCb, handle, conn));
}

// void ProxySvrImpl::RequestCb(const uevent::ConnectionUeventPtr& conn) {
//   size_t readable = conn->ReadableLength();

//   int ret = 0;
//   size_t packsize;
//   ret = conn->ReceiveData(&packsize, sizeof(size_t));
//   if (ret < 0) {
//     conn->ForceClose();
//     LOG_ERROR << "ReceiveData on a closed connection";
//     return;
//   }

//   auto* loop = conn->GetLoop();
//   loop->AssertInLoopThread();
//   auto* loop_handle = loop->GetLoopHandle();
//   auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
//   handle->RequestCb(conn);
// }

void ProxySvrImpl::DispatchMatMulAsync(torch::Tensor& mat_a,
                                       torch::Tensor& mat_b) {
  LOG_INFO << "[DispatchMatMulAsync] Starting dispatch - mm_count="
           << mm_count_;

  outputs_[mm_count_].set_data(CreateOutputMatrix(mat_a, mat_b));
  auto partitions = PartitionMatrices(mat_a, mat_b, ctx_.block_size);
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  if (partitions.empty()) {
    LOG_ERROR << "[DispatchMatMulAsync] No partitions generated!";
    return;
  }

  auto cur_ver = partitions[0]->version;
  LOG_INFO << "[" << cur_ver << "] Number of partitions: " << partitions.size()
           << " for A: " << a_shape << " and B: " << b_shape;

  LOG_INFO << "[DispatchMatMulAsync] Before random_shuffle - partitions.size()="
           << partitions.size();
  std::random_shuffle(partitions.begin(), partitions.end());

  // LOG_INFO
  //     << "[DispatchMatMulAsync] Before RephrasePartitions - ctx_.num_device="
  //     << ctx_.num_device;

  // RephrasePartitions(partitions);

  // LOG_INFO << "[DispatchMatMulAsync] After RephrasePartitions - created "
  //          << partitions.size() << " partitions";
  auto start = std::chrono::high_resolution_clock::now();

  DecRspCbCount(mm_count_, partitions.size());

  LOG_INFO << "[DispatchMatMulAsync] Creating " << partitions.size()
           << " partitions as IDLE";

  // Add all partitions to tracker as IDLE - they will be dispatched by
  // SendIdlePartitions
  for (auto& partition : partitions) {
    partition->oid = mm_count_;

    // Add partition to tracker with ownership (automatically marked as IDLE)
    PARTITION_TRACKER.AddPartition(
        partition->dev_id, partition->GetPartitionKey(), mm_count_, partition);

    LOG_DEBUG << "[DispatchMatMulAsync] Created IDLE partition key="
              << partition->GetPartitionKey()
              << ", dev_id=" << partition->dev_id << ", oid=" << mm_count_;
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_INFO << "[DispatchMatMulAsync] Created " << partitions.size()
           << " IDLE partitions in "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count()
           << "us. Partitions will be sent by SendIdlePartitions timer.";
  mm_count_++;

  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_->GetLoopHandle());
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));
}

torch::Tensor ProxySvrImpl::WaitMatMul(int oid) {
  auto start = std::chrono::high_resolution_clock::now();
  LOG_INFO << "[WaitMatMul] Starting wait for oid=" << oid
           << ", rsp_cb_counts_[oid]=" << rsp_cb_counts_[oid];

  int poll_count = 0;
  while (rsp_cb_counts_[oid] > 0) {
    poll_count++;
    if (poll_count % 50 == 0) {  // Log every 5 seconds (50 * 100ms)
      LOG_WARN << "[WaitMatMul] Still waiting for oid=" << oid
               << ", rsp_cb_counts_[oid]=" << rsp_cb_counts_[oid]
               << ", poll_count=" << poll_count * 100 << "ms";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto shape = outputs_[oid].sizes().vec();
  auto wait_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  LOG_INFO << "[WaitMatMul] Completed for oid=" << oid
           << ", wait_time=" << wait_time << "us, shape=" << shape;

  mm_count_--;
  return outputs_[oid];
}

void ProxySvrImpl::IncRspCbCount(int oid, size_t count) {
  int prev = rsp_cb_counts_[oid];
  rsp_cb_counts_[oid] -= count;
  LOG_DEBUG << "[IncRspCbCount] oid=" << oid << ", count=" << count
            << ", prev=" << prev << ", now=" << rsp_cb_counts_[oid];
}

void ProxySvrImpl::RephrasePartitions(
    std::vector<MatrixPartitionPtr>& partitions,
    const std::unordered_set<int64_t>& excluded_devices) {
  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> device_ids = tracker.GetConnectedDevices();

  int actual_num_devices = static_cast<int>(device_ids.size());
  LOG_INFO << "[RephrasePartitions] Starting with " << partitions.size()
           << " partitions, actual connected devices=" << actual_num_devices
           << ", excluded_devices=" << excluded_devices.size();

  if (actual_num_devices == 0) {
    LOG_ERROR << "[RephrasePartitions] No devices connected!";
    return;
  }

  std::sort(device_ids.begin(), device_ids.end());

  auto assignments = ctx_.sched_policy->AssignPartitionsToDevices(
      partitions, excluded_devices);

  if (assignments.size() != partitions.size()) {
    LOG_ERROR << "[RephrasePartitions] Policy returned " << assignments.size()
              << " assignments for " << partitions.size() << " partitions";
    return;
  }

  // Apply assignments to partitions
  for (size_t i = 0; i < partitions.size(); ++i) {
    partitions[i]->dev_id = assignments[i];
    LOG_DEBUG << "[RephrasePartitions] Partition " << i
              << " assigned to device_id " << assignments[i];
  }

  LOG_INFO << "[RephrasePartitions] Completed partitioning";
}

void ProxySvrImpl::HandleDeviceFailure(int64_t failed_device_id) {
  // Get partitions before redistribution to count OIDs
  auto failed_partitions =
      PARTITION_TRACKER.GetDevicePartitions(failed_device_id);

  if (failed_partitions.empty()) {
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " has no pending partitions to redistribute";
    return;
  }

  // Count FAILED partitions and OIDs (only those that were RUNNING)
  size_t num_failed_partitions = 0;
  std::unordered_map<int64_t, size_t> oid_counts;
  for (const auto& part : failed_partitions) {
    if (part->state == PartitionState::IDLE) {
      num_failed_partitions++;
      oid_counts[part->oid]++;
    }
  }

  if (num_failed_partitions == 0) {
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " has no FAILED partitions to redistribute";
    return;
  }

  LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
           << " failed with " << num_failed_partitions
           << " partitions. Redistributing across all connected devices";

  LOG_INFO << "[HandleDeviceFailure] OID breakdown for failed partitions:";
  for (const auto& [oid, count] : oid_counts) {
    LOG_INFO << "  - OID " << oid << ": " << count << " partitions";
  }

  // Redistribute partitions across all connected devices
  // PARTITION_TRACKER.RedistributeFailedDevicePartitions(failed_device_id);

  // Get connected devices to send redistributed partitions to them
  std::vector<int64_t> connected_devices = DEVICE_TRACKER.GetConnectedDevices();

  // CRITICAL: Decrement response counters for partitions from failed device
  // These partitions were in-flight when the device failed, so they will never
  // produce responses. We must decrement their response counters to prevent
  // WaitMatMul from hanging forever.
  for (const auto& [oid, count] : oid_counts) {
    LOG_INFO << "[HandleDeviceFailure] Decrementing response counter for OID "
             << oid << " by " << count << " (in-flight partitions lost)";
    for (size_t i = 0; i < count; ++i) {
      IncRspCbCount(oid, 1);  // Decrement the counter
    }
  }

  LOG_INFO << "[HandleDeviceFailure] Completed failure handling for device "
           << failed_device_id;
}

void ProxySvrHandle::SendIdlePartitions() {
  auto idle_partitions = PARTITION_TRACKER.GetIdlePartitions();

  if (idle_partitions.empty()) {
    LOG_DEBUG << "[SendIdlePartitions] No IDLE partitions to send";
    return;
  }

  auto redistributed =
      ctx_.sched_policy->RedistributePartitions(idle_partitions);

  if (redistributed.empty()) {
    LOG_DEBUG << "[SendIdlePartitions] No available devices for redistribution";
    return;
  }

  LOG_INFO << "[SendIdlePartitions] Sending " << idle_partitions.size()
           << " IDLE partitions to devices";

  for (const auto& part_info : idle_partitions) {
    // Mark partition as RUNNING before sent
    PARTITION_TRACKER.MarkPartitionRunning(
        part_info->partition->GetPartitionKey());
  }

  // Send each IDLE partition
  for (const auto& part_info : idle_partitions) {
    // Update partition's dev_id to match the assigned device
    part_info->partition->dev_id = part_info->owner_device_id;
    
    LOG_DEBUG << "[SendIdlePartitions] Sending partition " << part_info->key
              << " (oid=" << part_info->oid << ") to device "
              << part_info->owner_device_id;
    // if (part_info->state != PartitionState::RUNNING) {
    //   LOG_ERROR << "[SendIdlePartitions] Partition " << part_info->key
    //             << " not in RUNNING state before sending!";
    //   continue;
    // }
    auto target_conn =
        DEVICE_TRACKER.GetDeviceConnection(part_info->owner_device_id);
    if (!target_conn) {
      LOG_ERROR << "[SendIdlePartitions] No connection for device "
                << part_info->owner_device_id;
      continue;
    }
    auto* loop = target_conn->GetLoop();
    auto* handle = reinterpret_cast<ProxySvrHandle*>(loop->GetLoopHandle());
    loop->RunInLoop(bind(&ProxySvrHandle::SendInLoop, handle, target_conn,
                         part_info->partition));
  }

  LOG_INFO << "[SendIdlePartitions] Completed sending "
           << idle_partitions.size() << " partitions to devices";
}

void ProxySvrImpl::CheckFailedPartitions() {
  auto& tracker = DEVICE_TRACKER;

  // Get all devices
  std::vector<int64_t> all_devices = tracker.GetAllDevices();

  LOG_DEBUG << "[CheckFailedPartitions] Checking " << all_devices.size()
            << " devices for failed partitions";

  for (int64_t device_id : all_devices) {
    // Skip connected devices
    if (tracker.IsDeviceConnected(device_id)) {
      continue;
    }

    // Check if disconnected device has pending partitions
    if (PARTITION_TRACKER.HasPendingPartitions(device_id)) {
      size_t pending_count =
          PARTITION_TRACKER.GetDevicePartitionCount(device_id);
      LOG_WARN << "[CheckFailedPartitions] Detected disconnected device "
               << device_id << " with " << pending_count
               << " pending partitions. Initiating failure handling.";

      HandleDeviceFailure(device_id);
    }
  }
}

/*********************************ProxySvr***************************************/
typedef ProxySvr::Status ProxyStatus;
typedef ProxyStatus::StatusType ProxyStatusType;
const map<ProxyStatusType, string> ProxyStatus::status_str_ = {
    {kOK, ""},
    {kFatal, "Proxy svr fatal:"},
    {kUnknown, "Proxy svr unknown:"},
};

ProxySvr::ProxySvr() : svr_(nullptr), loop_thread_(nullptr) {}

void ProxySvr::Initialize(const std::string& cfg_file) {
  context_.Initialize(cfg_file);
  svr_ = make_shared<ProxySvrImpl>(context_);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxySvrHandle::CreateMyself, ref(context_), _1),
      bind(&ProxySvrImpl::Initialize, svr_, _1), "Proxy svr main thread");
}

void ProxySvr::Start() { loop_thread_->StartLoop(); }

}  // namespace backend
}  // namespace morphling
