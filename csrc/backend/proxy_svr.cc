#include "proxy_svr.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/generator.h"
#include "common/stats.h"
#include "network/eventloop_libevent.h"
#include "network/listener_libevent.h"
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
      // Check if client is registered
      if (!conn_registered_[client_addr]) {
        LOG_ERROR << "Client " << client_addr
                  << " not registered, disconnecting";
        conn->ForceClose();
        return;
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
  auto& tracker = DEVICE_TRACKER;
  int64_t device_id = tracker.RegisterDevice(client_addr, profile);

  // Store device info and mark as registered
  device_info_[client_addr] = profile;
  conn_registered_[client_addr] = true;

  LOG_DEBUG << "Client " << client_addr
            << " registered with device_id=" << device_id << ": "
            << profile.DebugString();
}

void ProxySvrHandle::HandleMatMul(const ConnectionUeventPtr& conn,
                                  const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();

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

  // Remove partition from tracker when completed
  auto& tracker = DEVICE_TRACKER;
  tracker.RemovePartitionByKey(part_key);
  tracker.RecordPartitionProcessed(partition.dev_id);

  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->IncRspCbCount(partition.oid, 1);
}

void ProxySvrHandle::SendInLoop(const ConnectionUeventPtr& conn,
                                const MatrixPartitionPtr partition) {
  string client_addr = conn->GetPeerAddress().ToString();
  task_queue_.push_back([this, conn, partition, client_addr]() {
    // Use protobuf serialization instead of binary
    auto buffer = partition->Serialize();
    conn->SendData(buffer->GetBuffer(), buffer->GetSize());
    conn_inflight_[client_addr] += 1;
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

  // Mark as not registered initially
  conn_registered_[client_addr] = false;

  // Send registration request to client
  SendRegisterRequest(conn);
}

void ProxySvrHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected from " << client_addr;
  conn_inflight_.erase(client_addr);
  conn_registered_.erase(client_addr);
  device_info_.erase(client_addr);
}

/********************************ProxySvrImpl****************************************/

ProxySvrImpl::ProxySvrImpl(ProxyEnvCfg& ctx)
    : ctx_(ctx), listener_(nullptr), rsp_cb_counts_(5) {}

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

  // Start periodic partition health check (every 5 seconds)
  failed_partition_check_timer_ = loop->RunEvery(
      0.1, std::bind(&ProxySvrImpl::CheckFailedPartitions, this));
  LOG_INFO << "[ProxySvrImpl::Initialize] Started periodic partition health "
              "check (interval=5s)";
}

void ProxySvrImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);

  std::string conn_addr = conn->GetPeerAddress().ToString();
  auto& tracker = DEVICE_TRACKER;

  // Check if this is a reconnection from an existing device
  int64_t device_id = tracker.GetDeviceIdByAddr(conn_addr);

  if (device_id != -1) {
    // Reconnection: reuse the same device_id
    LOG_INFO << "[ConnectionSuccessCb] Reconnection from " << conn_addr
             << ", reusing device_id=" << device_id;
    tracker.MarkDeviceConnected(device_id, conn_addr);
  } else {
    // New connection: will be registered when we receive device profile
    LOG_INFO << "[ConnectionSuccessCb] New connection from " << conn_addr
             << ", awaiting device profile";
  }

  conn_map_[conn_addr] = conn;

  LOG_INFO << "[ConnectionSuccessCb] Total active connections: "
           << conn_map_.size();
  LOG_INFO << "[ConnectionSuccessCb] Connection map contents:";
  for (const auto& conn_pair : conn_map_) {
    LOG_INFO << "  - " << conn_pair.first << " -> "
             << (conn_pair.second ? "valid" : "null");
  }
}

void ProxySvrImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);

  std::string conn_addr = conn->GetPeerAddress().ToString();
  auto& tracker = DEVICE_TRACKER;

  // Find device ID by connection address
  int64_t device_id = tracker.GetDeviceIdByAddr(conn_addr);

  LOG_INFO << "[ConnectionClosedCb] Device " << device_id
           << " (addr: " << conn_addr << ") disconnected";

  // Step 1: Check if failed device has pending partitions
  bool has_pending_partitions = tracker.HasPendingPartitions(device_id);
  size_t pending_count = tracker.GetDevicePartitionCount(device_id);

  if (has_pending_partitions) {
    LOG_WARN << "[ConnectionClosedCb] Device " << device_id << " failed with "
             << pending_count << " pending partitions";
  }

  // Step 2: Mark device as disconnected
  tracker.MarkDeviceDisconnected(device_id);
  tracker.MarkPartitionsAsFailed(device_id);

  // Step 3: Remove connection from map
  conn_map_.erase(conn_addr);

  LOG_INFO << "[ConnectionClosedCb] Connection removed. Remaining connections: "
           << conn_map_.size();
  LOG_INFO << "[ConnectionClosedCb] Connection map contents:";
  for (const auto& conn_pair : conn_map_) {
    LOG_INFO << "  - " << conn_pair.first << " -> "
             << (conn_pair.second ? "valid" : "null");
  }

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
  LOG_INFO << "[DispatchMatMulAsync] Starting dispatch - mm_count=" << mm_count_
           << ", conn_map_.size()=" << conn_map_.size();

  outputs_[mm_count_].set_data(CreateOutputMatrix(mat_a, mat_b));
  auto partitions = PartitionMatrices(mat_a, mat_b, ctx_.block_size);
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  if (partitions.empty()) {
    LOG_ERROR << "[DispatchMatMulAsync] No partitions generated!";
    return;
  }

  auto cur_ver = partitions[0].version;
  LOG_INFO << "[" << cur_ver << "] Number of partitions: " << partitions.size()
           << " for A: " << a_shape << " and B: " << b_shape;

  LOG_INFO << "[DispatchMatMulAsync] Before random_shuffle - partitions.size()="
           << partitions.size();
  std::random_shuffle(partitions.begin(), partitions.end());

  LOG_INFO
      << "[DispatchMatMulAsync] Before RephrasePartitions - ctx_.num_device="
      << ctx_.num_device;
  LOG_INFO << "[DispatchMatMulAsync] Current connections:";
  for (const auto& conn_pair : conn_map_) {
    LOG_INFO << "  - " << conn_pair.first << " -> "
             << (conn_pair.second ? "valid" : "null");
  }

  RephrasePartitions(partitions);
  // make shared partitions
  std::vector<MatrixPartitionPtr> shared_partitions;
  for (auto& partition : partitions) {
    shared_partitions.push_back(std::make_shared<MatrixPartition>(partition));
  }

  LOG_INFO << "[DispatchMatMulAsync] After RephrasePartitions - created "
           << shared_partitions.size() << " shared partitions";

  auto start = std::chrono::high_resolution_clock::now();

  DecRspCbCount(mm_count_, partitions.size());

  LOG_INFO << "[DispatchMatMulAsync] Starting to dispatch partitions - "
              "conn_map_.size()="
           << conn_map_.size();

  auto& tracker = DEVICE_TRACKER;

  for (auto& partition : shared_partitions) {
    partition->oid = mm_count_;

    // Add partition to tracker with ownership
    tracker.AddPartition(partition->dev_id, partition->GetPartitionKey(),
                         mm_count_, partition);

    // Debug:
    LOG_INFO << "[DispatchMatMulAsync] partition key="
             << partition->GetPartitionKey() << ", dev_id=" << partition->dev_id
             << ", conn_map_.size()=" << conn_map_.size();
    if (conn_map_.empty()) {
      LOG_ERROR << "[DispatchMatMulAsync] conn_map_ is empty!";
      continue;
    }

    // Find connection by device_id through the tracker
    auto& tracker = DEVICE_TRACKER;
    std::string device_addr = tracker.GetDeviceAddr(partition->dev_id);

    if (device_addr.empty()) {
      LOG_ERROR << "[DispatchMatMulAsync] No address found for dev_id "
                << partition->dev_id;
      continue;
    }

    auto conn_it = conn_map_.find(device_addr);
    if (conn_it == conn_map_.end()) {
      LOG_ERROR << "[DispatchMatMulAsync] No connection found for dev_id "
                << partition->dev_id << " (addr: " << device_addr << ")";
      continue;
    }

    uevent::ConnectionUeventPtr target_conn = conn_it->second;

    LOG_DEBUG << "[DispatchMatMulAsync] Found connection for dev_id "
              << partition->dev_id;

    auto* loop = target_conn->GetLoop();
    if (!loop) {
      LOG_ERROR << "[DispatchMatMulAsync] loop is nullptr for dev_id "
                << partition->dev_id;
      continue;
    }
    auto* handle = reinterpret_cast<ProxySvrHandle*>(loop->GetLoopHandle());
    if (!handle) {
      LOG_ERROR << "[DispatchMatMulAsync] handle is nullptr for dev_id "
                << partition->dev_id;
      continue;
    }
    LOG_DEBUG << "[DispatchMatMulAsync] SendInLoop for dev_id="
              << partition->dev_id;
    loop->RunInLoop(
        bind(&ProxySvrHandle::SendInLoop, handle, target_conn, partition));
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_INFO << "Publish time: "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count()
           << "us, number of partitions: " << partitions.size();
  mm_count_++;
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
    std::vector<MatrixPartition>& partitions,
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

  std::vector<float> device_time(actual_num_devices, 0);
  device_tensors_.assign(actual_num_devices, std::unordered_set<TensorKey>());

  LOG_INFO << "[RephrasePartitions] Initialized device_time vector with size "
           << device_time.size() << ", connected devices: ";
  for (size_t i = 0; i < device_ids.size(); ++i) {
    LOG_INFO << "  [" << i << "] -> device_id=" << device_ids[i];
  }

  // greedy algorithm to select the minimal time
  for (size_t part_idx = 0; part_idx < partitions.size(); ++part_idx) {
    auto& partition = partitions[part_idx];
    LOG_DEBUG << "[RephrasePartitions] Processing partition " << part_idx << "/"
              << partitions.size();

    float min_time = std::numeric_limits<float>::max();
    int min_device_idx = 0;
    int64_t min_device_id = -1;
    auto version = partition.version;
    auto tensor_key_row = partition.GetRowKey();
    auto tensor_key_col = partition.GetColKey();
    bool min_r_cached = false;
    bool min_c_cached = false;

    LOG_INFO << "[RephrasePartitions] Processing partition " << part_idx
             << " - checking " << actual_num_devices << " devices";

    for (int i = 0; i < actual_num_devices; i++) {
      int64_t device_id = device_ids[i];

      // Skip excluded devices (used for retry scenarios)
      if (excluded_devices.find(device_id) != excluded_devices.end()) {
        LOG_DEBUG << "[RephrasePartitions] Skipping excluded device "
                  << device_id;
        continue;
      }

      auto& tensors = device_tensors_[i];

      bool r_cached = tensors.find(tensor_key_row) != tensors.end();
      bool c_cached = tensors.find(tensor_key_col) != tensors.end();

      auto r_size = std::get<1>(partition.mat[0]);
      auto c_size = std::get<1>(partition.mat[1]);
      auto cached_r_size = (r_cached) ? 0 : r_size;
      auto cached_c_size = (c_cached) ? 0 : c_size;

      int64_t num_rows = r_size / partition.h_dim / sizeof(float);
      int64_t num_cols = c_size / partition.h_dim / sizeof(float);

      float ul_time = (float)(num_rows * num_cols) * sizeof(float) / MB;
      float dl_time = (float)(cached_r_size + cached_c_size) / MB;
      float flops = (float)2.0 * num_rows * num_cols * partition.h_dim / TB;

      float time = std::max(std::max(ul_time, dl_time), flops) + device_time[i];
      if (time < min_time) {
        min_time = time;
        min_device_idx = i;
        min_device_id = device_id;
        min_r_cached = r_cached;
        min_c_cached = c_cached;
      }
    }

    LOG_INFO << "[RephrasePartitions] Partition " << part_idx
             << " assigned to device_id " << min_device_id
             << " (index=" << min_device_idx << ") with time " << min_time;

    if (min_device_id == -1 || min_time == std::numeric_limits<float>::max()) {
      LOG_ERROR << "[RephrasePartitions] Failed to find available device for "
                   "partition "
                << part_idx
                << ". excluded_devices.size()=" << excluded_devices.size()
                << ", actual_num_devices=" << actual_num_devices;
      assert(false && "No available device found for partition");
      continue;
    }

    // update the time for the device
    device_time[min_device_idx] = min_time;
    partition.dev_id = min_device_id;  // Use actual device_id, not index
    device_tensors_[min_device_idx].insert(tensor_key_row);
    device_tensors_[min_device_idx].insert(tensor_key_col);

    // Note: AddPartition will be called after shared_partitions are created
    // in DispatchMatMulAsync, where we have MatrixPartitionPtr available

    if (!ctx_.enable_cli_cache) continue;

    if (min_r_cached) {
      partition.mat[0] = {nullptr, 0};
    }
    if (min_c_cached) {
      partition.mat[1] = {nullptr, 0};
    }
  }

  LOG_INFO << "[RephrasePartitions] Completed partitioning";
  LOG_INFO << "Device time: " << device_time;
}

void ProxySvrImpl::HandleDeviceFailure(int64_t failed_device_id) {
  auto& tracker = DEVICE_TRACKER;

  // First, mark all partitions owned by this device as failed
  // This sets is_failed flag on the shared partition objects
  tracker.MarkPartitionsAsFailed(failed_device_id);

  // Prepare redistribution (collects partition info and OID counts)
  auto redistribution =
      tracker.PrepareDeviceFailureRedistribution(failed_device_id);

  if (redistribution.partitions.empty()) {
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " has no pending partitions to redistribute";
    return;
  }

  // Find best target device
  int64_t target_device_id = tracker.FindBestTargetDevice(failed_device_id);

  if (target_device_id == -1) {
    LOG_ERROR << "[HandleDeviceFailure] No suitable target device found for "
                 "redistribution";
    return;
  }

  redistribution.target_device_id = target_device_id;

  size_t num_failed_partitions = redistribution.partitions.size();
  LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
           << " failed with " << num_failed_partitions
           << " partitions. Redistributing to device " << target_device_id;

  LOG_INFO << "[HandleDeviceFailure] OID breakdown for failed partitions:";
  for (const auto& [oid, count] : redistribution.oid_counts) {
    LOG_INFO << "  - OID " << oid << ": " << count << " partitions";
  }

  // Apply redistribution (moves partition metadata to target device)
  tracker.ApplyFailureRedistribution(redistribution);

  // CRITICAL: Decrement response counters for partitions from failed device
  // These partitions were in-flight when the device failed, so they will never
  // produce responses. We must decrement their response counters to prevent
  // WaitMatMul from hanging forever.
  for (const auto& [oid, count] : redistribution.oid_counts) {
    LOG_INFO << "[HandleDeviceFailure] Decrementing response counter for OID "
             << oid << " by " << count << " (in-flight partitions lost)";
    for (size_t i = 0; i < count; ++i) {
      IncRspCbCount(oid, 1);  // Decrement the counter
    }
  }

  LOG_INFO << "[HandleDeviceFailure] Completed failure handling for device "
           << failed_device_id;
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
    if (tracker.HasPendingPartitions(device_id)) {
      size_t pending_count = tracker.GetDevicePartitionCount(device_id);
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
