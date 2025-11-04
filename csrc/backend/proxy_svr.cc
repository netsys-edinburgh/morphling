#include "proxy_svr.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/generator.h"
#include "common/stats.h"
#include "network/eventloop_libevent.h"
#include "network/listener_libevent.h"
#include "utils/logging.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <iostream>
#include <set>

#include "base/logging.h"

/*********************************ProxySvrHandle************************************/

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

void ProxySvrHandle::RequestCb(const ConnectionUeventPtr& conn) {
  while (true) {
    size_t readable = conn->ReadableLength();

    int ret = 0;
    uint32_t packsize;
    ret = conn->ReceiveData(&packsize, sizeof(uint32_t));
    if (ret < 0) {
      LOG_ERROR << "ReceiveData packsize err";
      return;
    }
    packsize = ntohl(packsize);
    size_t datasize = packsize + sizeof(packsize);

    LOG_TRACE  << "packsize: " << packsize << ", datasize: " << datasize
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

    HandleMatMul(raw_data, datasize);
    conn_inflight_[conn->GetPeerAddress().ToString()] -= 1;

    if (!task_queue_.empty()) {
      auto task = task_queue_.front();
      task_queue_.pop_front();
      task();
    }
  }
}

void ProxySvrHandle::HandleMatMul(const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  // Use protobuf deserialization instead of binary
  partition.DeserializeFromProto(payload, size);
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

  // Remove partition from tracker when completed (completed = delete from dict)
  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->RemovePartitionFromTracker(partition.dev_id, part_key);

  // std::string uuid = std::to_string(partition.dev_id);
  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->IncRspCbCount(partition.oid, 1);
}

void ProxySvrHandle::SendInLoop(const ConnectionUeventPtr& conn,
                                const MatrixPartitionPtr partition) {
  string client_addr = conn->GetPeerAddress().ToString();
  task_queue_.push_back([this, conn, partition, client_addr]() {
    // Use protobuf serialization instead of binary
    auto [data, size] = partition->SerializeToProto();
    conn->SendData(data, size);
    free(data);
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
}

void ProxySvrHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected from " << client_addr;
  conn_inflight_.erase(client_addr);
}

/********************************ProxySvrImpl****************************************/

ProxySvrImpl::ProxySvrImpl(ProxyEnvCfg& ctx)
    : ctx_(ctx), listener_(nullptr), rsp_cb_counts_(5) {}

void ProxySvrImpl::Initialize(UeventLoop* loop) {
  LOG_INFO << "[ProxySvrImpl::Initialize] Starting server initialization";
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - listen_ip=" << ctx_.listen_ip
           << ", listen_port=" << ctx_.listen_port;
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - num_device="
           << ctx_.num_device << ", thread=" << ctx_.thread;

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
}

void ProxySvrImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);

  conn_map_[conn->GetPeerAddress().ToString()] = conn;

  LOG_INFO << "[ConnectionSuccessCb] New connection added. Total connections: "
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

  conn_map_.erase(conn->GetPeerAddress().ToString());

  LOG_INFO << "[ConnectionClosedCb] Connection removed. Remaining connections: "
           << conn_map_.size();
  LOG_INFO << "[ConnectionClosedCb] Connection map contents:";
  for (const auto& conn_pair : conn_map_) {
    LOG_INFO << "  - " << conn_pair.first << " -> "
             << (conn_pair.second ? "valid" : "null");
  }
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

  for (auto& partition : shared_partitions) {
    partition->oid = mm_count_;
    // Debug:
    LOG_DEBUG << "[DispatchMatMulAsync] partition dev_id=" << partition->dev_id
              << ", conn_map_.size()=" << conn_map_.size();
    if (conn_map_.empty()) {
      LOG_ERROR << "[DispatchMatMulAsync] conn_map_ is empty!";
      continue;
    }
    if (partition->dev_id < 0 ||
        static_cast<size_t>(partition->dev_id) >= conn_map_.size()) {
      LOG_ERROR << "[DispatchMatMulAsync] partition->dev_id "
                << partition->dev_id
                << " out of range! conn_map_.size()=" << conn_map_.size();
      continue;
    }

    LOG_DEBUG << "[DispatchMatMulAsync] Looking for connection at index "
              << partition->dev_id;

    auto it = conn_map_.begin();
    std::advance(it, partition->dev_id);
    if (it == conn_map_.end() || !it->second) {
      LOG_ERROR << "[DispatchMatMulAsync] Invalid connection for dev_id "
                << partition->dev_id;
      continue;
    }

    LOG_DEBUG << "[DispatchMatMulAsync] Found connection for dev_id "
              << partition->dev_id << ": " << it->first;

    auto* loop = it->second->GetLoop();
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
        bind(&ProxySvrHandle::SendInLoop, handle, it->second, partition));
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
  while (rsp_cb_counts_[oid] > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto shape = outputs_[oid].sizes().vec();
  auto wait_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  LOG_DEBUG << "Waiting time: " << wait_time << "us for oid: " << oid
            << ", shape: " << shape;

  mm_count_--;
  return outputs_[oid];
}

void ProxySvrImpl::RephrasePartitions(
    std::vector<MatrixPartition>& partitions,
    const std::unordered_set<int64_t>& excluded_devices) {
  // Use actual number of connected devices instead of configured num_device
  int actual_num_devices = static_cast<int>(conn_map_.size());
  LOG_INFO << "[RephrasePartitions] Starting with " << partitions.size()
           << " partitions, configured num_device=" << ctx_.num_device
           << ", actual connected devices=" << actual_num_devices
           << ", excluded_devices=" << excluded_devices.size();

  if (actual_num_devices == 0) {
    LOG_ERROR << "[RephrasePartitions] No devices connected!";
    return;
  }

  std::vector<float> device_time(actual_num_devices, 0);
  device_tensors_.assign(actual_num_devices, std::unordered_set<TensorKey>());

  LOG_INFO << "[RephrasePartitions] Initialized device_time vector with size "
           << device_time.size();

  // greedy algorithm to select the minimal time
  for (size_t part_idx = 0; part_idx < partitions.size(); ++part_idx) {
    auto& partition = partitions[part_idx];
    LOG_DEBUG << "[RephrasePartitions] Processing partition " << part_idx << "/"
              << partitions.size();

    float min_time = std::numeric_limits<float>::max();
    int min_device = 0;
    auto version = partition.version;
    auto tensor_key_row = partition.GetRowKey();
    auto tensor_key_col = partition.GetColKey();
    bool min_r_cached = false;
    bool min_c_cached = false;

    LOG_INFO << "[RephrasePartitions] Processing partition " << part_idx
             << " - checking " << actual_num_devices << " devices";

    for (int i = 0; i < actual_num_devices; i++) {
      // Skip excluded devices (used for retry scenarios)
      if (excluded_devices.find(i) != excluded_devices.end()) {
        LOG_DEBUG << "[RephrasePartitions] Skipping excluded device " << i;
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
      // ul_time + dl_time + flops + device_time[i];
      if (time < min_time) {
        min_time = time;
        min_device = i;
        min_r_cached = r_cached;
        min_c_cached = c_cached;
        // fprintf(stderr, "Device: %d, UL time: %f, DL time: %f, FLOPS: %f,
        // Time: %f\n", i, ul_time, dl_time, flops, time);
      }
    }

    LOG_INFO << "[RephrasePartitions] Partition " << part_idx
             << " assigned to device " << min_device << " with time "
             << min_time;

    if (min_time == std::numeric_limits<float>::max()) {
      LOG_ERROR << "[RephrasePartitions] Failed to find available device for partition "
                << part_idx << ". excluded_devices.size()=" << excluded_devices.size()
                << ", actual_num_devices=" << actual_num_devices;
      // This should not happen if there are available devices
      assert(false && "No available device found for partition");
      continue;
    }

    // update the time for the device
    device_time[min_device] = min_time;
    partition.dev_id = min_device;
    device_tensors_[min_device].insert(tensor_key_row);
    device_tensors_[min_device].insert(tensor_key_col);

    // Add partition to tracker (dict: device_id -> {partition_keys})
    AddPartitionToTracker(min_device, partition.GetPartitionKey());

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

void ProxySvrImpl::AddPartitionToTracker(int64_t device_id,
                                        const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(partition_tracker_mutex_);
  partition_tracker_[device_id].insert(partition_key);
  LOG_DEBUG << "[AddPartitionToTracker] Added partition " << partition_key
            << " to device " << device_id;
}

void ProxySvrImpl::RemovePartitionFromTracker(int64_t device_id,
                                             const std::string& partition_key) {
  std::lock_guard<std::mutex> lock(partition_tracker_mutex_);
  auto it = partition_tracker_.find(device_id);
  if (it != partition_tracker_.end()) {
    size_t erased = it->second.erase(partition_key);
    if (erased > 0) {
      LOG_DEBUG << "[RemovePartitionFromTracker] Removed partition "
                << partition_key << " from device " << device_id;
      // If device has no more partitions, remove the device entry
      if (it->second.empty()) {
        partition_tracker_.erase(it);
        LOG_INFO << "[RemovePartitionFromTracker] Device " << device_id
                 << " removed from tracker (no more partitions)";
      }
    } else {
      LOG_WARN << "[RemovePartitionFromTracker] Partition " << partition_key
               << " not found on device " << device_id;
    }
  } else {
    LOG_WARN << "[RemovePartitionFromTracker] Device " << device_id
             << " not found in tracker";
  }
}

void ProxySvrImpl::HandleDeviceFailure(int64_t failed_device_id,
                                      int64_t target_device_id) {
  std::lock_guard<std::mutex> lock(partition_tracker_mutex_);
  auto it = partition_tracker_.find(failed_device_id);
  if (it != partition_tracker_.end()) {
    auto& failed_partitions = it->second;
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " failed with " << failed_partitions.size()
             << " partitions. Redistributing to device " << target_device_id; 

    // Merge all partitions from failed device to target device (set union)
    // unordered_set::insert with iterators is efficient and automatically handles duplicates
    auto& target_partitions = partition_tracker_[target_device_id];
    target_partitions.insert(failed_partitions.begin(),
                             failed_partitions.end());

    LOG_INFO << "[HandleDeviceFailure] Redistributed " << failed_partitions.size()
             << " partitions. Target device now has "
             << target_partitions.size() << " total partitions";

    // Remove failed device from tracker
    partition_tracker_.erase(it);
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " removed from tracker";
  } else {
    LOG_WARN << "[HandleDeviceFailure] Device " << failed_device_id
             << " not found in tracker";
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
