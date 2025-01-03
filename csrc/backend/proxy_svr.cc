#include "proxy_svr.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/stats.h"
#include "eventloop_libevent.h"
#include "listener_libevent.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <iostream>
#include <set>

#include "base/logging.h"

/*********************************ProxySvrHandle************************************/

ProxySvrHandle::ProxySvrHandle(ProxyEnvCfg& ctx, UeventLoop* loop)
    : ctx_(ctx), loop_(loop) {}

void ProxySvrHandle::ThreadInit(uevent::UeventLoop* loop) {
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  // handle->RegisterService();
}

void ProxySvrHandle::RequestCb(const ConnectionUeventPtr& conn) {
  size_t readable = conn->ReadableLength();

  int ret = 0;
  size_t packsize;
  ret = conn->ReceiveData(&packsize, sizeof(size_t));
  if (ret < 0) {
    LOG_ERROR << "ReceiveData packsize err";
    return;
  }
  packsize = ntohl(packsize);
  uint32_t datasize = packsize + sizeof(uint32_t);

  if (static_cast<uint32_t>(readable) < datasize) {
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
}

void ProxySvrHandle::HandleMatMul(const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.Deserialize(message->payload, message->payloadlen);
  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG("{} RSP Deserialization time: {}us", part_key,
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());

  auto [o_ptr, o_size] = partition.mat[0];
  int64_t row_size = o_size / partition.h_dim / sizeof(float);
  int64_t col_size = partition.h_dim;

  uint64_t ul_overhead = CurrentTimeMicros() - partition.timestamp;

  LOG_DEBUG("{} partition: {}", part_key, partition.DebugString());

  start = std::chrono::high_resolution_clock::now();
  auto output = torch::from_blob(o_ptr, {row_size, col_size},
                                 FLOAT32_TENSOR_OPTIONS(torch::kCPU));
  {
    // std::lock_guard<std::mutex> lock(outputs_mutex_[partition.oid]);
    auto& output_matrix = reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
                              ->GetOutputMatrix(partition.oid);
    IndexPutMatrixBlock(output_matrix, output, partition.row, partition.col,
                        partition.pivot, block_size_);
  }
  end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG("UpdateMatrixBlock time: {}us",
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());

  std::string uuid = std::to_string(partition.dev_id);
  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->IncRspCbCount(partition.oid, 1);
  LOG_DEBUG("Number of responses left: {}",
            rsp_cb_counts_[partition.oid].load());
}

void ProxySvrHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected from " << client_addr;
}

void ProxySvrHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected from " << client_addr;
}

/********************************ProxySvrImpl****************************************/

ProxySvrImpl::ProxySvrImpl(ProxyEnvCfg& ctx) : ctx_(ctx), listener_(nullptr) {}

void ProxySvrImpl::Initialize(UeventLoop* loop) {
  auto create_handle_cb = bind(ProxySvrHandle::CreateMyself, ref(ctx_), _1);
  UsockAddress addr(ctx_.listen_ip, ctx_.listen_port);
  listener_ =
      make_shared<ListenerLibevent>(loop, addr, "proxy_listener", Option());
  listener_->SetCreateLoopHandleCb(create_handle_cb);
  listener_->SetThreadInitCb(ProxySvrHandle::ThreadInit);
  listener_->SetConnectionSuccessCb(
      bind(&ProxySvrImpl::ConnectionSuccessCb, shared_from_this(), _1));
  listener_->SetMessageReadCb(
      bind(&ProxySvrImpl::RpcRequestCb, shared_from_this(), _1));
  listener_->SetThreadNum(ctx_.thread);
  listener_->StartPrimaryLoop();

  LOG_DEBUG << "ProxySvrImpl listen on:" << ctx_.listen_ip << ":"
            << ctx_.listen_port;

  ctx_.instance = this;

  // Start();
  InitLogger();

  // no more than 5 MAtMul in parallel
  outputs_ = std::move(std::vector<torch::Tensor>(5));
  // rsp_cb_counts_ = std::move(std::vector<std::atomic_ullong>(5));
  // outputs_mutex_ = std::move(std::vector<std::mutex>(5));
  for (int i = 0; i < 5; i++) {
    outputs_[i] = torch::empty({0, 0});
    rsp_cb_counts_[i] = 0;
  }
}

void ProxySvrImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);
}

void ProxySvrImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);
}

void ProxySvrImpl::RequestCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);

  string client_addr = conn->GetPeerAddress().ToString();
  handle->RequestCb(conn, client_addr);
}

void ProxySvrImpl::RequestCb(const uevent::ConnectionUeventPtr& conn) {
  size_t readable = conn->ReadableLength();

  int ret = 0;
  size_t packsize;
  ret = conn->ReceiveData(&packsize, sizeof(size_t));
  if (ret < 0) {
    conn->ForceClose();
    LOG_ERROR << "ReceiveData on a closed connection";
    return;
  }

  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->RequestCb(conn);
}

void ProxySvrImpl::DispatchMatMulAsync(torch::Tensor& mat_a,
                                       torch::Tensor& mat_b) {
  outputs_[mm_count_].set_data(CreateOutputMatrix(mat_a, mat_b));
  auto partitions = PartitionMatrices(mat_a, mat_b, block_size_);
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  auto cur_ver = partitions[0].version;
  LOG_INFO("[{}] Number of partitions: {} for A: {} and B: {}", cur_ver,
           partitions.size(), a_shape, b_shape);

  RephrasePartitions(partitions);

  auto start = std::chrono::high_resolution_clock::now();

  DecRspCbCount(mm_count_, partitions.size());
  auto it = conn_map_.begin();
  for (auto& partition : partitions) {
    partition.oid = mm_count_;
    auto [data, size] = partition.Serialize();
    it->second->SendData(data, size);
    free(data);
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_INFO("Publish time: {}us",
           std::chrono::duration_cast<std::chrono::microseconds>(end - start)
               .count());
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
  LOG_DEBUG("Waiting time: {}us for oid: {}, shape: {}", wait_time, oid, shape);

  mm_count_--;
  return outputs_[oid];
}

void ProxySvrImpl::RephrasePartitions(
    std::vector<MatrixPartition>& partitions) {
  std::vector<float> device_time(num_devices_, 0);
  std::vector<std::unordered_set<TensorKey>> device_tensors(num_devices_);

  std::random_shuffle(partitions.begin(), partitions.end());

  // greedy algorithm to select the minimal time
  for (auto& partition : partitions) {
    float min_time = std::numeric_limits<float>::max();
    int min_device = 0;
    auto version = partition.version;
    auto tensor_key_row = partition.GetRowKey();
    auto tensor_key_col = partition.GetColKey();
    bool min_r_cached = false;
    bool min_c_cached = false;
    for (int i = 0; i < num_devices_; i++) {
      auto& tensors = device_tensors[i];

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
    assert(min_time != std::numeric_limits<float>::max());
    // update the time for the device
    device_time[min_device] = min_time;
    partition.dev_id = min_device;
    device_tensors[min_device].insert(tensor_key_row);
    device_tensors[min_device].insert(tensor_key_col);

    if (min_r_cached) {
      partition.mat[0] = {nullptr, 0};
    }
    if (min_c_cached) {
      partition.mat[1] = {nullptr, 0};
    }
  }
  LOG_INFO("Device time: {}", device_time);
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

ProxyStatus ProxySvr::Initialize(const std::string& cfg_file) {
  Status status;
  context_.Initialize(cfg_file);
  svr_ = make_shared<ProxySvrImpl>(context_);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxySvrHandle::CreateMyself, ref(context_), _1),
      bind(&ProxySvrImpl::Initialize, svr_, _1), "Proxy svr main thread");

  return status;
}

ProxyStatus ProxySvr::Start() {
  Status status;
  loop_thread_->StartLoop();
  return status;
}
