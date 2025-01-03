#include "proxy_cli.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/stats.h"
#include "network/eventloop_libevent.h"
#include "network/ueventloop_thread_pool.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <iostream>
#include <set>

#include "base/logging.h"

/*********************************ProxyCliHandle************************************/

ProxyCliHandle::ProxyCliHandle(ProxyEnvCfg& ctx, UeventLoop* loop)
    : ctx_(ctx), loop_(loop) {}

void ProxyCliHandle::ThreadInit(uevent::UeventLoop* loop) {
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  // handle->RegisterService();
}

// void ProxyCliHandle::RegisterService() {
//   proxy_cbs_.insert(make_pair(NFSPROC4_NULL, NullService::CreateMyself));
//   proxy_cbs_.insert(
//       make_pair(NFSPROC4_COMPOUND, CompoundService::CreateMyself));
// }

void ProxyCliHandle::ResponseToCaller(const uevent::ConnectionUeventPtr& conn,
                                      const MatrixPartition& partition) {
  assert(conn);

  string client_addr = conn->GetPeerAddress().ToString();
  if (conn->IsClosed()) {
    LOG_WARN << "connection already closed:" << client_addr;
    return;
  }

  auto [data, size] = response.Serialize();

  conn->SendData(data, size);

  RECORD_SRV_COUNT(SRV_TOTAL_QUERY, 1);
  RECORD_SRV_COUNT(SRV_TOTAL_TRAFFIC, size);
}

void ProxyCliHandle::HandlePartition(const uevent::ConnectionUeventPtr& conn,
                                     const MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();
  // create tensors from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  auto start = std::chrono::high_resolution_clock::now();
  auto row = torch::from_blob(r_ptr, {row_size, partition.h_dim},
                              FLOAT32_TENSOR_OPTIONS(torch::kCPU))
                 .to(torch::kCUDA, 0);
  auto col = torch::from_blob(c_ptr, {col_size, partition.h_dim},
                              FLOAT32_TENSOR_OPTIONS(torch::kCPU))
                 .to(torch::kCUDA, 0);

  LOG_DEBUG("{} Row: {}, Col: {}", part_key, row.sizes().vec(),
            col.sizes().vec());

  auto result = torch::mm(row, col.transpose(0, 1)).to(torch::kCPU);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG("{} Matmul real time: {}us, Matmul logical time: {}us", part_key,
            duration.count(), mm_time);
  mm_time = duration.count();

  MatrixPartition response = partition;
  response.h_dim = result.size(1);
  response.timestamp = CurrentTimeMicros();
  response.mat.clear();
  response.mat.push_back({result.data_ptr(), result.numel() * sizeof(float)});

  ResponseToCaller(conn, response);
}

void ProxyCliHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected from " << client_addr;
}

void ProxyCliHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected from " << client_addr;
}

/********************************ProxyCliImpl****************************************/

ProxyCliImpl::ProxyCliImpl(ProxyEnvCfg& ctx) : ctx_(ctx), connector_(nullptr) {}

void ProxyCliImpl::Initialize(UeventLoop* loop) {
  UsockAddress addr(ctx_.listen_ip, ctx_.listen_port);
  connector_ =
      make_shared<ConnectorLibevent>(loop, addr, "proxy_connector", Option());
  connector_->SetConnectionSuccessCb(
      bind(&ProxyCliImpl::ConnectionSuccessCb, shared_from_this(), _1));
  connector_->SetConnectionClosedCb(
      bind(&ProxyCliImpl::ConnectionClosedCb, shared_from_this(), _1));
  connector_->SetMessageReadCb(
      bind(&ProxyCliImpl::RequestCb, shared_from_this(), _1));
  connector_->Connect();

  LOG_DEBUG << "ProxyCliImpl connected to:" << ctx_.listen_ip << ":"
            << ctx_.listen_port;

  InitLogger();
}

void ProxyCliImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);
  cached_tensors_ =
      FixSizeLRUCache<TensorKey, torch::Tensor>(2ull * 1024 * 1024 * 1024);
}

void ProxyCliImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);
}

void ProxyCliImpl::RequestCb(const ConnectionUeventPtr& conn) {
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

  auto partition = DecodeRequest(raw_data, datasize);
  HandleMatMul(conn, partition);
}

void ProxyCliImpl::HandleMatMul(const ConnectionUeventPtr& conn,
                                MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();
  LOG_DEBUG("{} partition: {}", part_key, partition.DebugString());

  // create tensor from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  assert(row_size * partition.h_dim * sizeof(float) == r_size);
  assert(col_size * partition.h_dim * sizeof(float) == c_size);

  auto tensor_key_row = partition.GetRowKey();
  auto tensor_key_col = partition.GetColKey();

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (r_size > 0) {
      CacheTensor(tensor_key_row, r_ptr, r_size, partition.h_dim);
    }

    if (c_size > 0) {
      CacheTensor(tensor_key_col, c_ptr, c_size, partition.h_dim);
    }

    auto r_cached = cached_tensors_.Exist(tensor_key_row);
    auto c_cached = cached_tensors_.Exist(tensor_key_col);

    LOG_DEBUG("{} Row cached: {}, row size: {}, Col cached: {}, col size: {}",
              part_key, r_cached, row_size, c_cached, col_size);

    FillPartition(partition);

    if (r_size == 0 && !r_cached) {
      LOG_WARN("{} Row not cached, saving for next msg", part_key);
      SavePartition(partition);
      return;
    }

    if (c_size == 0 && !c_cached) {
      LOG_WARN("{} Col not cached, saving for next msg", part_key);
      SavePartition(partition);
      return;
    }
  }

  LOG_DEBUG("{} Handle partition immediately", part_key);
  HandlePartition(conn, partition);

  std::vector<std::string> keys;
  for (auto& c_part : cached_partitions_) {
    FillPartition(c_part);
    r_size = std::get<1>(c_part.mat[0]);
    c_size = std::get<1>(c_part.mat[1]);
    if (r_size > 0 && c_size > 0) {
      auto key = c_part.GetPartitionKey();
      LOG_DEBUG("{} Handle partition from cache", key);
      HandlePartition(c_part);
      keys.push_back(key);
    } else {
      LOG_WARN("{} Partition is not ready, r_size: {}, c_size: {}",
               c_part.GetPartitionKey(), r_size, c_size);
    }
  }

  for (auto it = cached_partitions_.begin(); it != cached_partitions_.end();) {
    if (std::find(keys.begin(), keys.end(), it->GetPartitionKey()) !=
        keys.end()) {
      it = cached_partitions_.erase(it);
    } else {
      ++it;
    }
  }
}

void ProxyCliImpl::HandlePartition(const ConnectionUeventPtr& conn,
                                   const MatrixPartition& partition) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  handle->HandlePartition(conn, partition);
}

MatrixPartition ProxyCliImpl::DecodeRequest(const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.Deserialize(message->payload, message->payloadlen);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG("{} REQ Deserialization time: {}us", part_key, duration.count());
  return partition;
}

void ProxyCliImpl::CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                               int64_t h_dim) {
  if (cached_tensors_.Exist(key)) {
    return;
  }
  void* cpy_ptr = kCachingAllocator->Allocate(size);
  int64_t ld_size = size / h_dim / sizeof(float);
  std::memcpy(cpy_ptr, ptr, size);
  cached_tensors_.Put(key,
                      torch::from_blob(cpy_ptr, {ld_size, h_dim},
                                       FLOAT32_TENSOR_OPTIONS(torch::kCPU)),
                      size);
}

void ProxyCliImpl::FillPartition(MatrixPartition& partition) {
  auto r_size = std::get<1>(partition.mat[0]);
  auto c_size = std::get<1>(partition.mat[1]);
  auto tensor_key_row = partition.GetRowKey();
  auto tensor_key_col = partition.GetColKey();
  auto r_cached = cached_tensors_.Exist(tensor_key_row);
  auto c_cached = cached_tensors_.Exist(tensor_key_col);
  if (r_size == 0 && r_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_row);
    partition.mat[0] = {cached_tensor.data_ptr(),
                        cached_tensor.numel() * sizeof(float)};
  }

  if (c_size == 0 && c_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_col);
    partition.mat[1] = {cached_tensor.data_ptr(),
                        cached_tensor.numel() * sizeof(float)};
  }
}

void ProxyCliImpl::SavePartition(MatrixPartition& partition) {
  for (auto& mat : partition.mat) {
    mat = {nullptr, 0};
  }
  cached_partitions_.push_back(partition);
}

/*********************************ProxyCli***************************************/
typedef ProxyCli::Status ProxyStatus;
typedef ProxyStatus::StatusType ProxyStatusType;
const map<ProxyStatusType, string> ProxyStatus::status_str_ = {
    {kOK, ""},
    {kFatal, "Proxy svr fatal:"},
    {kUnknown, "Proxy svr unknown:"},
};

ProxyCli::ProxyCli() : svr_(nullptr), loop_thread_(nullptr) {}

ProxyStatus ProxyCli::Initialize(const std::string& cfg_file) {
  Status status;
  context_.Initialize(cfg_file);
  svr_ = make_shared<ProxyCliImpl>(context_);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxyCliHandle::CreateMyself, ref(context_), _1),
      bind(&ProxyCliImpl::Initialize, svr_, _1), "Proxy svr main thread");
  return status;
}

ProxyStatus ProxyCli::Start() {
  Status status;
  loop_thread_->StartLoop();
  return status;
}
