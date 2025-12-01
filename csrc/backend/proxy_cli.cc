#include "proxy_cli.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/generator.h"
#include "common/pytorch_defs.h"
#include "common/stats.h"
#include "device_tracker.h"
#include "network/eventloop_libevent.h"
#include "network/ueventloop_thread_pool.h"
#include "utils/logging.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <iostream>
#include <set>

#include "base/logging.h"

/*********************************ProxyCliHandle************************************/

ProxyCliHandle::ProxyCliHandle(ProxyEnvCfg& ctx, UeventLoop* loop)
    : ctx_(ctx), loop_(loop) {
  SRV_STATS->Initialize();
}

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

  auto [data, size] = partition.SerializeToProto();
  conn->SendData(data, size);

  // Record bytes sent (upload response back to server)
  DEVICE_TRACKER.RecordBytesSent(partition.dev_id, size);
  
  // Log upload throughput after sending response
  double upload_tp = DEVICE_TRACKER.GetUploadThroughput(partition.dev_id);
  double last_packet_tp = DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp = DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();
  
  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us, end_us);
  
  LOG_INFO << "[ResponseToCaller] Device " << partition.dev_id 
           << " - Sent: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Upload TP: " << upload_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // LOG_DEBUG << "Response sent to " << client_addr;
  free(data);
  // LOG_DEBUG << "Free data";

  RECORD_SRV_COUNT(SRV_TOTAL_QUERY, 1);
  RECORD_SRV_COUNT(SRV_TOTAL_TRAFFIC, size);

  LOG_DEBUG << "Response sent to " << client_addr << ", size: " << size;
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

  LOG_DEBUG << part_key << " Row: " << row.sizes().vec()
            << ", Col: " << col.sizes().vec();

  auto result = torch::mm(row, col.transpose(0, 1)).to(torch::kCPU);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto mm_time = duration.count();
  LOG_DEBUG << part_key << " Matmul real time: " << duration.count()
            << "us, Matmul logical time: " << mm_time;

  MatrixPartition response = partition;
  response.h_dim = result.size(1);
  response.timestamp = CurrentTimeMicros();
  response.mat.clear();
  response.mat.push_back({result.data_ptr(), result.numel() * sizeof(float)});

  ResponseToCaller(conn, response);
}

void ProxyCliHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected to " << client_addr;
  
  // Register device in tracker for throughput monitoring
  DEVICE_TRACKER.RegisterDevice(client_addr);
}

void ProxyCliHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected to " << client_addr;
  
  // Unregister device from tracker
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(client_addr);
  if (device_id != -1) {
    DEVICE_TRACKER.UnregisterDevice(device_id);
  }
}

/********************************ProxyCliImpl****************************************/

ProxyCliImpl::ProxyCliImpl(ProxyEnvCfg& ctx)
    : ctx_(ctx), connector_(nullptr), cached_tensors_(2ull * GB) {}

void ProxyCliImpl::Initialize(UeventLoop* loop) {
  UsockAddress addr(ctx_.listen_ip, ctx_.listen_port);
  connector_ = make_shared<ConnectorLibevent>(loop, addr, "proxy_connector");
  connector_->SetConnectionSuccessCb(
      bind(&ProxyCliImpl::ConnectionSuccessCb, shared_from_this(), _1));
  connector_->SetConnectionClosedCb(
      bind(&ProxyCliImpl::ConnectionClosedCb, shared_from_this(), _1));
  connector_->SetMessageReadCb(
      bind(&ProxyCliImpl::RequestCb, shared_from_this(), _1));
  connector_->Connect();

  LOG_DEBUG << "ProxyCliImpl connected to:" << ctx_.listen_ip << ":"
            << ctx_.listen_port;

  // InitLogger();
  // CUDA context warmup and do random matmul (skip if no CUDA available)
  try {
    if (torch::cuda::is_available()) {
      torch::Tensor warmup_a = torch::rand({128, 4096}).to(torch::kCUDA, 0);
      torch::Tensor warmup_b = torch::rand({4096, 128}).to(torch::kCUDA, 0);
      torch::mm(warmup_a, warmup_b);
      LOG_DEBUG << "CUDA warmup completed";
    } else {
      LOG_DEBUG << "CUDA not available, skipping CUDA warmup";
    }
  } catch (const std::exception& e) {
    LOG_WARN << "CUDA warmup failed: " << e.what() << ", continuing with CPU";
  }
}

void ProxyCliImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);
}

void ProxyCliImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);
}

void ProxyCliImpl::RequestCb(const ConnectionUeventPtr& conn) {
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

    LOG_DEBUG << "packsize: " << packsize << ", datasize: " << datasize
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

    auto partition = DecodeRequest(raw_data, datasize);
    
    // Record bytes received (download request from server)
    DEVICE_TRACKER.RecordBytesReceived(partition.dev_id, datasize);
    
    // Log download throughput after receiving request
    double download_tp = DEVICE_TRACKER.GetDownloadThroughput(partition.dev_id);
    double last_packet_tp = DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
    double avg_packet_tp = DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
    double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();
    
    uint64_t start_us, end_us;
    DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us, end_us);
    
    LOG_INFO << "[RequestCb] Device " << partition.dev_id 
             << " - Received: " << datasize << " bytes"
             << " [" << start_us << " -> " << end_us << " us]"
             << ", Download TP: " << download_tp << " B/s"
             << ", Last Packet TP: " << last_packet_tp << " B/s"
             << ", Avg Packet TP: " << avg_packet_tp << " B/s"
             << " | Server Total TP: " << server_tp << " B/s";
    
    HandleMatMul(conn, partition);
    LOG_DEBUG << "Processed partition: " << partition.DebugString();
  }
}

void ProxyCliImpl::HandleMatMul(const ConnectionUeventPtr& conn,
                                MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();
  LOG_DEBUG << part_key << " partition: " << partition.DebugString();

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
    // std::lock_guard<std::mutex> lock(cache_mutex_);
    if (r_size > 0) {
      CacheTensor(tensor_key_row, r_ptr, r_size, partition.h_dim);
    }

    if (c_size > 0) {
      CacheTensor(tensor_key_col, c_ptr, c_size, partition.h_dim);
    }

    auto r_cached = cached_tensors_.Exist(tensor_key_row);
    auto c_cached = cached_tensors_.Exist(tensor_key_col);

    LOG_DEBUG << part_key << " Row cached: " << r_cached
              << ", row size: " << row_size << ", Col cached: " << c_cached
              << ", col size: " << col_size;

    FillPartition(partition);
    CheckCachedPartition(conn);

    if (r_size == 0 && !r_cached) {
      LOG_WARN << part_key << " Row not cached, saving for next msg";
      SavePartition(partition);
      return;
    }

    if (c_size == 0 && !c_cached) {
      LOG_WARN << part_key << " Col not cached, saving for next msg";
      SavePartition(partition);
      return;
    }
  }

  LOG_DEBUG << part_key << " Handle partition immediately";
  HandlePartition(conn, partition);
}

void ProxyCliImpl::CheckCachedPartition(
    const uevent::ConnectionUeventPtr& conn) {
  std::vector<std::string> keys;
  for (auto& c_part : cached_partitions_) {
    FillPartition(c_part);
    auto r_size = std::get<1>(c_part.mat[0]);
    auto c_size = std::get<1>(c_part.mat[1]);
    if (r_size > 0 && c_size > 0) {
      auto key = c_part.GetPartitionKey();
      LOG_DEBUG << key << " Handle partition from cache";
      HandlePartition(conn, c_part);
      keys.push_back(key);
    } else {
      // LOG_WARN << c_part.GetPartitionKey()
      //          << " Partition is not ready, r_size: " << r_size
      //          << ", c_size: " << c_size;
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
  // handle->HandlePartition(conn, partition);
  loop->RunInLoop(bind(&ProxyCliHandle::HandlePartition, handle, conn,
                       std::cref(partition)));
}

MatrixPartition ProxyCliImpl::DecodeRequest(const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.DeserializeFromProto(payload, size);
  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << part_key << " REQ Deserialization time: " << duration.count()
            << "us";
  return partition;
}

void ProxyCliImpl::CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                               int64_t h_dim) {
  // Add debug logging for all parameters
  LOG_INFO << "CacheTensor called: ptr=" << ptr << ", size=" << size
           << ", h_dim=" << h_dim;

  if (cached_tensors_.Exist(key)) {
    LOG_DEBUG << "Tensor already cached, skipping";
    return;
  }

  // Validate parameters before proceeding
  if (ptr == nullptr) {
    LOG_ERROR << "CacheTensor: ptr is nullptr!";
    return;
  }

  if (size <= 0) {
    LOG_ERROR << "CacheTensor: invalid size=" << size;
    return;
  }

  if (h_dim <= 0) {
    LOG_ERROR << "CacheTensor: invalid h_dim=" << h_dim;
    return;
  }

  LOG_DEBUG << "Allocating memory: size=" << size << " bytes";
  void* cpy_ptr = malloc(size);

  if (cpy_ptr == nullptr) {
    LOG_ERROR << "CacheTensor: malloc failed for size=" << size;
    return;
  }

  LOG_DEBUG << "malloc succeeded: cpy_ptr=" << cpy_ptr;

  int64_t ld_size = size / h_dim / sizeof(float);
  LOG_DEBUG << "Calculated ld_size=" << ld_size << " (size=" << size
            << " / h_dim=" << h_dim << " / sizeof(float)=" << sizeof(float)
            << ")";

  LOG_DEBUG << "Copying data: from " << ptr << " to " << cpy_ptr
            << ", size=" << size << " bytes";
  std::memcpy(cpy_ptr, ptr, size);

  LOG_DEBUG << "memcpy completed successfully";

  cached_tensors_.Put(
      key,
      torch::from_blob(
          cpy_ptr, {ld_size, h_dim}, [](void* ptr) { free(ptr); },
          FLOAT32_TENSOR_OPTIONS(torch::kCPU)),
      size);

  LOG_INFO << "CacheTensor completed successfully";
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

void ProxyCli::Initialize(const std::string& cfg_file) {
  context_.Initialize(cfg_file);
  svr_ = make_shared<ProxyCliImpl>(context_);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxyCliHandle::CreateMyself, ref(context_), _1),
      bind(&ProxyCliImpl::Initialize, svr_, _1), "Proxy svr main thread");
}

void ProxyCli::Start() { loop_thread_->StartLoop(); }

void ProxyCli::Send(const torch::Tensor& tensor, std::optional<int64_t> rank) {}
void ProxyCli::Receive(torch::Tensor& tensor, std::optional<int64_t> rank) {}
void ProxyCli::AsyncSend(const torch::Tensor& tensor,
                         std::optional<int64_t> rank) {}
void ProxyCli::AsyncReceive(torch::Tensor& tensor,
                            std::optional<int64_t> rank) {}
