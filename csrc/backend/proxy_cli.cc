#include "proxy_cli.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/generator.h"
#include "common/pytorch_defs.h"
#include "common/stats.h"
#include "device_tracker.h"
#include "network/eventloop_libevent.h"
#include "network/ueventloop_thread_pool.h"
#include "proto_base.h"
#include "utils/cuda_utils.h"
#include "utils/logging.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <iostream>
#include <set>

#include "base/logging.h"

namespace morphling {
namespace backend {

/*********************************ProxyCliHandle************************************/

ProxyCliHandle::ProxyCliHandle(ProxyEnvCfg& ctx, UeventLoop* loop)
    : ctx_(ctx), loop_(loop) {
  SRV_STATS->Initialize();
  cublasCreate(&cublas_handle_);
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

  auto buffer = partition.Serialize();
  auto data = buffer->GetBuffer();
  auto size = buffer->GetSize();
  conn->SendData(data, size);

  // Record bytes sent (upload response back to server)
  DEVICE_TRACKER.RecordBytesSent(partition.dev_id, size);

  // Log upload throughput after sending response
  double upload_tp = DEVICE_TRACKER.GetUploadThroughput(partition.dev_id);
  double last_packet_tp =
      DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp =
      DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();

  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us,
                                              end_us);

  LOG_INFO << "[ResponseToCaller] Device " << partition.dev_id
           << " - Sent: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Upload TP: " << upload_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // Log throughput to file
  DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, partition.gemm_id,
                                     "UPLOAD", size, upload_tp, start_us,
                                     end_us);

  RECORD_SRV_COUNT(SRV_TOTAL_QUERY, 1);
  RECORD_SRV_COUNT(SRV_TOTAL_TRAFFIC, size);

  LOG_DEBUG << "Response sent to " << client_addr << ", size: " << size;
}

void ProxyCliHandle::HandlePartition(const uevent::ConnectionUeventPtr& conn,
                                     const MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();

  // Record COMPUTE start time (virtual time)
  uint64_t vt_compute_start = VirtualClockNow();
  LOG_INFO << "[HandlePartition] Logging COMPUTE START for device "
           << partition.dev_id << ", gemm_id=" << partition.gemm_id
           << ", vt_start=" << vt_compute_start;
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "COMPUTE", "START", vt_compute_start,
                                     vt_compute_start);

  // create tensors from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  cudaHostRegister(r_ptr, r_size, cudaHostRegisterDefault);
  cudaHostRegister(c_ptr, c_size, cudaHostRegisterDefault);

  size_t result_size = r_size * c_size * sizeof(float);
  auto* result_ptr = aligned_alloc(4096, result_size);
  cudaHostRegister(result_ptr, result_size, cudaHostRegisterDefault);

  auto start = std::chrono::high_resolution_clock::now();
  // auto row = torch::from_blob(r_ptr, {row_size, partition.h_dim},
  //                             FLOAT32_TENSOR_OPTIONS(torch::kCPU))
  //                .to(torch::kCUDA, 0);
  // auto col = torch::from_blob(c_ptr, {col_size, partition.h_dim},
  //                             FLOAT32_TENSOR_OPTIONS(torch::kCPU))
  //                .to(torch::kCUDA, 0);

  // LOG_DEBUG << part_key << " Row: " << row.sizes().vec()
  //           << ", Col: " << col.sizes().vec();

  float alpha = 1.0f;
  float beta = 0.0f;

  // auto result = torch::mm(row, col.transpose(0, 1)).to(torch::kCPU);
  cublasStatus_t cublas_status =
      cublasSgemm(cublas_handle_,
                  CUBLAS_OP_N,      // col^T: transpose col (op(col) = col^T)
                  CUBLAS_OP_N,      // row: no transpose
                  c_size,           // m: number of rows in result (col_size)
                  r_size,           // n: number of cols in result (row_size)
                  partition.h_dim,  // k: inner dimension (h_dim)
                  &alpha,
                  (const float*)c_ptr,  // A: col (col_size x h_dim)
                  c_size,               // lda: leading dimension of col
                  (const float*)r_ptr,  // B: row (row_size x h_dim)
                  partition.h_dim,      // ldb: leading dimension of row
                  &beta,
                  (float*)result_ptr,  // C: result (col_size x row_size)
                  c_size               // ldc: leading dimension of result
      );

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto mm_time = duration.count();
  LOG_DEBUG << part_key << " Matmul real time: " << duration.count()
            << "us, Matmul logical time: " << mm_time;

  // Record COMPUTE end time (virtual time)
  uint64_t vt_compute_end = VirtualClockNow();
  LOG_INFO << "[HandlePartition] Logging COMPUTE END for device "
           << partition.dev_id << ", gemm_id=" << partition.gemm_id
           << ", vt_start=" << vt_compute_start
           << ", vt_end=" << vt_compute_end;
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "COMPUTE", "END", vt_compute_start,
                                     vt_compute_end);

  MatrixPartition response = partition;
  // response.h_dim = result.size(1);
  response.h_dim = c_size;
  response.timestamp = CurrentTimeMicros();
  response.mat.clear();
  // response.mat.push_back({result.data_ptr(), result.numel() *
  // sizeof(float)});
  response.mat.push_back({result_ptr, result_size});

  ResponseToCaller(conn, response);

  cudaHostUnregister(r_ptr);
  cudaHostUnregister(c_ptr);
  cudaHostUnregister(result_ptr);
  free(result_ptr);
}

void ProxyCliHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected to " << client_addr;
}

void ProxyCliHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected to " << client_addr;
}

/********************************ProxyCliImpl****************************************/

ProxyCliImpl::ProxyCliImpl(ProxyEnvCfg& ctx)
    : ctx_(ctx), connector_(nullptr), cached_tensors_(5ull * GB) {}

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

  // Initialize virtual clock
  base::VirtualClock::instance().Initialize();
  LOG_INFO << "[ProxyCliImpl::Initialize] Virtual clock initialized";

  // Initialize performance logging (client side)
  // Client processes requests from all devices, so we use device ID 0 for
  // client-side processing
  DEVICE_TRACKER.InitSeparatePerfLog("./logs", "device", 0);
  LOG_INFO << "[ProxyCliImpl::Initialize] Performance logging initialized at "
              "./logs/perf_device_0.log";

  // CUDA context warmup and do random matmul (skip if no CUDA available)
  if (torch::cuda::is_available()) {
    torch::Tensor warmup_a = torch::rand({128, 4096}).to(torch::kCUDA, 0);
    torch::Tensor warmup_b = torch::rand({4096, 128}).to(torch::kCUDA, 0);
    torch::mm(warmup_a, warmup_b);
    LOG_DEBUG << "CUDA warmup completed";
  } else {
    LOG_DEBUG << "CUDA not available, skipping CUDA warmup";
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
    uint32_t packsize = 0;
    ret = conn->ReceiveData(&packsize, sizeof(uint32_t));
    if (ret < 0) {
      LOG_ERROR << "ReceiveData packsize err";
      return;
    }
    packsize = ntohl(packsize);
    size_t datasize = packsize + sizeof(packsize);

    // LOG_DEBUG << "packsize: " << packsize << ", datasize: " << datasize
    //           << ", readable: " << readable;
    if (readable < datasize) {
      // std::unique_ptr<unsigned char[]> data(new unsigned char[readable]);
      // unsigned char* raw_data = data.get();
      // ret = conn->ReceiveData(raw_data, readable);

      // print raw_data in hex
      // LOG_DEBUG << "Partial data received (hex): " <<
      // BinaryToHex(static_cast<const unsigned char*>(raw_data), readable);

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

void ProxyCliImpl::DecodeAndDispatch(const ConnectionUeventPtr& conn,
                                     const void* payload, size_t size) {
  // Step 1: Decode proto message header to get message type
  int32_t message_type = GetMessageType(payload, size);

  if (message_type < 0) {
    LOG_FATAL << "Failed to decode message type";
    return;
  }

  // Step 2: Dispatch to appropriate handler based on message type
  switch (message_type) {
    case morphling::global_api::DEVICE_REGISTER_REQUEST:
      HandleRegisterRequest(conn, payload, size);
      break;

    case morphling::global_api::COMPUTE_GEMM_DATA:
      HandleMatMulRequest(conn, payload, size);
      break;

    default:
      LOG_FATAL << "Unknown message type: " << message_type;
      break;
  }
}

void ProxyCliImpl::HandleRegisterRequest(const ConnectionUeventPtr& conn,
                                         const void* payload, size_t size) {
  LOG_DEBUG << "Received registration request from server, size=" << size;

  // Use standard Deserialize interface
  DeviceRegisterRequest request;
  request.Deserialize(payload, size);

  // Request is empty, just send response with device profile
  SendRegisterResponse(conn);
}

void ProxyCliImpl::SendRegisterResponse(const ConnectionUeventPtr& conn) {
  LOG_DEBUG << "Sending device profile data to server";

  DeviceProfileData profile;
  profile.uuid = GenUUID64();
  profile.flops = 100000000000ull;              // 100 GFLOPS - placeholder
  profile.memory = 16ull * 1024 * 1024 * 1024;  // 16GB - placeholder
  profile.ul_bw = 10ull * 1024 * 1024 * 1024;   // 10 Gbps
  profile.dl_bw = 10ull * 1024 * 1024 * 1024;   // 10 Gbps
  profile.ul_lat = 1000;                        // 1ms
  profile.dl_lat = 1000;                        // 1ms

  auto buffer = profile.Serialize();
  int ret = conn->SendData(buffer->GetBuffer(), buffer->GetSize());
  if (ret < 0) {
    LOG_ERROR << "Failed to send device profile data";
    conn->ForceClose();
    return;
  }

  LOG_DEBUG << "Device profile data sent, size=" << buffer->GetSize()
            << ", profile: " << profile.DebugString() << ", hex: "
            << BinaryToHex(
                   static_cast<const unsigned char*>(buffer->GetBuffer()),
                   buffer->GetSize())
            << "";
}

void ProxyCliImpl::HandleMatMulRequest(const ConnectionUeventPtr& conn,
                                       const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();

  // Use standard Deserialize interface
  MatrixPartition partition;
  partition.Deserialize(payload, size);

  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << part_key << " REQ Deserialization time: " << duration.count()
            << "us";

  // Record bytes received (download request from server)
  DEVICE_TRACKER.RecordBytesReceived(partition.dev_id, size);

  // Log download throughput after receiving request
  double download_tp = DEVICE_TRACKER.GetDownloadThroughput(partition.dev_id);
  double last_packet_tp =
      DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp =
      DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();

  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us,
                                              end_us);

  LOG_INFO << "[HandleMatMulRequest] Device " << partition.dev_id
           << " - Received: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Download TP: " << download_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // Log throughput to file
  DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, partition.gemm_id,
                                     "DOWNLOAD", size, download_tp, start_us,
                                     end_us);

  // Process the partition
  HandleMatMul(conn, partition);
  LOG_DEBUG << "Processed partition: " << partition.DebugString();
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
  partition.Deserialize(payload, size, SerializationFormat::PROTOBUF);
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

}  // namespace backend
}  // namespace morphling
