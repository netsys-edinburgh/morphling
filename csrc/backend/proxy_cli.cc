#include "proxy_cli.h"

#include <cuda_runtime.h>

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

#include <sys/stat.h>

#include <iostream>
#include <set>

#include "base/log_file.h"
#include "base/logging.h"
#include "utils/device_log_tag.h"

namespace morphling {
namespace backend {

namespace {
std::unique_ptr<base::LogFile> g_client_log_file;

void TeeOutput(const char* msg, int len) {
  ::fwrite(msg, 1, len, stdout);
  if (g_client_log_file) {
    g_client_log_file->append(msg, len);
  }
}

void TeeFlush() {
  ::fflush(stdout);
  if (g_client_log_file) {
    g_client_log_file->flush();
  }
}

int CudaPinBuffer(void* ptr, size_t size) {
  auto err = cudaHostRegister(ptr, size, cudaHostRegisterMapped);
  return (err == cudaSuccess) ? 0 : -1;
}

void CudaUnpinBuffer(void* ptr, size_t size) { cudaHostUnregister(ptr); }
}  // namespace

int PosixPinBuffer(void* ptr, size_t size) {
  // No-op for CPU memory, but could add mlock here if desired
  return mlock(ptr, size);
}

void PosixUnpinBuffer(void* ptr, size_t size) { munlock(ptr, size); }

/*********************************ProxyCliHandle************************************/

ProxyCliHandle::ProxyCliHandle(ProxyEnvCfg& ctx, UeventLoop* loop,
                               int64_t device_id)
    : ctx_(ctx), loop_(loop), device_id_(device_id), cublas_handle_(nullptr) {
  SRV_STATS->Initialize();
  InitCublas();
}

ProxyCliHandle::~ProxyCliHandle() { CleanupCublas(); }

// Note: ThreadInit is no longer needed as cuBLAS is initialized in constructor

void ProxyCliHandle::InitCublas() {
  cublasStatus_t status = cublasCreate(&cublas_handle_);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLAS handle, status: " +
                             std::to_string(status));
  } else {
    LOG_DEBUG << "cuBLAS handle created successfully: "
              << (void*)cublas_handle_;
  }
}

void ProxyCliHandle::CleanupCublas() {
  if (cublas_handle_ != nullptr) {
    cublasStatus_t status = cublasDestroy(cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      LOG_ERROR << "Failed to destroy cuBLAS handle, status: " << status;
    } else {
      LOG_DEBUG << "cuBLAS handle destroyed successfully";
    }
    cublas_handle_ = nullptr;
  }
}

// void ProxyCliHandle::RegisterService() {
//   proxy_cbs_.insert(make_pair(NFSPROC4_NULL, NullService::CreateMyself));
//   proxy_cbs_.insert(
//       make_pair(NFSPROC4_COMPOUND, CompoundService::CreateMyself));
// }

// Context for zero-copy scatter-gather send from client.
// Ensures referenced memory (scatter-gather buffer + CUDA result buffer)
// stays alive until libevent finishes sending all segments.
struct ResponseSendContext {
  ScatterGatherBufferPtr sg_buffer;
  // Deferred CUDA managed memory release (result buffer)
  void* cuda_ptr = nullptr;

  ~ResponseSendContext() {
    if (cuda_ptr) {
      cudaFree(cuda_ptr);
    }
  }
};

static void ResponseSendCleanup(const void* /*data*/, size_t /*len*/,
                                void* arg) {
  delete static_cast<std::shared_ptr<ResponseSendContext>*>(arg);
}

void ProxyCliHandle::ResponseToCaller(const uevent::ConnectionUeventPtr& conn,
                                      const MatrixPartition& partition,
                                      void* deferred_cuda_ptr) {
  assert(conn);

  string client_addr = conn->GetPeerAddress().ToString();
  if (conn->IsClosed()) {
    LOG_WARN << DEV_TAG(device_id_, partition.gemm_id)
             << "connection already closed:" << client_addr;
    // Release deferred cuda managed memory since we won't send
    if (deferred_cuda_ptr) {
      cudaFree(deferred_cuda_ptr);
    }
    return;
  }

  // Record UPLOAD start time (virtual time)
  uint64_t vt_upload_start = VirtualClockNow();
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "UPLOAD", "START", vt_upload_start,
                                     vt_upload_start);

  // Zero-copy scatter-gather serialization (avoids tensor memcpy)
  auto sg_buffer = partition.SerializeZeroCopy();
  auto size = sg_buffer->GetTotalSize();

  // Create send context to keep scatter-gather buffer and cuda managed memory
  // alive until libevent finishes sending all segments
  auto ctx = std::make_shared<ResponseSendContext>();
  ctx->sg_buffer = sg_buffer;
  ctx->cuda_ptr = deferred_cuda_ptr;

  // Zero-copy send each segment
  for (const auto& segment : sg_buffer->GetSegments()) {
    auto* ref = new std::shared_ptr<ResponseSendContext>(ctx);
    conn->SendDataZeroCopy(segment.data, segment.size, ResponseSendCleanup,
                           ref);
  }

  // Record UPLOAD end time (virtual time)
  uint64_t vt_upload_end = VirtualClockNow();

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

  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[ResponseToCaller] Sent: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Upload TP: " << upload_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // Log virtual time event for UPLOAD
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "UPLOAD", "END", vt_upload_start,
                                     vt_upload_end);

  // Log throughput to file
  DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, partition.gemm_id,
                                     "UPLOAD", size, upload_tp, start_us,
                                     end_us);

  RECORD_SRV_COUNT(SRV_TOTAL_QUERY, 1);
  RECORD_SRV_COUNT(SRV_TOTAL_TRAFFIC, size);

  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << "Response sent to "
            << client_addr << ", size: " << size;
}

void ProxyCliHandle::HandlePartition(const uevent::ConnectionUeventPtr& conn,
                                     const MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();

  // Record COMPUTE start time (virtual time)
  uint64_t vt_compute_start = VirtualClockNow();
  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[HandlePartition] COMPUTE START"
           << ", vt_start=" << vt_compute_start;
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "COMPUTE", "START", vt_compute_start,
                                     vt_compute_start);

  // create tensors from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  if (row_size <= 0 || col_size <= 0 || partition.h_dim <= 0) {
    LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
              << "[HandlePartition] Invalid dimensions: row_size=" << row_size
              << ", col_size=" << col_size << ", h_dim=" << partition.h_dim;
    return;
  }

  // Input buffers come from CacheTensor (already cudaHostAlloc'd via cache)
  // No need to cudaHostRegister - they're already pinned

  // Result buffer via cudaMallocManaged (device-accessible unified memory)
  size_t result_size = 0;
  if (__builtin_mul_overflow(static_cast<size_t>(row_size),
                             static_cast<size_t>(col_size), &result_size) ||
      __builtin_mul_overflow(result_size, sizeof(float), &result_size)) {
    LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
              << "[HandlePartition] Result size overflow: row_size=" << row_size
              << ", col_size=" << col_size;
    return;
  }

  // register r_ptr and c_ptr as pinned memory (if not already pinned) to enable
  // zero-copy access from GPU This is a no-op if the memory is already pinned
  // (e.g. from cache)
  // if (cudaHostRegister(r_ptr, r_size, cudaHostRegisterDefault) !=
  // cudaSuccess) {
  //   LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
  //             << "[HandlePartition] Failed to register row buffer as pinned "
  //                "memory, code: "
  //             << cudaGetErrorString(cudaGetLastError());
  //   return;
  // }
  // if (cudaHostRegister(c_ptr, c_size, cudaHostRegisterDefault) !=
  // cudaSuccess) {
  //   LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
  //             << "[HandlePartition] Failed to register col buffer as pinned "
  //                "memory, code: "
  //             << cudaGetErrorString(cudaGetLastError());
  //   return;
  // }

  // We must be able to get device pointers for the input buffers to use with
  // cuBLAS. If the buffers are not already pinned (e.g. from cache), we attempt
  // to pin them here. If pinning fails, we cannot proceed with GPU computation.

  void* r_dev_ptr = nullptr;
  void* c_dev_ptr = nullptr;

  if (cudaHostGetDevicePointer(&r_dev_ptr, r_ptr, 0) != cudaSuccess) {
    LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
              << "[HandlePartition] Failed to get device pointer for row "
                 "buffer, code: "
              << cudaGetErrorString(cudaGetLastError());
    return;
  }

  if (cudaHostGetDevicePointer(&c_dev_ptr, c_ptr, 0) != cudaSuccess) {
    LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
              << "[HandlePartition] Failed to get device pointer for col "
                 "buffer, code: "
              << cudaGetErrorString(cudaGetLastError());
    return;
  }

  if (!r_dev_ptr || !c_dev_ptr) {
    LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
              << "[HandlePartition] Invalid device pointers: r_dev_ptr="
              << r_dev_ptr << ", c_dev_ptr=" << c_dev_ptr;
    return;
  }

  void* result_ptr = nullptr;
  cudaError_t alloc_err = cudaMallocManaged(&result_ptr, result_size);
  if (alloc_err != cudaSuccess) {
    LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
              << "[HandlePartition] cudaMallocManaged failed: "
              << cudaGetErrorString(alloc_err) << ", size=" << result_size;
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasStatus_t cublas_status =
      cublasGemmEx(cublas_handle_,
                   CUBLAS_OP_N,      // col^T: transpose col (op(col) = col^T)
                   CUBLAS_OP_N,      // row: no transpose
                   col_size,         // m: number of rows in result
                   row_size,         // n: number of cols in result
                   partition.h_dim,  // k: inner dimension (h_dim)
                   &alpha,
                   (const float*)c_dev_ptr,  // A: col (col_size x h_dim)
                   CUDA_R_32F,               // A type
                   col_size,                 // lda: leading dimension of col
                   (const float*)r_dev_ptr,  // B: row (row_size x h_dim)
                   CUDA_R_32F,               // B type
                   partition.h_dim,          // ldb: leading dimension of row
                   &beta,
                   (float*)result_ptr,  // C: result (col_size x row_size)
                   CUDA_R_32F,          // C type
                   col_size,            // ldc: leading dimension of result
                   CUBLAS_COMPUTE_32F,  // compute type
                   CUBLAS_GEMM_DEFAULT  // algo
      );

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " cuBLAS GEMM time: " << duration.count() << "us";

  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    LOG_ERROR << DEV_TAG(device_id_, partition.gemm_id)
              << "[HandlePartition] cuBLAS GEMM failed, status="
              << cublas_status;
    cudaFree(result_ptr);
    return;
  }

  // Record COMPUTE end time (virtual time)
  uint64_t vt_compute_end = VirtualClockNow();
  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[HandlePartition] COMPUTE END"
           << ", vt_start=" << vt_compute_start
           << ", vt_end=" << vt_compute_end;
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "COMPUTE", "END", vt_compute_start,
                                     vt_compute_end);

  // Prepare response with result
  MatrixPartition response = partition;
  response.h_dim = col_size;
  response.timestamp = CurrentTimeMicros();
  response.mat.clear();
  response.mat.push_back({result_ptr, result_size});

  // Zero-copy send: result buffer release is deferred to send completion
  ResponseToCaller(conn, response, result_ptr);
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

ProxyCliImpl::ProxyCliImpl(ProxyEnvCfg& ctx, int64_t device_id)
    : ctx_(ctx),
      device_id_(device_id),
      connector_(nullptr),
      cached_tensors_(5ull * GB) {}

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
  DEVICE_TRACKER.InitSeparatePerfLog("./logs", "device", device_id_);
  LOG_INFO << "[ProxyCliImpl::Initialize] Performance logging initialized at "
              "./logs/perf_device_"
           << device_id_ << ".log";

  // Tee logging: write LOG_* output to both console and rotating file
  ::mkdir("./logs", 0755);
  g_client_log_file = std::make_unique<base::LogFile>(
      "./logs/client_general", 256 * 1024 * 1024, true, 3);
  base::Logger::setOutput(TeeOutput);
  base::Logger::setFlush(TeeFlush);
  LOG_INFO << "[ProxyCliImpl::Initialize] Tee logging initialized";

  // CUDA context warmup (optional, can cause OOM if GPU memory is tight)
  // Disabled to avoid unnecessary GPU memory allocation during initialization
  // CUDA is already initialized when we call cudaMalloc in HandlePartition
  LOG_DEBUG << "CUDA initialization will happen on first GEMM computation";
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
      return;
    }

    // Zero-copy receive: get contiguous pointer into evbuffer
    unsigned char* raw_data = conn->PullupData(datasize);
    if (raw_data == nullptr) {
      LOG_ERROR << "PullupData failed for size " << datasize;
      return;
    }

    // Decode and dispatch message (processes data in-place before drain)
    DecodeAndDispatch(conn, raw_data, datasize);

    // Drain after processing is complete
    ret = conn->DrainData(datasize);
    if (ret < 0) {
      LOG_ERROR << "DrainData err";
      return;
    }
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
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " REQ Deserialization time: " << duration.count() << "us";

  // Record DOWNLOAD start time (virtual time)
  uint64_t vt_download_start = VirtualClockNow();
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "DOWNLOAD", "START", vt_download_start,
                                     vt_download_start);

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

  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[HandleMatMulRequest] Received: " << size << " bytes"
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
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id)
            << "Processed partition: " << partition.DebugString();
}

void ProxyCliImpl::HandleMatMul(const ConnectionUeventPtr& conn,
                                MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " partition: " << partition.DebugString();

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

    LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
              << " Row cached: " << r_cached << ", row size: " << row_size
              << ", Col cached: " << c_cached << ", col size: " << col_size;

    FillPartition(partition);
    CheckCachedPartition(conn);

    if (r_size == 0 && !r_cached) {
      LOG_WARN << DEV_TAG(device_id_, partition.gemm_id) << part_key
               << " Row not cached, saving for next msg";
      SavePartition(partition);
      return;
    }

    if (c_size == 0 && !c_cached) {
      LOG_WARN << DEV_TAG(device_id_, partition.gemm_id) << part_key
               << " Col not cached, saving for next msg";
      SavePartition(partition);
      return;
    }
  }

  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " Handle partition immediately";
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
      LOG_DEBUG << DEV_TAG(c_part.dev_id, c_part.gemm_id) << key
                << " Handle partition from cache";
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
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " REQ Deserialization time: " << duration.count() << "us";
  return partition;
}

void ProxyCliImpl::CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                               int64_t h_dim) {
  LOG_INFO << DEV_TAG_DEV(device_id_) << "CacheTensor called: ptr=" << ptr
           << ", size=" << size << ", h_dim=" << h_dim;

  if (cached_tensors_.Exist(key)) {
    LOG_DEBUG << DEV_TAG_DEV(device_id_) << "Tensor already cached, skipping";
    return;
  }

  if (ptr == nullptr) {
    LOG_ERROR << DEV_TAG_DEV(device_id_) << "CacheTensor: ptr is nullptr!";
    return;
  }

  if (size <= 0) {
    LOG_ERROR << DEV_TAG_DEV(device_id_)
              << "CacheTensor: invalid size=" << size;
    return;
  }

  if (h_dim <= 0) {
    LOG_ERROR << DEV_TAG_DEV(device_id_)
              << "CacheTensor: invalid h_dim=" << h_dim;
    return;
  }

  LOG_DEBUG << DEV_TAG_DEV(device_id_)
            << "Allocating pinned host memory: size=" << size << " bytes";
  auto [cpy_ptr, cpy_bucket] = CudaPinnedMemoryPool::Instance().Acquire(size);

  LOG_DEBUG << DEV_TAG_DEV(device_id_) << "Copying data: from " << ptr << " to "
            << cpy_ptr << ", size=" << size << " bytes";
  std::memcpy(cpy_ptr, ptr, size);

  LOG_DEBUG << DEV_TAG_DEV(device_id_) << "memcpy completed successfully";

  // Store pinned memory pointer and metadata directly (no torch::from_blob)
  // Create a dummy tensor just to store the metadata, but don't trigger CUDA
  // operations
  int64_t ld_size = size / h_dim / sizeof(float);

  // Store in cache with a lambda that returns pinned memory to pool on eviction
  cached_tensors_.Put(key,
                      torch::from_blob(
                          cpy_ptr, {ld_size, h_dim},
                          [cpy_bucket](void* ptr) {
                            CudaPinnedMemoryPool::Instance().Release(
                                ptr, cpy_bucket);
                          },
                          at::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(torch::kCPU)
                              .layout(torch::kStrided)),
                      size);

  LOG_INFO << DEV_TAG_DEV(device_id_) << "CacheTensor completed successfully";
}

void ProxyCliImpl::FillPartition(MatrixPartition& partition) {
  auto r_size = std::get<1>(partition.mat[0]);
  auto c_size = std::get<1>(partition.mat[1]);
  auto tensor_key_row = partition.GetRowKey();
  auto tensor_key_col = partition.GetColKey();
  auto r_cached = cached_tensors_.Exist(tensor_key_row);
  auto c_cached = cached_tensors_.Exist(tensor_key_col);

  if (!r_cached || !c_cached) {
    LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id)
              << partition.GetPartitionKey()
              << " FillPartition: r_cached=" << r_cached
              << ", c_cached=" << c_cached << ", r_size=" << r_size
              << ", c_size=" << c_size;
  }

  if (r_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_row);
    partition.mat[0] = {cached_tensor.data_ptr(),
                        cached_tensor.numel() * sizeof(float)};
  }
  if (c_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_col);
    partition.mat[1] = {cached_tensor.data_ptr(),
                        cached_tensor.numel() * sizeof(float)};
  }
  // if (r_size == 0 && r_cached) {
  //   auto cached_tensor = cached_tensors_.Get(tensor_key_row);
  //   partition.mat[0] = {cached_tensor.data_ptr(),
  //                       cached_tensor.numel() * sizeof(float)};
  // }

  // if (c_size == 0 && c_cached) {
  //   auto cached_tensor = cached_tensors_.Get(tensor_key_col);
  //   partition.mat[1] = {cached_tensor.data_ptr(),
  //                       cached_tensor.numel() * sizeof(float)};
  // }
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

void ProxyCli::Initialize(const std::string& cfg_file, int64_t device_id) {
  // detectm cuda availability and set device accordingly
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    LOG_WARN << "Failed to get CUDA device count: " << cudaGetErrorString(err)
             << ". Running in CPU-only mode.";
    AlignedBufferPool::instance().SetPinFunctions(PosixPinBuffer,
                                                  PosixUnpinBuffer);
  } else if (device_count == 0) {
    LOG_WARN << "No CUDA devices found. Running in CPU-only mode.";
    AlignedBufferPool::instance().SetPinFunctions(PosixPinBuffer,
                                                  PosixUnpinBuffer);
  } else {
    LOG_INFO << "CUDA devices detected: " << device_count
             << ". Running in GPU mode.";
    AlignedBufferPool::instance().SetPinFunctions(CudaPinBuffer,
                                                  CudaUnpinBuffer);
  }

  context_.Initialize(cfg_file);
  svr_ = make_shared<ProxyCliImpl>(context_, device_id);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxyCliHandle::CreateMyself, ref(context_), device_id, _1),
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
