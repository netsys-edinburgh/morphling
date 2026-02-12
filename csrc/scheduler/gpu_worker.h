#pragma once

#include <cublas_v2.h>
#include <cuda.h>  // CUDA driver API (green contexts)
#include <cuda_runtime_api.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include "intercept/interceptor.h"
#include "memory/caching_allocator.h"
#include "scheduling_policy.h"
#include "utils/cuda_utils.h"
#include "worker_base.h"

#define CUDA_MEMCPY_LOOP(trans, dst, src, ld, m, k, mode)                    \
  for (int col = 0; col < ((trans == 'N' || trans == 'n') ? k : m); col++) { \
    size_t num_elements = ((trans == 'N' || trans == 'n') ? m : k);          \
    size_t offset = col * ld;                                                \
    CHECK_CUDA_ERROR(cudaMemcpy(dst + offset, src + offset,                  \
                                num_elements * sizeof(float), mode))         \
  }

#define CUDA_MEMCPY_ASYNC_LOOP(trans, dst, src, ld, m, k, mode, stream)      \
  for (int col = 0; col < ((trans == 'N' || trans == 'n') ? k : m); col++) { \
    size_t num_elements = ((trans == 'N' || trans == 'n') ? m : k);          \
    size_t offset = col * ld;                                                \
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst + offset, src + offset,             \
                                     num_elements * sizeof(float), mode,     \
                                     stream))                                \
  }

#define CUDA_TRANS_OP(trans) \
  (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T

// One XtGemmWorker per logical partition on a GPU.
// Each worker owns a green context with a fraction of the GPU's SMs,
// a CUDA stream created within that green context, and a cublas handle
// bound to that stream. The API is cublasXt-style: caller passes host
// pointers, the worker handles H2D/D2H transfers internally.
class XtGemmWorker : public WorkerBase,
                     public std::enable_shared_from_this<XtGemmWorker> {
 public:
  // gpu_id: physical GPU index
  // num_partitions: how many workers share this GPU (for SM partitioning)
  // partition_idx: this worker's partition index [0, num_partitions)
  // buffer_size: CachingAllocator pool size
  XtGemmWorker(int gpu_id, int num_partitions, int partition_idx,
               size_t buffer_size);
  ~XtGemmWorker();

  DELETE_COPY_AND_ASSIGN(XtGemmWorker);

  // cublasXt-style API: host pointers in, host pointers out
  // Internally: H2D copy -> cublasSgemm on green ctx stream -> D2H copy
  //             -> stream sync
  void RunXtGemm(std::shared_ptr<GemmArgs> args);

  cudaStream_t GetStream() const { return stream_; }
  int GetGpuId() const { return gpu_id_; }
  int GetPartitionIdx() const { return partition_idx_; }

 private:
  void Run() override;  // Thread entry: set device, green ctx, cublas, alloc
  void InitGreenContext();
  void DestroyGreenContext();

  int gpu_id_;
  int num_partitions_;
  int partition_idx_;
  size_t buffer_size_;

  // CUDA green context resources (driver API)
  CUdevice cu_device_ = 0;
  CUdevResource partition_resource_ = {};
  CUdevResourceDesc resource_desc_ = nullptr;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext cuda_ctx_ = nullptr;

  // Stream + cublas within the green context
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
  std::unique_ptr<CachingAllocator> allocator_;
};

// Pool of XtGemmWorkers, potentially multiple per GPU
class XtGemmWorkerPool : public noncopyable {
 public:
  // workers_per_gpu: number of green context partitions per GPU
  // buffer_size: CachingAllocator pool size per worker
  // policy: scheduling policy for task distribution
  XtGemmWorkerPool(int workers_per_gpu, size_t buffer_size,
                   SchedulingPolicyType policy);
  ~XtGemmWorkerPool();

  DELETE_COPY_AND_ASSIGN(XtGemmWorkerPool);

  void EnqueueGemm(const std::string& task_id,
                   std::shared_ptr<GemmArgs> args);
  void WaitAll();
  void Wait(const std::string& task_id);

  size_t GetWorkerCount() const { return workers_.size(); }

 private:
  std::vector<std::shared_ptr<XtGemmWorker>> workers_;
  std::unique_ptr<SchedulingPolicy> scheduler_;
};
