#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include "intercept/interceptor.h"
#include "memory/caching_allocator.h"
#include "scheduling_policy.h"
#include "worker_base.h"

#define CUDA_MEMCPY_LOOP(trans, dst, src, ld, m, k, mode)                    \
  for (int col = 0; col < ((trans == 'N' || trans == 'n') ? k : m); col++) { \
    size_t num_elements = ((trans == 'N' || trans == 'n') ? m : k);          \
    size_t offset = col * ld;                                                \
    CHECK_CUDA_ERROR(cudaMemcpy(dst + offset, src + offset,                  \
                                num_elements * sizeof(float), mode))         \
  }

#define CUDA_TRANS_OP(trans) \
  (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T

class GPUWorker : public WorkerBase, std::enable_shared_from_this<GPUWorker> {
 public:
  explicit GPUWorker(int gpu_id, size_t size);
  ~GPUWorker();

  cudaStream_t GetInStream() const { return in_stream_; }
  cudaStream_t GetOutStream() const { return out_stream_; }
  cudaStream_t GetCompStream() const { return comp_stream_; }

  DELETE_COPY_AND_ASSIGN(GPUWorker);

  // void EnqueueGemm(std::shared_ptr<GemmArgs> args);
  // void Wait();
  void RunCublasGemm(std::shared_ptr<GemmArgs> args);

 private:
  void Run() override;

  cublasOperation_t CudaTransConvert(char trans);

 private:
  std::unique_ptr<CachingAllocator> allocator_;
  int gpu_id_;
  size_t buffer_size_;
  cudaStream_t in_stream_;
  cudaStream_t out_stream_;
  cudaStream_t comp_stream_;
  cublasHandle_t handle_;

  std::deque<cudaEvent_t> in_events_;
  std::deque<cudaEvent_t> out_events_;
  std::deque<cudaEvent_t> comp_events_;
};

// create a pool of GPU workers, one worker for each GPU
class GPUWorkerPool : public noncopyable {
 public:
  explicit GPUWorkerPool(size_t size, SchedulingPolicyType policy);
  ~GPUWorkerPool();

  DELETE_COPY_AND_ASSIGN(GPUWorkerPool);

  void EnqueueGemmWithPolicy(std::shared_ptr<GemmArgs> args);
  void WaitAll();

 private:
  std::vector<std::shared_ptr<GPUWorker>> workers_;
  SchedulingPolicyType policy_;
  size_t size_;
  std::unique_ptr<SchedulingPolicy> scheduler_;
};