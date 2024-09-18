#pragma once

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
    CHECK_CUDA_ERROR(cudaMemcpy(dst + col * ld, src + col * ld,              \
                                num_elements * sizeof(float),                \
                                mode) != cudaSuccess)                        \
  }

class GPUWorker : public WorkerBase {
 public:
  explicit GPUWorker(int gpu_id, size_t size);
  ~GPUWorker();

  cudaStream_t GetInStream() const { return in_stream_; }
  cudaStream_t GetOutStream() const { return out_stream_; }
  cudaStream_t GetCompStream() const { return comp_stream_; }

  DELETE_COPY_AND_ASSIGN(GPUWorker);

  void EnqueueGemm(const GemmArgs& args);

 private:
  void Run() override;
  void RunCublasGemm(const GemmArgs& args);

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

  void EnqueueGemmWithPolicy(const GemmArgs& args);

 private:
  std::vector<std::unique_ptr<GPUWorker>> workers_;
  SchedulingPolicyType policy_;
  size_t size_;
  std::unique_ptr<SchedulingPolicy> scheduler_;
};