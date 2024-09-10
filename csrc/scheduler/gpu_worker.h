#pragma once

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include "worker_base.h"

class GPUWorker : public WorkerBase {
 public:
  explicit GPUWorker(int gpu_id, size_t size);
  ~GPUWorker();

  void Run() override;

  cudaStream_t GetInStream() const { return in_stream_; }
  cudaStream_t GetOutStream() const { return out_stream_; }
  cudaStream_t GetCompStream() const { return comp_stream_; }

  DELETE_COPY_AND_ASSIGN(GPUWorker);

 private:
  void* buffer_;
  int gpu_id_;
  size_t buffer_size_;
  cudaStream_t in_stream_;
  cudaStream_t out_stream_;
  cudaStream_t comp_stream_;
};