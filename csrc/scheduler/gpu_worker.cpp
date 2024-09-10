#include "gpu_worker.h"

#include "utils/logger.h"

void GPUWorker::Run() {
  // set cuda context in thread
  cudaSetDevice(gpu_id_);

  cudaMalloc(&buffer_, buffer_size_);
  cudaStreamCreateWithFlags(&in_stream_, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&out_stream_, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&comp_stream_, cudaStreamNonBlocking);

  // call the parent Run()
  WorkerBase::Run();
}

GPUWorker::GPUWorker(int gpu_id, size_t size)
    : buffer_(nullptr),
      gpu_id_(gpu_id),
      buffer_size_(size),
      in_stream_(nullptr),
      out_stream_(nullptr),
      comp_stream_(nullptr) {
  worker_ = std::thread([this] { Run(); });
  LOG_INFO("GPUWorker created on {} with buffer size {}GB", gpu_id,
           int(size / GB));
}

GPUWorker::~GPUWorker() {
  cudaFree(buffer_);
  cudaStreamDestroy(in_stream_);
  cudaStreamDestroy(out_stream_);
  cudaStreamDestroy(comp_stream_);
  LOG_INFO("GPUWorker destroyed on {}", gpu_id_);
}