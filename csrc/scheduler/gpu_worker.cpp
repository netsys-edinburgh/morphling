#include "gpu_worker.h"

#include "utils/logger.h"

void GPUWorker::RunCublasGemm(const GemmArgsPtr& args) {
  cublasHandle_t handle;
  CHECK_CUBLAS_ERROR(cublasCreate(&handle));

  CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  if (args->group_size > 1) {
    LOG_FATAL("Grouped gemm not supported yet");
  }

  // allocate device memory for matrices A, B, and C
  auto [size_a, size_b, size_c] = CalculateTaskSizes(args);
  auto* d_A = (float*)allocator_->Allocate(size_a);
  auto* d_B = (float*)allocator_->Allocate(size_b);
  auto* d_C = (float*)allocator_->Allocate(size_c);

  // validate leading dimensions
  CUDA_MEMCPY_LOOP(*args->transa, d_A, *args->a, *args->lda, *args->m, *args->k,
                   cudaMemcpyHostToDevice);
  CUDA_MEMCPY_LOOP(*args->transb, d_B, *args->b, *args->ldb, *args->n, *args->k,
                   cudaMemcpyHostToDevice);

  // Set cuBLAS operation modes
  cublasOperation_t transa = (*args->transa == 'N' || *args->transa == 'n')
                                 ? CUBLAS_OP_N
                                 : CUBLAS_OP_T;
  cublasOperation_t transb = (*args->transb == 'N' || *args->transb == 'n')
                                 ? CUBLAS_OP_N
                                 : CUBLAS_OP_T;

  // Perform matrix multiplication
  CHECK_CUBLAS_ERROR(cublasSgemm_v2(handle, transa, transb, *args->m, *args->n,
                                    *args->k, args->alpha, d_A, *args->lda, d_B,
                                    *args->ldb, args->beta, d_C, *args->ldc));
  CUDA_MEMCPY_LOOP('N', *args->c, d_C, *args->ldc, *args->m, *args->n,
                   cudaMemcpyDeviceToHost);
}

void GPUWorker::EnqueueGemm(const GemmArgsPtr& args) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto func = std::bind(&GPUWorker::RunCublasGemm, this, std::cref(args));
    tasks_.emplace_back(func);
  }
  cv_.notify_one();
}

void GPUWorker::Run() {
  // set cuda context in thread
  cudaSetDevice(gpu_id_);
  CHECK_CUBLAS_ERROR(cublasCreate(&handle_));
  CHECK_CUBLAS_ERROR(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));

  allocator_ = std::make_unique<CachingAllocator>(buffer_size_,
                                                  MemoryType::CUDA, gpu_id_);
  cudaStreamCreateWithFlags(&in_stream_, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&out_stream_, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&comp_stream_, cudaStreamNonBlocking);

  // call the parent Run()
  WorkerBase::Run();
}

GPUWorker::GPUWorker(int gpu_id, size_t size)
    : gpu_id_(gpu_id),
      buffer_size_(size),
      in_stream_(nullptr),
      out_stream_(nullptr),
      comp_stream_(nullptr) {
  worker_ = std::thread([this] { Run(); });
  LOG_INFO("GPUWorker created on {} with buffer size {}GB", gpu_id,
           int(size / GB));
}

GPUWorker::~GPUWorker() {
  cudaStreamDestroy(in_stream_);
  cudaStreamDestroy(out_stream_);
  cudaStreamDestroy(comp_stream_);
  LOG_INFO("GPUWorker destroyed on {}", gpu_id_);
}

GPUWorkerPool::GPUWorkerPool(size_t size, SchedulingPolicyType policy)
    : size_(size), policy_(policy) {
  int device_count;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  for (int i = 0; i < device_count; i++) {
    workers_.emplace_back(new GPUWorker(i, size));
  }

  switch (policy) {
    case SchedulingPolicyType::kRoundRobinGemm:
      scheduler_ = std::make_unique<RoundRobinGemmPolicy>(device_count);
      break;
    default:
      LOG_FATAL("Unsupported scheduling policy type: {}", policy);
  }

  LOG_INFO("GPUWorkerPool created with {} workers, policy: {}", device_count,
           policy);
}

GPUWorkerPool::~GPUWorkerPool() {
  for (auto& worker : workers_) {
    worker->Stop();
  }
}

void GPUWorkerPool::EnqueueGemmWithPolicy(const GemmArgsPtr& args) {
  auto [gpu_id, priority] = scheduler_->Schedule(args.get());
  workers_[gpu_id]->EnqueueGemm(args);
}
