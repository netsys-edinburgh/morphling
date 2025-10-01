#include "gpu_worker.h"

#include "utils/logger.h"

void GPUWorker::RunCublasGemm(std::shared_ptr<GemmArgs> args) {
  // cublasHandle_t handle;
  // CHECK_CUBLAS_ERROR(cublasCreate(&handle));

  // LOG_DEBUG("Created cuBLAS handle");

  // CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  // if (args->group_size > 1) {
  //   LOG_FATAL("Grouped gemm not supported yet");
  // }

  LOG_DEBUG << "Running cublasSgemm_v2 on GPU " << gpu_id_ << " with "
            << args->DebugString();

  // allocate device memory for matrices A, B, and C
  auto [size_a, size_b, size_c] = CalculateTaskSizes(args.get());
  LOG_DEBUG << "Allocating device memory for matrices A: " << size_a
            << ", B: " << size_b << ", and C: " << size_c;
  auto* d_A = (float*)allocator_->Allocate(size_a);
  auto* d_B = (float*)allocator_->Allocate(size_b);
  auto* d_C = (float*)allocator_->Allocate(size_c);

  LOG_DEBUG << "Allocated device memory for matrices A, B, and C";

  // validate leading dimensions
  CUDA_MEMCPY_LOOP(args->transa[0], d_A, args->a[0], args->lda[0], args->m[0],
                   args->k[0], cudaMemcpyHostToDevice);
  CUDA_MEMCPY_LOOP(args->transb[0], d_B, args->b[0], args->ldb[0], args->k[0],
                   args->n[0], cudaMemcpyHostToDevice);

  LOG_DEBUG << "Copied matrices A and B to device";

  // Set cuBLAS operation modes
  cublasOperation_t transa = CUDA_TRANS_OP(args->transa[0]);
  cublasOperation_t transb = CUDA_TRANS_OP(args->transb[0]);

  LOG_DEBUG << "Set cuBLAS operation modes";

  // Perform matrix multiplication
  CHECK_CUBLAS_ERROR(cublasSgemm_v2(
      handle_, transa, transb, args->m[0], args->n[0], args->k[0], args->alpha,
      d_A, args->lda[0], d_B, args->ldb[0], args->beta, d_C, args->ldc[0]));
  LOG_DEBUG(
      "cublasSgemm_v2: m: {}, n: {}, k: {}, alpha: {}, beta: {}, lda: {}, ldb: "
      "{}, ldc: {}",
      args->m[0], args->n[0], args->k[0], args->alpha[0], args->beta[0],
      args->lda[0], args->ldb[0], args->ldc[0]);
  CUDA_MEMCPY_LOOP('N', args->c[0], d_C, args->ldc[0], args->m[0], args->n[0],
                   cudaMemcpyDeviceToHost);
  LOG_DEBUG << "Copied matrix C back to host";
  allocator_->Free(d_A);
  allocator_->Free(d_B);
  allocator_->Free(d_C);
  LOG_DEBUG << "Freed device memory for matrices A, B, and C";
}

// void GPUWorker::EnqueueGemm(std::shared_ptr<GemmArgs> args) {
//   {
//     std::lock_guard<std::mutex> guard(mutex_);
//     auto func = std::bind(&GPUWorker::RunCublasGemm, this, args);
//     tasks_.emplace_back(func);
//   }
//   cv_.notify_one();
//   LOG_DEBUG("Enqueued gemm task on GPU {}", gpu_id_);
// }

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
  LOG_INFO << "GPUWorker created on " << gpu_id << " with buffer size "
           << int(size / GB) << "GB";
}

GPUWorker::~GPUWorker() {
  cudaStreamDestroy(in_stream_);
  cudaStreamDestroy(out_stream_);
  cudaStreamDestroy(comp_stream_);
  LOG_INFO << "GPUWorker destroyed on " << gpu_id_;
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
      LOG_FATAL << "Unsupported scheduling policy type: "
                << SchedulingPolicyTypeToString(policy);
  }

  LOG_INFO << "GPUWorkerPool created with " << device_count
           << " workers, policy: " << SchedulingPolicyTypeToString(policy);
}

GPUWorkerPool::~GPUWorkerPool() {
  for (auto& worker : workers_) {
    worker->Stop();
  }
}

void GPUWorkerPool::EnqueueGemmWithPolicy(const std::string& task_id,
                                          std::shared_ptr<GemmArgs> args) {
  auto [gpu_id, priority] = scheduler_->Schedule(args.get());
  auto task = std::bind(&GPUWorker::RunCublasGemm, workers_[gpu_id], args);
  workers_[gpu_id]->AddTask(task_id, task);
}

void GPUWorkerPool::WaitAll() {
  for (auto& worker : workers_) {
    worker->WaitTaskDone();
  }
}

void GPUWorkerPool::Wait(const std::string& task_id) {
  for (auto& worker : workers_) {
    worker->WaitTaskDone(task_id);
  }
}
