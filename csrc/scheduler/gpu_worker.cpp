#include "gpu_worker.h"

#include "utils/logger.h"

// ---------------------------------------------------------------------------
// XtGemmWorker
// ---------------------------------------------------------------------------

XtGemmWorker::XtGemmWorker(int gpu_id, int num_partitions, int partition_idx,
                           size_t buffer_size)
    : gpu_id_(gpu_id),
      num_partitions_(num_partitions),
      partition_idx_(partition_idx),
      buffer_size_(buffer_size) {
  worker_ = std::thread([this] { Run(); });
  LOG_INFO << "XtGemmWorker created: gpu=" << gpu_id_
           << " partition=" << partition_idx_ << "/" << num_partitions_
           << " buffer=" << int(buffer_size_ / GB) << "GB";
}

XtGemmWorker::~XtGemmWorker() {
  if (cublas_handle_) {
    cublasDestroy(cublas_handle_);
    cublas_handle_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  DestroyGreenContext();
  LOG_INFO << "XtGemmWorker destroyed: gpu=" << gpu_id_
           << " partition=" << partition_idx_;
}

void XtGemmWorker::InitGreenContext() {
  CHECK_CU_RESULT(cuDeviceGet(&cu_device_, gpu_id_));

  // Step 1: Get the device's full SM resource
  CUdevResource device_sm_resource = {};
  CHECK_CU_RESULT(cuDeviceGetDevResource(
      cu_device_, &device_sm_resource, CU_DEV_RESOURCE_TYPE_SM));

  // Query SM count for logging
  int sm_count = 0;
  CHECK_CU_RESULT(cuDeviceGetAttribute(
      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_device_));
  LOG_INFO << "GPU " << gpu_id_ << " has " << sm_count << " SMs, "
           << "partitioning " << num_partitions_ << " ways for partition "
           << partition_idx_;

  // Step 2: Split SM resource into num_partitions_ equal groups
  unsigned int min_sm_per_partition =
      static_cast<unsigned int>(sm_count / num_partitions_);
  LOG_FATAL_IF(min_sm_per_partition == 0)
      << "Not enough SMs (" << sm_count << ") for " << num_partitions_
      << " partitions";

  // First query: how many groups will be created?
  unsigned int nb_groups = 0;
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      nullptr, &nb_groups, &device_sm_resource, nullptr,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
      min_sm_per_partition));
  LOG_FATAL_IF(nb_groups < static_cast<unsigned int>(num_partitions_))
      << "Cannot create " << num_partitions_ << " partitions, "
      << "only " << nb_groups << " possible with minCount="
      << min_sm_per_partition;

  // Second call: actually split into groups
  std::vector<CUdevResource> split_results(nb_groups);
  CUdevResource remaining = {};
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      split_results.data(), &nb_groups, &device_sm_resource, &remaining,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
      min_sm_per_partition));

  // Pick this partition's resource
  partition_resource_ = split_results[partition_idx_];

  // Step 3: Generate resource descriptor from the partition
  CHECK_CU_RESULT(cuDevResourceGenerateDesc(
      &resource_desc_, &partition_resource_, 1));

  // Step 4: Create green context
  CHECK_CU_RESULT(cuGreenCtxCreate(
      &green_ctx_, resource_desc_, cu_device_,
      CU_GREEN_CTX_DEFAULT_STREAM));
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&cuda_ctx_, green_ctx_));
  CHECK_CU_RESULT(cuCtxSetCurrent(cuda_ctx_));

  LOG_INFO << "Green context created: gpu=" << gpu_id_
           << " partition=" << partition_idx_
           << " min_sm_count=" << min_sm_per_partition;
}

void XtGemmWorker::DestroyGreenContext() {
  if (cuda_ctx_) {
    cuCtxDestroy(cuda_ctx_);
    cuda_ctx_ = nullptr;
  }
  if (green_ctx_) {
    cuGreenCtxDestroy(green_ctx_);
    green_ctx_ = nullptr;
  }
}

void XtGemmWorker::Run() {
  // Set device for this thread
  cudaSetDevice(gpu_id_);

  // Create green context with SM partitioning
  InitGreenContext();

  // Create stream within the green context (driver API)
  // Note: CU_STREAM_NON_BLOCKING is required by cuGreenCtxStreamCreate
  CUstream cu_stream = nullptr;
  CHECK_CU_RESULT(
      cuGreenCtxStreamCreate(&cu_stream, green_ctx_,
                             CU_STREAM_NON_BLOCKING, 0));
  stream_ = cu_stream;

  // Create cublas handle bound to this stream
  CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle_));
  CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle_, stream_));
  CHECK_CUBLAS_ERROR(
      cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH));

  // Create per-worker caching allocator for device memory
  allocator_ = std::make_unique<CachingAllocator>(
      buffer_size_, MemoryType::CUDA, gpu_id_);

  LOG_INFO << "XtGemmWorker ready: gpu=" << gpu_id_
           << " partition=" << partition_idx_;

  // Enter the WorkerBase task loop
  WorkerBase::Run();
}

void XtGemmWorker::RunXtGemm(std::shared_ptr<GemmArgs> args) {
  LOG_DEBUG << "RunXtGemm on gpu=" << gpu_id_
            << " partition=" << partition_idx_ << " "
            << args->DebugString();

  // Calculate buffer sizes for A, B, C
  auto [size_a, size_b, size_c] = CalculateTaskSizes(args.get());
  LOG_DEBUG << "Allocating device memory: A=" << size_a << " B=" << size_b
            << " C=" << size_c;

  // Allocate device buffers via per-worker allocator
  auto* d_A = static_cast<float*>(allocator_->Allocate(size_a));
  auto* d_B = static_cast<float*>(allocator_->Allocate(size_b));
  auto* d_C = static_cast<float*>(allocator_->Allocate(size_c));

  // H2D: copy input matrices to device asynchronously on this stream
  CUDA_MEMCPY_ASYNC_LOOP(args->transa[0], d_A, args->a[0], args->lda[0],
                         args->m[0], args->k[0],
                         cudaMemcpyHostToDevice, stream_);
  CUDA_MEMCPY_ASYNC_LOOP(args->transb[0], d_B, args->b[0], args->ldb[0],
                         args->k[0], args->n[0],
                         cudaMemcpyHostToDevice, stream_);
  // C must also be copied when beta != 0 (cublas reads d_C for beta*C term)
  if (args->beta[0] != 0.0f) {
    CUDA_MEMCPY_ASYNC_LOOP('N', d_C, args->c[0], args->ldc[0],
                           args->m[0], args->n[0],
                           cudaMemcpyHostToDevice, stream_);
  }

  // Set cublas operation modes
  cublasOperation_t transa = CUDA_TRANS_OP(args->transa[0]);
  cublasOperation_t transb = CUDA_TRANS_OP(args->transb[0]);

  // Execute GEMM on this worker's stream (restricted to green ctx SMs)
  CHECK_CUBLAS_ERROR(cublasSgemm_v2(
      cublas_handle_, transa, transb, args->m[0], args->n[0], args->k[0],
      args->alpha, d_A, args->lda[0], d_B, args->ldb[0], args->beta, d_C,
      args->ldc[0]));

  // D2H: copy result back to host asynchronously
  CUDA_MEMCPY_ASYNC_LOOP('N', args->c[0], d_C, args->ldc[0], args->m[0],
                         args->n[0], cudaMemcpyDeviceToHost, stream_);

  // Synchronize only this worker's stream (no device-wide sync)
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  // Free device buffers back to pool
  allocator_->Free(d_A);
  allocator_->Free(d_B);
  allocator_->Free(d_C);

  LOG_DEBUG << "RunXtGemm completed on gpu=" << gpu_id_
            << " partition=" << partition_idx_;
}

// ---------------------------------------------------------------------------
// XtGemmWorkerPool
// ---------------------------------------------------------------------------

XtGemmWorkerPool::XtGemmWorkerPool(int workers_per_gpu, size_t buffer_size,
                                   SchedulingPolicyType policy) {
  int device_count = 0;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));

  int total_workers = workers_per_gpu * device_count;

  for (int gpu = 0; gpu < device_count; gpu++) {
    for (int p = 0; p < workers_per_gpu; p++) {
      workers_.emplace_back(std::make_shared<XtGemmWorker>(
          gpu, workers_per_gpu, p, buffer_size));
    }
  }

  switch (policy) {
    case SchedulingPolicyType::kRoundRobinGemm:
      scheduler_ =
          std::make_unique<RoundRobinGemmPolicy>(total_workers);
      break;
    default:
      LOG_FATAL << "Unsupported scheduling policy: "
                << SchedulingPolicyTypeToString(policy);
  }

  LOG_INFO << "XtGemmWorkerPool created: " << total_workers
           << " workers (" << workers_per_gpu << " per GPU, "
           << device_count << " GPUs), policy="
           << SchedulingPolicyTypeToString(policy);
}

XtGemmWorkerPool::~XtGemmWorkerPool() {
  for (auto& worker : workers_) {
    worker->Stop();
  }
}

void XtGemmWorkerPool::EnqueueGemm(const std::string& task_id,
                                   std::shared_ptr<GemmArgs> args) {
  auto [worker_idx, priority] = scheduler_->Schedule(args.get());
  auto task = std::bind(&XtGemmWorker::RunXtGemm,
                        workers_[worker_idx], args);
  workers_[worker_idx]->AddTask(task_id, std::move(task));
}

void XtGemmWorkerPool::WaitAll() {
  for (auto& worker : workers_) {
    worker->WaitTaskDone();
  }
}

void XtGemmWorkerPool::Wait(const std::string& task_id) {
  for (auto& worker : workers_) {
    worker->WaitTaskDone(task_id);
  }
}
