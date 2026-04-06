#include "gpu_worker.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include "core/logger.h"
#include "sliding_window_tracker.h"

static constexpr size_t kDefaultPoolBytes = 256ull * 1024 * 1024;

static size_t ResolvePoolBytes(size_t buffer_size) {
  if (const char* v = std::getenv("MORPHLING_WORKER_POOL_SIZE")) {
    return std::stoull(v);
  }
  if (buffer_size > 0) return buffer_size;
  return kDefaultPoolBytes;
}

ContextSlot::~ContextSlot() {
  if (cuda_ctx) {
    cuCtxSetCurrent(cuda_ctx);
  }
  if (cublas_handle) {
    cublasDestroy(cublas_handle);
    cublas_handle = nullptr;
  }
  if (xt_handle) {
    cublasXtDestroy(xt_handle);
    xt_handle = nullptr;
  }
  if (copy_stream) {
    cudaStreamDestroy(copy_stream);
    copy_stream = nullptr;
  }
  if (compute_stream) {
    cudaStreamDestroy(compute_stream);
    compute_stream = nullptr;
  }
  stream = nullptr;
  if (cuda_ctx) {
    cuCtxDestroy(cuda_ctx);
    cuda_ctx = nullptr;
  }
  if (green_ctx) {
    cuGreenCtxDestroy(green_ctx);
    green_ctx = nullptr;
  }
}

ContextSlot::ContextSlot(ContextSlot&& other) noexcept
    : sm_count(other.sm_count),
      resource_desc(other.resource_desc),
      green_ctx(other.green_ctx),
      cuda_ctx(other.cuda_ctx),
      compute_stream(other.compute_stream),
      copy_stream(other.copy_stream),
      stream(other.stream),
      cublas_handle(other.cublas_handle),
      xt_handle(other.xt_handle) {
  other.sm_count = 0;
  other.resource_desc = nullptr;
  other.green_ctx = nullptr;
  other.cuda_ctx = nullptr;
  other.compute_stream = nullptr;
  other.copy_stream = nullptr;
  other.stream = nullptr;
  other.cublas_handle = nullptr;
  other.xt_handle = nullptr;
}

ContextSlot& ContextSlot::operator=(ContextSlot&& other) noexcept {
  if (this != &other) {
    this->~ContextSlot();
    sm_count = other.sm_count;
    resource_desc = other.resource_desc;
    green_ctx = other.green_ctx;
    cuda_ctx = other.cuda_ctx;
    compute_stream = other.compute_stream;
    copy_stream = other.copy_stream;
    stream = other.stream;
    cublas_handle = other.cublas_handle;
    xt_handle = other.xt_handle;
    other.sm_count = 0;
    other.resource_desc = nullptr;
    other.green_ctx = nullptr;
    other.cuda_ctx = nullptr;
    other.compute_stream = nullptr;
    other.copy_stream = nullptr;
    other.stream = nullptr;
    other.cublas_handle = nullptr;
    other.xt_handle = nullptr;
  }
  return *this;
}

XtGemmWorker::XtGemmWorker(int gpu_id, int num_partitions, int partition_idx,
                           size_t buffer_size)
    : gpu_id_(gpu_id),
      num_partitions_(num_partitions),
      partition_idx_(partition_idx),
      buffer_size_(buffer_size) {
  worker_ = std::thread([this] { Run(); });
  LOG_DEBUG << "XtGemmWorker created: gpu=" << gpu_id_
            << " partition=" << partition_idx_ << "/" << num_partitions_;
}

XtGemmWorker::~XtGemmWorker() {
  if (worker_.joinable()) {
    Stop();
  }

  if (active_slot_ && active_slot_->cuda_ctx) {
    CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
  }

  DestroyPipelineState();
  ReleaseDeviceBuffers();

  active_slot_ = nullptr;
  allocator_.reset();
  context_slots_.clear();
  cudaSetDevice(gpu_id_);
  LOG_DEBUG << "XtGemmWorker destroyed: gpu=" << gpu_id_
            << " partition=" << partition_idx_;
}

void XtGemmWorker::InitAllContexts() {
  CHECK_CU_RESULT(cuDeviceGet(&cu_device_, gpu_id_));

  int sm_count = 0;
  CHECK_CU_RESULT(cuDeviceGetAttribute(
      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_device_));
  LOG_INFO << "GPU " << gpu_id_ << " has " << sm_count << " SMs, "
           << "partitioning " << num_partitions_ << " ways for partition "
           << partition_idx_;

  CUdevResource device_sm_resource = {};
  CHECK_CU_RESULT(cuDeviceGetDevResource(cu_device_, &device_sm_resource,
                                         CU_DEV_RESOURCE_TYPE_SM));

  unsigned int nb_groups = 0;
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      nullptr, &nb_groups, &device_sm_resource, nullptr,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));
  LOG_FATAL_IF(nb_groups == 0) << "Cannot split SMs into groups";

  sm_groups_.resize(nb_groups);
  CUdevResource remaining = {};
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      sm_groups_.data(), &nb_groups, &device_sm_resource, &remaining,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

  sm_step_ = static_cast<int>(sm_groups_[0].sm.smCount);
  LOG_INFO << "SM step size: " << sm_step_ << " (" << nb_groups
           << " groups total)";

  unsigned int groups_per_partition = nb_groups / num_partitions_;
  LOG_FATAL_IF(groups_per_partition == 0)
      << "Not enough SM groups (" << nb_groups << ") for " << num_partitions_
      << " partitions";

  partition_sm_count_ = static_cast<int>(groups_per_partition) * sm_step_;
  unsigned int base_offset = partition_idx_ * groups_per_partition;

  LOG_INFO << "Partition " << partition_idx_ << ": " << groups_per_partition
           << " groups, " << partition_sm_count_
           << " SMs (offset=" << base_offset << ")";

  for (unsigned int n = 1; n <= groups_per_partition; n++) {
    int slot_sm_count = static_cast<int>(n) * sm_step_;
    auto slot = CreateContextSlot(&sm_groups_[base_offset], static_cast<int>(n),
                                  slot_sm_count);
    context_slots_.emplace(slot_sm_count, std::move(slot));
    LOG_INFO << "Created context slot: " << slot_sm_count << " SMs";
  }

  active_slot_ = &context_slots_.at(partition_sm_count_);
  CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));

  size_t pool_bytes = ResolvePoolBytes(buffer_size_);
  allocator_ =
      std::make_unique<CachingAllocator>(pool_bytes, MemoryType::CUDA, gpu_id_);
  LOG_INFO << "XtGemmWorker gpu=" << gpu_id_ << " partition=" << partition_idx_
           << " allocator initialized: " << pool_bytes << " bytes";

  LOG_INFO << "XtGemmWorker initialized: " << context_slots_.size()
           << " context slots, active=" << partition_sm_count_ << " SMs";
}

ContextSlot XtGemmWorker::CreateContextSlot(CUdevResource* groups,
                                            int num_groups, int sm_count) {
  ContextSlot slot;
  slot.sm_count = sm_count;

  CHECK_CU_RESULT(
      cuDevResourceGenerateDesc(&slot.resource_desc, groups, num_groups));

  CHECK_CU_RESULT(cuGreenCtxCreate(&slot.green_ctx, slot.resource_desc,
                                   cu_device_, CU_GREEN_CTX_DEFAULT_STREAM));
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&slot.cuda_ctx, slot.green_ctx));
  CHECK_CU_RESULT(cuCtxSetCurrent(slot.cuda_ctx));

  CUstream compute_cu_stream = nullptr;
  CHECK_CU_RESULT(cuGreenCtxStreamCreate(&compute_cu_stream, slot.green_ctx,
                                         CU_STREAM_NON_BLOCKING, 0));
  slot.compute_stream = compute_cu_stream;
  slot.stream = slot.compute_stream;

  CUstream copy_cu_stream = nullptr;
  CHECK_CU_RESULT(cuGreenCtxStreamCreate(&copy_cu_stream, slot.green_ctx,
                                         CU_STREAM_NON_BLOCKING, 0));
  slot.copy_stream = copy_cu_stream;

  CHECK_CUBLAS_ERROR(cublasCreate(&slot.cublas_handle));
  CHECK_CUBLAS_ERROR(cublasSetStream(slot.cublas_handle, slot.compute_stream));

  CHECK_CUBLAS_ERROR(cublasXtCreate(&slot.xt_handle));
  int device_id = gpu_id_;
  CHECK_CUBLAS_ERROR(cublasXtDeviceSelect(slot.xt_handle, 1, &device_id));

  return slot;
}

bool XtGemmWorker::SwitchContext(int num_sms) {
  auto it = context_slots_.find(num_sms);
  if (it == context_slots_.end()) return false;
  active_slot_ = &it->second;
  CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
  if (active_slot_->cublas_handle) {
    CHECK_CUBLAS_ERROR(cublasSetStream(active_slot_->cublas_handle,
                                       active_slot_->compute_stream));
  }
  return true;
}

int XtGemmWorker::GetActiveSmCount() const {
  return active_slot_ ? active_slot_->sm_count : 0;
}

std::vector<int> XtGemmWorker::GetAvailableSmCounts() const {
  std::vector<int> counts;
  counts.reserve(context_slots_.size());
  for (const auto& [sm_count, _] : context_slots_) {
    counts.push_back(sm_count);
  }
  std::sort(counts.begin(), counts.end());
  return counts;
}

void XtGemmWorker::Run() {
  CHECK_CUDA_ERROR(cudaSetDevice(gpu_id_));
  InitAllContexts();

  LOG_INFO << "XtGemmWorker ready: gpu=" << gpu_id_
           << " partition=" << partition_idx_;

  if (GetPipelineMode() == WorkerPipelineMode::kPipelined) {
    RunPipelined();
    return;
  }
  WorkerBase::Run();
}

void XtGemmWorker::RunXtGemm(std::shared_ptr<GemmArgs> args) {
  LOG_DEBUG << "RunXtGemm on gpu=" << gpu_id_ << " partition=" << partition_idx_
            << " sms=" << (active_slot_ ? active_slot_->sm_count : 0) << " "
            << args->DebugString();

  cublasXtHandle_t handle = active_slot_->xt_handle;

  cublasOperation_t transa = CUDA_TRANS_OP(args->transa[0]);
  cublasOperation_t transb = CUDA_TRANS_OP(args->transb[0]);

  CHECK_CUBLAS_ERROR(
      cublasXtSgemm(handle, transa, transb, args->m[0], args->n[0], args->k[0],
                    args->alpha, args->a[0], args->lda[0], args->b[0],
                    args->ldb[0], args->beta, args->c[0], args->ldc[0]));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(active_slot_->stream));

  LOG_DEBUG << "RunXtGemm completed on gpu=" << gpu_id_
            << " partition=" << partition_idx_;
}

void XtGemmWorker::EnsureDeviceBuffers(size_t size_a, size_t size_b,
                                       size_t size_c) {
  LOG_FATAL_IF(!allocator_) << "Allocator not initialized";

  if (size_a > d_a_cap_) {
    if (d_a_) allocator_->Free(d_a_);
    d_a_ = static_cast<float*>(allocator_->Allocate(size_a));
    LOG_FATAL_IF(!d_a_) << "Failed to allocate device buffer A: " << size_a;
    d_a_cap_ = size_a;
  }
  if (size_b > d_b_cap_) {
    if (d_b_) allocator_->Free(d_b_);
    d_b_ = static_cast<float*>(allocator_->Allocate(size_b));
    LOG_FATAL_IF(!d_b_) << "Failed to allocate device buffer B: " << size_b;
    d_b_cap_ = size_b;
  }
  if (size_c > d_c_cap_) {
    if (d_c_) allocator_->Free(d_c_);
    d_c_ = static_cast<float*>(allocator_->Allocate(size_c));
    LOG_FATAL_IF(!d_c_) << "Failed to allocate device buffer C: " << size_c;
    d_c_cap_ = size_c;
  }

  CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
}

void XtGemmWorker::ReleaseDeviceBuffers() {
  if (!allocator_) {
    d_a_ = nullptr;
    d_b_ = nullptr;
    d_c_ = nullptr;
    d_a_cap_ = 0;
    d_b_cap_ = 0;
    d_c_cap_ = 0;
    return;
  }

  if (active_slot_) {
    CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
  }

  if (d_a_) {
    allocator_->Free(d_a_);
    d_a_ = nullptr;
    d_a_cap_ = 0;
  }
  if (d_b_) {
    allocator_->Free(d_b_);
    d_b_ = nullptr;
    d_b_cap_ = 0;
  }
  if (d_c_) {
    allocator_->Free(d_c_);
    d_c_ = nullptr;
    d_c_cap_ = 0;
  }
}

void XtGemmWorker::RunGemm(std::shared_ptr<GemmArgs> args) {
  LOG_DEBUG << "RunGemm on gpu=" << gpu_id_ << " partition=" << partition_idx_
            << " sms=" << (active_slot_ ? active_slot_->sm_count : 0) << " "
            << args->DebugString();

  auto [size_a, size_b, size_c] = CalculateTaskSizes(args.get());
  EnsureDeviceBuffers(size_a, size_b, size_c);

  cudaStream_t copy_stream = active_slot_->copy_stream;
  cudaStream_t compute_stream = active_slot_->compute_stream;

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a_, args->a[0], size_a,
                                   cudaMemcpyHostToDevice, copy_stream));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b_, args->b[0], size_b,
                                   cudaMemcpyHostToDevice, copy_stream));

  cudaEvent_t h2d_done = nullptr;
  cudaEvent_t gemm_done = nullptr;
  CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&h2d_done, cudaEventDisableTiming));
  CHECK_CUDA_ERROR(
      cudaEventCreateWithFlags(&gemm_done, cudaEventDisableTiming));

  CHECK_CUDA_ERROR(cudaEventRecord(h2d_done, copy_stream));
  CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream, h2d_done, 0));

  float alpha = args->alpha[0];
  float beta = args->beta[0];
  CHECK_CUBLAS_ERROR(
      cublasSetStream(active_slot_->cublas_handle, compute_stream));
  CHECK_CUBLAS_ERROR(
      cublasSgemm(active_slot_->cublas_handle, CUDA_TRANS_OP(args->transa[0]),
                  CUDA_TRANS_OP(args->transb[0]), args->m[0], args->n[0],
                  args->k[0], &alpha, d_a_, args->lda[0], d_b_, args->ldb[0],
                  &beta, d_c_, args->ldc[0]));

  CHECK_CUDA_ERROR(cudaEventRecord(gemm_done, compute_stream));
  CHECK_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, gemm_done, 0));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(args->c[0], d_c_, size_c,
                                   cudaMemcpyDeviceToHost, copy_stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(copy_stream));

  CHECK_CUDA_ERROR(cudaEventDestroy(h2d_done));
  CHECK_CUDA_ERROR(cudaEventDestroy(gemm_done));

  LOG_DEBUG << "RunGemm completed on gpu=" << gpu_id_
            << " partition=" << partition_idx_;
}

void XtGemmWorker::RegisterPipelinedTask(const std::string& task_id,
                                         std::shared_ptr<GemmArgs> args) {
  std::lock_guard<std::mutex> lock(mutex_);
  pipelined_task_args_[task_id] = std::move(args);
}

std::optional<std::tuple<std::string, WorkerBase::Task, TaskHandle>>
XtGemmWorker::DequeueTask(bool block) {
  if (block) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !tasks_.empty() || quit_; });
    if (tasks_.empty()) return std::nullopt;
    auto entry = std::move(tasks_.front());
    tasks_.pop_front();
    return entry;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (tasks_.empty()) return std::nullopt;
  auto entry = std::move(tasks_.front());
  tasks_.pop_front();
  return entry;
}

std::shared_ptr<GemmArgs> XtGemmWorker::TakePipelinedTaskArgs(
    const std::string& task_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pipelined_task_args_.find(task_id);
  if (it == pipelined_task_args_.end()) return nullptr;
  auto args = std::move(it->second);
  pipelined_task_args_.erase(it);
  return args;
}

void XtGemmWorker::CompleteTaskState(const std::string& task_id,
                                     const TaskHandle& state,
                                     const std::string& error) {
  TaskCallback cb;
  std::string cb_task_id;
  if (state) {
    {
      std::lock_guard<std::mutex> lk(state->mutex);
      if (!error.empty()) {
        state->error = error;
      }
      state->completed = true;
      cb = std::move(state->callback);
      cb_task_id = state->task_id;
    }
    state->cv.notify_all();
  }

  task_count_--;
  cv_.notify_all();

  if (cb) {
    cb(cb_task_id);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  task_handles_.erase(task_id);
  pipelined_task_args_.erase(task_id);
}

void XtGemmWorker::ExecuteTaskEntry(const std::string& task_id, Task&& task,
                                    const TaskHandle& state) {
  std::string error;
  try {
    task();
  } catch (const std::exception& e) {
    LOG_ERROR << "[XtGemmWorker] Task " << task_id << " threw: " << e.what();
    error = e.what();
  } catch (...) {
    LOG_ERROR << "[XtGemmWorker] Task " << task_id
              << " threw unknown exception";
    error = "unknown exception";
  }

  CompleteTaskState(task_id, state, error);
}

void XtGemmWorker::EnsurePipelineBuffers(PipelineSlot& slot, size_t size_a,
                                         size_t size_b, size_t size_c) {
  LOG_FATAL_IF(!allocator_) << "Allocator not initialized";

  if (size_a > slot.a_cap) {
    if (slot.d_a) allocator_->Free(slot.d_a);
    slot.d_a = static_cast<float*>(allocator_->Allocate(size_a));
    LOG_FATAL_IF(!slot.d_a)
        << "Failed to allocate pipeline buffer A: " << size_a;
    slot.a_cap = size_a;
  }
  if (size_b > slot.b_cap) {
    if (slot.d_b) allocator_->Free(slot.d_b);
    slot.d_b = static_cast<float*>(allocator_->Allocate(size_b));
    LOG_FATAL_IF(!slot.d_b)
        << "Failed to allocate pipeline buffer B: " << size_b;
    slot.b_cap = size_b;
  }
  if (size_c > slot.c_cap) {
    if (slot.d_c) allocator_->Free(slot.d_c);
    slot.d_c = static_cast<float*>(allocator_->Allocate(size_c));
    LOG_FATAL_IF(!slot.d_c)
        << "Failed to allocate pipeline buffer C: " << size_c;
    slot.c_cap = size_c;
  }

  CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
}

void XtGemmWorker::ReleasePipelineBuffers(PipelineSlot& slot) {
  if (!allocator_) {
    slot.d_a = nullptr;
    slot.d_b = nullptr;
    slot.d_c = nullptr;
    slot.a_cap = 0;
    slot.b_cap = 0;
    slot.c_cap = 0;
    return;
  }

  if (active_slot_) {
    CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
  }

  if (slot.d_a) {
    allocator_->Free(slot.d_a);
    slot.d_a = nullptr;
    slot.a_cap = 0;
  }
  if (slot.d_b) {
    allocator_->Free(slot.d_b);
    slot.d_b = nullptr;
    slot.b_cap = 0;
  }
  if (slot.d_c) {
    allocator_->Free(slot.d_c);
    slot.d_c = nullptr;
    slot.c_cap = 0;
  }
}

void XtGemmWorker::InitPipelineEvents() {
  for (auto& slot : pipeline_slots_) {
    if (!slot.h2d_done) {
      CHECK_CUDA_ERROR(
          cudaEventCreateWithFlags(&slot.h2d_done, cudaEventDisableTiming));
    }
    if (!slot.gemm_done) {
      CHECK_CUDA_ERROR(
          cudaEventCreateWithFlags(&slot.gemm_done, cudaEventDisableTiming));
    }
    if (!slot.d2h_done) {
      CHECK_CUDA_ERROR(
          cudaEventCreateWithFlags(&slot.d2h_done, cudaEventDisableTiming));
    }
  }
}

void XtGemmWorker::DestroyPipelineState() {
  for (auto& slot : pipeline_slots_) {
    if (slot.occupied && slot.d2h_done) {
      CHECK_CUDA_ERROR(cudaEventSynchronize(slot.d2h_done));
      CompleteTaskState(slot.task_id, slot.state);
      slot.occupied = false;
      slot.args.reset();
      slot.state.reset();
      slot.task_id.clear();
      slot.callback = nullptr;
    }

    ReleasePipelineBuffers(slot);

    if (slot.d2h_done) {
      CHECK_CUDA_ERROR(cudaEventDestroy(slot.d2h_done));
      slot.d2h_done = nullptr;
    }
    if (slot.gemm_done) {
      CHECK_CUDA_ERROR(cudaEventDestroy(slot.gemm_done));
      slot.gemm_done = nullptr;
    }
    if (slot.h2d_done) {
      CHECK_CUDA_ERROR(cudaEventDestroy(slot.h2d_done));
      slot.h2d_done = nullptr;
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  pipelined_task_args_.clear();
}

void XtGemmWorker::BeginH2D(PipelineSlot& slot) {
  auto [size_a, size_b, size_c] = CalculateTaskSizes(slot.args.get());
  EnsurePipelineBuffers(slot, size_a, size_b, size_c);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(slot.d_a, slot.args->a[0], size_a,
                                   cudaMemcpyHostToDevice,
                                   active_slot_->copy_stream));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(slot.d_b, slot.args->b[0], size_b,
                                   cudaMemcpyHostToDevice,
                                   active_slot_->copy_stream));
  CHECK_CUDA_ERROR(cudaEventRecord(slot.h2d_done, active_slot_->copy_stream));
}

void XtGemmWorker::LaunchGemm(PipelineSlot& slot) {
  CHECK_CUDA_ERROR(
      cudaStreamWaitEvent(active_slot_->compute_stream, slot.h2d_done, 0));

  float alpha = slot.args->alpha[0];
  float beta = slot.args->beta[0];
  CHECK_CUBLAS_ERROR(cublasSetStream(active_slot_->cublas_handle,
                                     active_slot_->compute_stream));
  CHECK_CUBLAS_ERROR(cublasSgemm(
      active_slot_->cublas_handle, CUDA_TRANS_OP(slot.args->transa[0]),
      CUDA_TRANS_OP(slot.args->transb[0]), slot.args->m[0], slot.args->n[0],
      slot.args->k[0], &alpha, slot.d_a, slot.args->lda[0], slot.d_b,
      slot.args->ldb[0], &beta, slot.d_c, slot.args->ldc[0]));

  CHECK_CUDA_ERROR(
      cudaEventRecord(slot.gemm_done, active_slot_->compute_stream));
}

void XtGemmWorker::BeginD2H(PipelineSlot& slot) {
  size_t size_c = std::get<2>(CalculateTaskSizes(slot.args.get()));
  CHECK_CUDA_ERROR(
      cudaStreamWaitEvent(active_slot_->copy_stream, slot.gemm_done, 0));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(slot.args->c[0], slot.d_c, size_c,
                                   cudaMemcpyDeviceToHost,
                                   active_slot_->copy_stream));
  CHECK_CUDA_ERROR(cudaEventRecord(slot.d2h_done, active_slot_->copy_stream));
}

void XtGemmWorker::CompletePipelineSlot(PipelineSlot& slot) {
  if (!slot.occupied) return;
  CHECK_CUDA_ERROR(cudaEventSynchronize(slot.d2h_done));
  CompleteTaskState(slot.task_id, slot.state);
  slot.occupied = false;
  slot.args.reset();
  slot.state.reset();
  slot.task_id.clear();
  slot.callback = nullptr;
}

void XtGemmWorker::RunPipelined() {
  InitPipelineEvents();
  active_pipeline_slot_idx_ = 0;

  while (true) {
    bool has_inflight =
        pipeline_slots_[0].occupied || pipeline_slots_[1].occupied;
    auto entry = DequeueTask(!has_inflight);

    if (entry.has_value()) {
      auto [task_id, task, state] = std::move(*entry);
      auto args = TakePipelinedTaskArgs(task_id);

      if (!args) {
        ExecuteTaskEntry(task_id, std::move(task), state);
        continue;
      }

      int next_idx = 0;
      if (!pipeline_slots_[0].occupied) {
        next_idx = 0;
      } else if (!pipeline_slots_[1].occupied) {
        next_idx = 1;
      } else {
        int drain_idx = 1 - active_pipeline_slot_idx_;
        CompletePipelineSlot(pipeline_slots_[drain_idx]);
        next_idx = drain_idx;
      }

      auto& slot = pipeline_slots_[next_idx];
      (void)task;
      slot.args = std::move(args);
      slot.task_id = task_id;
      slot.state = state;
      slot.occupied = true;

      BeginH2D(slot);
      LaunchGemm(slot);
      BeginD2H(slot);

      active_pipeline_slot_idx_ = next_idx;

      if (pipeline_slots_[0].occupied && pipeline_slots_[1].occupied) {
        int drain_idx = 1 - active_pipeline_slot_idx_;
        CompletePipelineSlot(pipeline_slots_[drain_idx]);
      }
      continue;
    }

    if (pipeline_slots_[0].occupied || pipeline_slots_[1].occupied) {
      int drain_idx = pipeline_slots_[1 - active_pipeline_slot_idx_].occupied
                          ? (1 - active_pipeline_slot_idx_)
                          : active_pipeline_slot_idx_;
      CompletePipelineSlot(pipeline_slots_[drain_idx]);
      continue;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (quit_ && tasks_.empty()) {
      break;
    }
  }

  CompletePipelineSlot(pipeline_slots_[0]);
  CompletePipelineSlot(pipeline_slots_[1]);
  DestroyPipelineState();
}

bool XtGemmWorker::LoadSmSchedule(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG_ERROR << "Cannot open SM schedule file: " << path;
    return false;
  }

  std::vector<SmScheduleEntry> schedule;
  std::string line;
  int line_num = 0;
  while (std::getline(file, line)) {
    line_num++;
    size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') continue;

    std::istringstream iss(line);
    SmScheduleEntry entry;
    if (!(iss >> entry.time_offset_us >> entry.num_sms >> entry.duration_us)) {
      LOG_ERROR << "Parse error at line " << line_num << ": " << line;
      return false;
    }

    if (!schedule.empty() &&
        entry.time_offset_us < schedule.back().time_offset_us) {
      LOG_ERROR << "Non-monotonic time_offset_us at line " << line_num << ": "
                << entry.time_offset_us << " < "
                << schedule.back().time_offset_us;
      return false;
    }

    if (context_slots_.find(entry.num_sms) == context_slots_.end()) {
      LOG_ERROR << "Invalid num_sms=" << entry.num_sms << " at line "
                << line_num << " (not in context_slots_)";
      return false;
    }

    schedule.push_back(entry);
  }

  sm_schedule_ = std::move(schedule);
  LOG_INFO << "Loaded SM schedule: " << sm_schedule_.size() << " entries from "
           << path;
  return true;
}

void XtGemmWorker::RunSmSchedule() {
  if (sm_schedule_.empty()) return;

  auto start = SlidingWindowDurationTracker<>::Now();

  for (const auto& entry : sm_schedule_) {
    while (SlidingWindowDurationTracker<>::ElapsedUs(start) <
           entry.time_offset_us) {
      std::this_thread::yield();
    }

    SwitchContext(entry.num_sms);
    LOG_DEBUG << "SM schedule: switched to " << entry.num_sms << " SMs at t="
              << SlidingWindowDurationTracker<>::ElapsedUs(start) << "us";
  }

  const auto& last = sm_schedule_.back();
  int64_t end_us = last.time_offset_us + last.duration_us;
  while (SlidingWindowDurationTracker<>::ElapsedUs(start) < end_us) {
    std::this_thread::yield();
  }

  LOG_DEBUG << "SM schedule completed: total "
            << SlidingWindowDurationTracker<>::ElapsedUs(start) << "us";
}

XtGemmWorkerPool::XtGemmWorkerPool(int workers_per_gpu, size_t buffer_size,
                                   WorkerSchedulingPolicy policy) {
  int device_count = 0;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));

  int total_workers = workers_per_gpu * device_count;

  for (int gpu = 0; gpu < device_count; gpu++) {
    for (int p = 0; p < workers_per_gpu; p++) {
      workers_.emplace_back(
          std::make_shared<XtGemmWorker>(gpu, workers_per_gpu, p, buffer_size));
    }
  }

  switch (policy) {
    case WorkerSchedulingPolicy::kRoundRobinGemm:
      scheduler_ = std::make_unique<RoundRobinGemmPolicy>(total_workers);
      break;
    default:
      LOG_FATAL << "Unsupported scheduling policy: "
                << WorkerSchedulingPolicyToString(policy);
  }

  LOG_INFO << "XtGemmWorkerPool created: " << total_workers << " workers ("
           << workers_per_gpu << " per GPU, " << device_count
           << " GPUs), policy=" << WorkerSchedulingPolicyToString(policy);
}

XtGemmWorkerPool::~XtGemmWorkerPool() {
  for (auto& worker : workers_) {
    worker->Stop();
  }
}

TaskHandle XtGemmWorkerPool::EnqueueGemm(const std::string& task_id,
                                         std::shared_ptr<GemmArgs> args,
                                         TaskCallback callback) {
  int worker_idx = 0;
  int priority = 0;
  std::tie(worker_idx, priority) = scheduler_->Schedule(args.get());
  (void)priority;
  auto mode = XtGemmWorker::GetPipelineMode();

  if (mode == WorkerPipelineMode::kLegacy) {
    auto task = std::bind(&XtGemmWorker::RunXtGemm, workers_[worker_idx], args);
    return workers_[worker_idx]->AddTask(task_id, std::move(task),
                                         std::move(callback));
  }

  if (mode == WorkerPipelineMode::kDualStream) {
    auto task = std::bind(&XtGemmWorker::RunGemm, workers_[worker_idx], args);
    return workers_[worker_idx]->AddTask(task_id, std::move(task),
                                         std::move(callback));
  }

  workers_[worker_idx]->RegisterPipelinedTask(task_id, args);
  WorkerBase::Task task = [] {};
  return workers_[worker_idx]->AddTask(task_id, std::move(task),
                                       std::move(callback));
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
