#include "gpu_worker.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "core/logger.h"
#include "sliding_window_tracker.h"
#include "sm_notify_source.h"

namespace {
constexpr size_t kDefaultPoolBytes = 1024ull * 1024 * 1024;  // 1 GB

size_t ResolvePoolBytes(size_t buffer_size) {
  if (const char* v = std::getenv("MORPHLING_WORKER_POOL_SIZE")) {
    return std::stoull(v);
  }
  if (buffer_size > 0) return buffer_size;
  return kDefaultPoolBytes;
}
}  // namespace

// ---------------------------------------------------------------------------
// ContextSlot RAII
// ---------------------------------------------------------------------------

ContextSlot::~ContextSlot() {
  if (cuda_ctx) {
    cuCtxSetCurrent(cuda_ctx);
  }
  if (blas_handle) {
    cublasDestroy(blas_handle);
    blas_handle = nullptr;
  }
  if (stream) {
    cudaStreamDestroy(stream);
    stream = nullptr;
  }
  if (green_ctx) {
    if (cuda_ctx) {
      cuCtxDestroy(cuda_ctx);
      cuda_ctx = nullptr;
    }
    cuGreenCtxDestroy(green_ctx);
    green_ctx = nullptr;
  }
}

ContextSlot::ContextSlot(ContextSlot&& other) noexcept
    : sm_count(other.sm_count),
      resource_desc(other.resource_desc),
      green_ctx(other.green_ctx),
      cuda_ctx(other.cuda_ctx),
      stream(other.stream),
      blas_handle(other.blas_handle) {
  other.sm_count = 0;
  other.resource_desc = nullptr;
  other.green_ctx = nullptr;
  other.cuda_ctx = nullptr;
  other.stream = nullptr;
  other.blas_handle = nullptr;
}

ContextSlot& ContextSlot::operator=(ContextSlot&& other) noexcept {
  if (this != &other) {
    // Destroy current resources
    this->~ContextSlot();
    // Move from other
    sm_count = other.sm_count;
    resource_desc = other.resource_desc;
    green_ctx = other.green_ctx;
    cuda_ctx = other.cuda_ctx;
    stream = other.stream;
    blas_handle = other.blas_handle;
    // Null out other
    other.sm_count = 0;
    other.resource_desc = nullptr;
    other.green_ctx = nullptr;
    other.cuda_ctx = nullptr;
    other.stream = nullptr;
    other.blas_handle = nullptr;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// XtGemmWorker
// ---------------------------------------------------------------------------

XtGemmWorker::XtGemmWorker(int gpu_id, int num_partitions, int partition_idx,
                           size_t buffer_size)
    : XtGemmWorker(gpu_id, num_partitions, partition_idx, buffer_size,
                   DynConfig{}, nullptr) {}

XtGemmWorker::XtGemmWorker(int gpu_id, int num_partitions, int partition_idx,
                           size_t buffer_size, DynConfig dyn,
                           std::unique_ptr<SmTargetSource> source)
    : gpu_id_(gpu_id),
      num_partitions_(num_partitions),
      partition_idx_(partition_idx),
      buffer_size_(buffer_size),
      source_(std::move(source)),
      dyn_cfg_(std::move(dyn)) {
  worker_ = std::thread([this] { Run(); });
  LOG_DEBUG << "XtGemmWorker created: gpu=" << gpu_id_
            << " partition=" << partition_idx_ << "/" << num_partitions_;
}

XtGemmWorker::~XtGemmWorker() {
  if (worker_.joinable()) {
    Stop();
  }
  active_slot_ = nullptr;
  allocator_.reset();
  context_slots_.clear();
  if (use_primary_ctx_) {
    cuDevicePrimaryCtxRelease(cu_device_);
  }
  cudaSetDevice(gpu_id_);
  LOG_DEBUG << "XtGemmWorker destroyed: gpu=" << gpu_id_
            << " partition=" << partition_idx_;
}

static bool NoGreenCtxRequested() {
  const char* v = std::getenv("MORPHLING_NO_GREEN_CTX");
  return v && std::string(v) == "1";
}

void XtGemmWorker::InitAllContexts() {
  CHECK_CU_RESULT(cuDeviceGet(&cu_device_, gpu_id_));

  int sm_count = 0;
  CHECK_CU_RESULT(cuDeviceGetAttribute(
      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_device_));

  if (NoGreenCtxRequested()) {
    LOG_INFO << "GPU " << gpu_id_
             << ": green contexts DISABLED (MORPHLING_NO_GREEN_CTX=1)"
             << ", using primary context with " << sm_count << " SMs";
    InitPrimaryContextSlot(sm_count);
    return;
  }

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
    LOG_INFO << "  Created context slot: " << slot_sm_count << " SMs"
             << " (" << n << " groups)";
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

void XtGemmWorker::InitPrimaryContextSlot(int sm_count) {
  ContextSlot slot;
  slot.sm_count = sm_count;
  slot.resource_desc = nullptr;
  slot.green_ctx = nullptr;

  CHECK_CUDA_ERROR(cudaSetDevice(gpu_id_));
  CHECK_CU_RESULT(cuDevicePrimaryCtxRetain(&slot.cuda_ctx, cu_device_));
  CHECK_CU_RESULT(cuCtxSetCurrent(slot.cuda_ctx));

  cudaStream_t s = nullptr;
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  slot.stream = s;

  CHECK_CUBLAS_ERROR(cublasCreate(&slot.blas_handle));
  CHECK_CUBLAS_ERROR(cublasSetStream(slot.blas_handle, slot.stream));

  sm_step_ = sm_count;
  partition_sm_count_ = sm_count;
  use_primary_ctx_ = true;
  context_slots_.emplace(sm_count, std::move(slot));
  active_slot_ = &context_slots_.at(sm_count);

  size_t pool_bytes = ResolvePoolBytes(buffer_size_);
  allocator_ =
      std::make_unique<CachingAllocator>(pool_bytes, MemoryType::CUDA, gpu_id_);
  LOG_INFO << "XtGemmWorker gpu=" << gpu_id_ << " partition=" << partition_idx_
           << " allocator initialized: " << pool_bytes << " bytes"
           << " (primary context, no green ctx)";

  LOG_INFO << "XtGemmWorker ready: gpu=" << gpu_id_
           << " partition=" << partition_idx_ << " (primary context, "
           << sm_count << " SMs)";
}

ContextSlot XtGemmWorker::CreateContextSlot(CUdevResource* groups,
                                            int num_groups, int sm_count) {
  ContextSlot slot;
  slot.sm_count = sm_count;

  // Combine groups into a resource descriptor
  CHECK_CU_RESULT(
      cuDevResourceGenerateDesc(&slot.resource_desc, groups, num_groups));

  // Create green context
  CHECK_CU_RESULT(cuGreenCtxCreate(&slot.green_ctx, slot.resource_desc,
                                   cu_device_, CU_GREEN_CTX_DEFAULT_STREAM));
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&slot.cuda_ctx, slot.green_ctx));
  CHECK_CU_RESULT(cuCtxSetCurrent(slot.cuda_ctx));

  // Create stream within the green context
  CUstream cu_stream = nullptr;
  CHECK_CU_RESULT(cuGreenCtxStreamCreate(&cu_stream, slot.green_ctx,
                                         CU_STREAM_NON_BLOCKING, 0));
  slot.stream = cu_stream;

  CHECK_CUBLAS_ERROR(cublasCreate(&slot.blas_handle));
  CHECK_CUBLAS_ERROR(cublasSetStream(slot.blas_handle, slot.stream));

  return slot;
}

bool XtGemmWorker::SwitchContext(int num_sms) {
  auto it = context_slots_.find(num_sms);
  if (it == context_slots_.end()) return false;
  active_slot_ = &it->second;
  CHECK_CU_RESULT(cuCtxSetCurrent(active_slot_->cuda_ctx));
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
  cudaSetDevice(gpu_id_);
  InitAllContexts();

  LOG_INFO << "XtGemmWorker ready: gpu=" << gpu_id_
           << " partition=" << partition_idx_;

  // Enter the WorkerBase task loop
  WorkerBase::Run();
}

int XtGemmWorker::SnapDown(int target) const {
  if (target <= 0 || target >= partition_sm_count_) return partition_sm_count_;
  int best = sm_step_;
  for (const auto& kv : context_slots_) {
    if (kv.first <= target && kv.first > best) best = kv.first;
  }
  return best;
}

void XtGemmWorker::SetSmTargetSource(std::unique_ptr<SmTargetSource> src) {
  source_ = std::move(src);
}

void XtGemmWorker::RunChunkedGemm(std::shared_ptr<GemmArgs> args) {
  const int m = args->m;
  const int n = args->n;
  const int k = args->k;
  const int lda = args->lda;
  const int ldb = args->ldb;
  const int ldc = args->ldc;
  const float alpha = args->alpha;
  const float beta = args->beta;
  const cublasOperation_t opa = CUDA_TRANS_OP(args->transa);
  const cublasOperation_t opb = CUDA_TRANS_OP(args->transb);
  const bool transb_n = (args->transb == 'N' || args->transb == 'n');

  const size_t a_elems = (args->transa == 'N' || args->transa == 'n')
                             ? static_cast<size_t>(lda) * k
                             : static_cast<size_t>(lda) * m;
  const size_t b_elems =
      transb_n ? static_cast<size_t>(ldb) * n : static_cast<size_t>(ldb) * k;
  const size_t c_elems = static_cast<size_t>(ldc) * n;

  const bool dynamic = dyn_cfg_.enabled && source_ != nullptr;
  const int64_t work = static_cast<int64_t>(m) * n * k;
  const bool chunked = dynamic && work >= dyn_cfg_.min_gemm_chunk_threshold;

  // Pick the initial context before staging so H2D and compute stay ordered on
  // one stream (a later switch synchronizes before changing streams).
  SwitchContext(chunked ? SnapDown(source_->Read()) : partition_sm_count_);

  float* dA =
      static_cast<float*>(allocator_->Allocate(a_elems * sizeof(float)));
  float* dB =
      static_cast<float*>(allocator_->Allocate(b_elems * sizeof(float)));
  float* dC =
      static_cast<float*>(allocator_->Allocate(c_elems * sizeof(float)));

  cudaStream_t stg = active_slot_->stream;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(dA, args->a, a_elems * sizeof(float),
                                   cudaMemcpyHostToDevice, stg));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(dB, args->b, b_elems * sizeof(float),
                                   cudaMemcpyHostToDevice, stg));
  if (beta != 0.0f) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dC, args->c, c_elems * sizeof(float),
                                     cudaMemcpyHostToDevice, stg));
  }

  auto run_chunk = [&](int j, int cw) {
    // transb=N: op(B) columns [j,j+cw) are contiguous at offset j*ldb.
    // transb=T: those columns are a row-slice of stored B at element offset j;
    // cublas transposes either way.
    const float* bptr =
        dB + (transb_n ? static_cast<size_t>(j) * ldb : static_cast<size_t>(j));
    float* cptr = dC + static_cast<size_t>(j) * ldc;
    CHECK_CUBLAS_ERROR(cublasSgemm(active_slot_->blas_handle, opa, opb, m, cw,
                                   k, &alpha, dA, lda, bptr, ldb, &beta, cptr,
                                   ldc));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(args->c + static_cast<size_t>(j) * ldc, cptr,
                        static_cast<size_t>(cw) * ldc * sizeof(float),
                        cudaMemcpyDeviceToHost, active_slot_->stream));
  };

  if (!chunked) {
    run_chunk(0, n);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(active_slot_->stream));
  } else {
    cudaEvent_t ev0, ev1;
    CHECK_CUDA_ERROR(cudaEventCreate(&ev0));
    CHECK_CUDA_ERROR(cudaEventCreate(&ev1));
    double cols_per_us = 0.0;
    int dwell = dyn_cfg_.min_dwell_chunks;
    int cw = std::max(dyn_cfg_.min_chunk_cols, 1);
    for (int j = 0; j < n;) {
      int want = SnapDown(source_->Read());
      if (want != active_slot_->sm_count &&
          dwell >= dyn_cfg_.min_dwell_chunks) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(active_slot_->stream));
        SwitchContext(want);
        dwell = 0;
      }
      int this_cw = std::min(cw, n - j);
      if (this_cw < 1) this_cw = 1;

      CHECK_CUDA_ERROR(cudaEventRecord(ev0, active_slot_->stream));
      run_chunk(j, this_cw);
      CHECK_CUDA_ERROR(cudaEventRecord(ev1, active_slot_->stream));
      CHECK_CUDA_ERROR(cudaEventSynchronize(ev1));

      float ms = 0.0f;
      CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, ev0, ev1));
      double us = std::max(static_cast<double>(ms) * 1000.0, 1.0);
      double inst = static_cast<double>(this_cw) / us;
      cols_per_us = cols_per_us > 0.0 ? 0.5 * cols_per_us + 0.5 * inst : inst;

      j += this_cw;
      ++dwell;

      int next =
          static_cast<int>(std::lround(cols_per_us * dyn_cfg_.chunk_target_us));
      next = std::max(next, dyn_cfg_.min_chunk_cols);
      if (dyn_cfg_.max_chunk_cols > 0) {
        next = std::min(next, dyn_cfg_.max_chunk_cols);
      }
      cw = next;
    }
    CHECK_CUDA_ERROR(cudaEventDestroy(ev0));
    CHECK_CUDA_ERROR(cudaEventDestroy(ev1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(active_slot_->stream));
  }

  allocator_->Free(dA);
  allocator_->Free(dB);
  allocator_->Free(dC);
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
    // Skip comments and blank lines
    size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') continue;

    std::istringstream iss(line);
    SmScheduleEntry entry;
    if (!(iss >> entry.time_offset_us >> entry.num_sms >> entry.duration_us)) {
      LOG_ERROR << "Parse error at line " << line_num << ": " << line;
      return false;
    }

    // Validate monotonic time offsets
    if (!schedule.empty() &&
        entry.time_offset_us < schedule.back().time_offset_us) {
      LOG_ERROR << "Non-monotonic time_offset_us at line " << line_num << ": "
                << entry.time_offset_us << " < "
                << schedule.back().time_offset_us;
      return false;
    }

    // Validate SM count exists
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
    // Spin-wait until this entry's time offset
    while (SlidingWindowDurationTracker<>::ElapsedUs(start) <
           entry.time_offset_us) {
      std::this_thread::yield();
    }

    SwitchContext(entry.num_sms);
    LOG_DEBUG << "SM schedule: switched to " << entry.num_sms << " SMs at t="
              << SlidingWindowDurationTracker<>::ElapsedUs(start) << "us";
  }

  // Hold until last entry's duration expires
  const auto& last = sm_schedule_.back();
  int64_t end_us = last.time_offset_us + last.duration_us;
  while (SlidingWindowDurationTracker<>::ElapsedUs(start) < end_us) {
    std::this_thread::yield();
  }

  LOG_DEBUG << "SM schedule completed: total "
            << SlidingWindowDurationTracker<>::ElapsedUs(start) << "us";
}

// ---------------------------------------------------------------------------
// XtGemmWorkerPool
// ---------------------------------------------------------------------------

XtGemmWorkerPool::XtGemmWorkerPool(int workers_per_gpu, size_t buffer_size,
                                   WorkerSchedulingPolicy policy,
                                   XtGemmWorker::DynConfig dyn) {
  int device_count = 0;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));

  int total_workers = workers_per_gpu * device_count;

  // Attach shm sources on this thread first so a require_shm failure aborts
  // before any worker thread (and its GPU contexts) is spawned.
  std::vector<std::unique_ptr<SmTargetSource>> sources(total_workers);
  if (dyn.enabled) {
    int idx = 0;
    for (int gpu = 0; gpu < device_count; gpu++) {
      for (int p = 0; p < workers_per_gpu; p++) {
        sources[idx++] = std::make_unique<ShmSmTargetSource>(
            ShmSmTargetSource::Options{dyn.shm_name, gpu, p, dyn.require_shm});
      }
    }
  }

  int widx = 0;
  for (int gpu = 0; gpu < device_count; gpu++) {
    for (int p = 0; p < workers_per_gpu; p++) {
      workers_.emplace_back(
          std::make_shared<XtGemmWorker>(gpu, workers_per_gpu, p, buffer_size,
                                         dyn, std::move(sources[widx++])));
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
  auto [worker_idx, priority] = scheduler_->Schedule(args.get());
  auto task =
      std::bind(&XtGemmWorker::RunChunkedGemm, workers_[worker_idx], args);
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
