#include "green_context_runtime.h"

#include <algorithm>
#include <chrono>

#include "utils/logger.h"

// ---------------------------------------------------------------------------
// GreenContextSlot RAII
// ---------------------------------------------------------------------------

GreenContextSlot::~GreenContextSlot() {
  if (cuda_ctx) {
    cuCtxSetCurrent(cuda_ctx);
  }
  // Destroy streams before context
  for (auto& [role, s] : streams) {
    if (s) {
      cudaStreamDestroy(s);
      s = nullptr;
    }
  }
  streams.clear();
  if (cuda_ctx) {
    cuCtxDestroy(cuda_ctx);
    cuda_ctx = nullptr;
  }
  if (green_ctx) {
    cuGreenCtxDestroy(green_ctx);
    green_ctx = nullptr;
  }
}

GreenContextSlot::GreenContextSlot(GreenContextSlot&& other) noexcept
    : sm_count(other.sm_count),
      resource_desc(other.resource_desc),
      green_ctx(other.green_ctx),
      cuda_ctx(other.cuda_ctx),
      streams(std::move(other.streams)),
      generation(other.generation) {
  other.sm_count = 0;
  other.resource_desc = nullptr;
  other.green_ctx = nullptr;
  other.cuda_ctx = nullptr;
  other.generation = 0;
}

GreenContextSlot& GreenContextSlot::operator=(
    GreenContextSlot&& other) noexcept {
  if (this != &other) {
    this->~GreenContextSlot();
    sm_count = other.sm_count;
    resource_desc = other.resource_desc;
    green_ctx = other.green_ctx;
    cuda_ctx = other.cuda_ctx;
    streams = std::move(other.streams);
    generation = other.generation;
    other.sm_count = 0;
    other.resource_desc = nullptr;
    other.green_ctx = nullptr;
    other.cuda_ctx = nullptr;
    other.generation = 0;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// GreenContextRuntime
// ---------------------------------------------------------------------------

GreenContextRuntime::GreenContextRuntime(const Options& opts) : opts_(opts) {
  // Check CUDA device availability
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    supported_ = false;
    unsupported_reason_ = "No CUDA devices available";
    return;
  }
  if (opts_.gpu_id >= device_count) {
    supported_ = false;
    unsupported_reason_ = "GPU ID " + std::to_string(opts_.gpu_id) +
                          " out of range (have " +
                          std::to_string(device_count) + ")";
    return;
  }

  // Check CUDA driver version (green contexts require CUDA 12.5+)
  int driver_version = 0;
  CUresult dr_err = cuDriverGetVersion(&driver_version);
  if (dr_err != CUDA_SUCCESS || driver_version < 12050) {
    supported_ = false;
    unsupported_reason_ = "CUDA driver version " +
                          std::to_string(driver_version) +
                          " too old (need >= 12050 for green contexts)";
    if (opts_.strict) {
      LOG_FATAL << unsupported_reason_;
    }
    LOG_WARN << unsupported_reason_;
    return;
  }

  supported_ = true;

  // Initialize on the correct device
  cudaSetDevice(opts_.gpu_id);
  InitAllContexts();
}

GreenContextRuntime::~GreenContextRuntime() { Close(); }

void GreenContextRuntime::Close() {
  if (closed_) return;
  closed_ = true;

  // Clear slots (destroys green contexts, streams)
  slots_.clear();

  // Restore primary context
  if (supported_) {
    cudaSetDevice(opts_.gpu_id);
  }

  LOG_DEBUG << "GreenContextRuntime closed: gpu=" << opts_.gpu_id;
}

void GreenContextRuntime::InitAllContexts() {
  CHECK_CU_RESULT(cuDeviceGet(&cu_device_, opts_.gpu_id));

  // Query total SM count
  int sm_count = 0;
  CHECK_CU_RESULT(cuDeviceGetAttribute(
      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_device_));
  LOG_INFO << "GreenContextRuntime gpu=" << opts_.gpu_id
           << " total_sms=" << sm_count
           << " partitions=" << opts_.num_partitions
           << " partition_idx=" << opts_.partition_idx;

  // Split into finest-grained groups (minCount=2)
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

  // Divide groups among partitions
  unsigned int groups_per_part = nb_groups / opts_.num_partitions;
  LOG_FATAL_IF(groups_per_part == 0)
      << "Not enough SM groups (" << nb_groups << ") for "
      << opts_.num_partitions << " partitions";

  partition_sm_count_ = static_cast<int>(groups_per_part) * sm_step_;
  unsigned int base_offset = opts_.partition_idx * groups_per_part;

  LOG_INFO << "Partition " << opts_.partition_idx << ": " << groups_per_part
           << " groups, " << partition_sm_count_
           << " SMs (offset=" << base_offset << ")";

  // Create a slot for each valid SM count
  for (unsigned int n = 1; n <= groups_per_part; n++) {
    int slot_sm = static_cast<int>(n) * sm_step_;
    auto slot = CreateContextSlot(&sm_groups_[base_offset], static_cast<int>(n),
                                  slot_sm);
    slots_.emplace(slot_sm, std::move(slot));
    LOG_INFO << "  Created green ctx slot: " << slot_sm << " SMs (" << n
             << " groups)";
  }

  // Default: use all partition SMs
  active_sm_count_.store(partition_sm_count_, std::memory_order_relaxed);

  LOG_INFO << "GreenContextRuntime initialized: " << slots_.size()
           << " slots, active=" << partition_sm_count_ << " SMs";
}

GreenContextSlot GreenContextRuntime::CreateContextSlot(CUdevResource* groups,
                                                        int num_groups,
                                                        int sm_count) {
  GreenContextSlot slot;
  slot.sm_count = sm_count;
  slot.generation = generation_.load(std::memory_order_relaxed);

  // Combine groups into a resource descriptor
  CHECK_CU_RESULT(
      cuDevResourceGenerateDesc(&slot.resource_desc, groups, num_groups));

  // Create green context
  CHECK_CU_RESULT(cuGreenCtxCreate(&slot.green_ctx, slot.resource_desc,
                                   cu_device_, CU_GREEN_CTX_DEFAULT_STREAM));
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&slot.cuda_ctx, slot.green_ctx));
  CHECK_CU_RESULT(cuCtxSetCurrent(slot.cuda_ctx));

  // Create one stream per role within the green context
  for (auto role : opts_.roles) {
    CUstream cu_stream = nullptr;
    CHECK_CU_RESULT(cuGreenCtxStreamCreate(&cu_stream, slot.green_ctx,
                                           CU_STREAM_NON_BLOCKING,
                                           opts_.stream_priority));
    slot.streams[role] = cu_stream;
  }

  return slot;
}

// ── Introspection ───────────────────────────────────────

int GreenContextRuntime::ActiveSmCount() const {
  return active_sm_count_.load(std::memory_order_relaxed);
}

std::vector<int> GreenContextRuntime::AvailableSmCounts() const {
  std::vector<int> counts;
  counts.reserve(slots_.size());
  for (const auto& [sm, _] : slots_) {
    counts.push_back(sm);
  }
  std::sort(counts.begin(), counts.end());
  return counts;
}

// ── Stream Table ────────────────────────────────────────

cudaStream_t GreenContextRuntime::GetStream(int sm_count,
                                            StreamRole role) const {
  std::lock_guard<std::mutex> lk(mutex_);
  auto sit = slots_.find(sm_count);
  if (sit == slots_.end()) return nullptr;
  auto rit = sit->second.streams.find(role);
  if (rit == sit->second.streams.end()) return nullptr;
  return rit->second;
}

// ── Control ─────────────────────────────────────────────

bool GreenContextRuntime::SetActiveSmCount(int num_sms) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = slots_.find(num_sms);
  if (it == slots_.end()) return false;
  int prev = active_sm_count_.exchange(num_sms, std::memory_order_relaxed);
  if (prev != num_sms) {
    switch_count_.fetch_add(1, std::memory_order_relaxed);
    LOG_DEBUG << "GreenContextRuntime: switched " << prev << " → " << num_sms
              << " SMs";
  }
  return true;
}

// ── Trace ───────────────────────────────────────────────

bool GreenContextRuntime::LoadTrace(const std::string& path) {
  std::vector<GreenTraceEntry> entries;
  GreenTraceConfig config;
  std::string error;

  if (!GreenTraceParser::Parse(path, entries, config, error)) {
    LOG_ERROR << "Failed to parse trace: " << error;
    return false;
  }

  // Validate SM counts against available slots
  std::set<int> valid;
  for (const auto& [sm, _] : slots_) {
    valid.insert(sm);
  }
  if (!GreenTraceParser::Validate(entries, valid, error)) {
    LOG_ERROR << "Trace validation failed: " << error;
    return false;
  }

  std::lock_guard<std::mutex> lk(mutex_);
  trace_entries_ = std::move(entries);
  trace_config_ = config;
  LOG_INFO << "Loaded trace: " << trace_entries_.size() << " entries from "
           << path;
  return true;
}

int GreenContextRuntime::SmCountAtTime(int64_t elapsed_us) const {
  return GreenTraceParser::SmCountAtTime(trace_entries_, elapsed_us,
                                         partition_sm_count_);
}

int GreenContextRuntime::SmCountAtStep(int64_t step) const {
  // For step mode, timestamp IS the step index
  return GreenTraceParser::SmCountAtTime(trace_entries_, step,
                                         partition_sm_count_);
}

// ── Activation ──────────────────────────────────────────

int GreenContextRuntime::ActivateForThread() {
  int sm = active_sm_count_.load(std::memory_order_acquire);
  return ActivateSmForThread(sm);
}

int GreenContextRuntime::ActivateSmForThread(int num_sms) {
  if (!supported_) return 0;

  std::lock_guard<std::mutex> lk(mutex_);
  auto it = slots_.find(num_sms);
  if (it == slots_.end()) {
    LOG_ERROR << "ActivateSmForThread: invalid sm_count=" << num_sms;
    return active_sm_count_.load(std::memory_order_relaxed);
  }

  // Push the green context onto the CUDA context stack
  auto t_start = std::chrono::steady_clock::now();
  CHECK_CU_RESULT(cuCtxSetCurrent(it->second.cuda_ctx));
  auto t_end = std::chrono::steady_clock::now();

  int prev = active_sm_count_.exchange(num_sms, std::memory_order_relaxed);
  if (prev != num_sms) {
    auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();
    switch_count_.fetch_add(1, std::memory_order_relaxed);
    swap_count_.fetch_add(1, std::memory_order_relaxed);
    swap_overhead_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
  }
  return prev;
}

void GreenContextRuntime::DeactivateForThread(int prev_sm_count) {
  if (!supported_) return;

  std::lock_guard<std::mutex> lk(mutex_);
  if (prev_sm_count <= 0) {
    // Restore primary context
    cudaSetDevice(opts_.gpu_id);
    return;
  }
  auto it = slots_.find(prev_sm_count);
  if (it != slots_.end()) {
    CHECK_CU_RESULT(cuCtxSetCurrent(it->second.cuda_ctx));
  } else {
    cudaSetDevice(opts_.gpu_id);
  }
  active_sm_count_.store(prev_sm_count, std::memory_order_relaxed);
}

SwapStats GreenContextRuntime::GetAndResetSwapStats() {
  SwapStats stats;
  stats.count = swap_count_.exchange(0, std::memory_order_relaxed);
  stats.total_overhead_us =
      swap_overhead_us_.exchange(0, std::memory_order_relaxed);
  return stats;
}
