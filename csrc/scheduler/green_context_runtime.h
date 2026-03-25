#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "green_trace_parser.h"
#include "utils/cuda_utils.h"

// Stream roles for multi-stream training patterns.
enum class StreamRole {
  kCompute = 0,
  kRecv = 1,
  kSend = 2,
  kDp = 3,
};

inline const char* StreamRoleToString(StreamRole r) {
  switch (r) {
    case StreamRole::kCompute:
      return "compute";
    case StreamRole::kRecv:
      return "recv";
    case StreamRole::kSend:
      return "send";
    case StreamRole::kDp:
      return "dp";
    default:
      return "unknown";
  }
}

// Synchronization policy when switching between green contexts.
enum class SwitchSyncMode {
  kNone,        // No sync — fastest, in-flight work continues
  kEventChain,  // Record event on old, wait on new (dependency)
  kStreamSync,  // Full stream synchronize before switch
};

// ---------------------------------------------------------------------------
// GreenContextSlot — RAII wrapper for one SM partition
// ---------------------------------------------------------------------------
// Similar to ContextSlot but without cublasXt. Holds a green context,
// converted CUcontext, and one stream per role. Move-only.

struct GreenContextSlot {
  int sm_count = 0;
  CUdevResourceDesc resource_desc = nullptr;
  CUgreenCtx green_ctx = nullptr;
  CUcontext cuda_ctx = nullptr;

  // One stream per role, all created within this green context
  std::unordered_map<StreamRole, cudaStream_t> streams;

  // Monotonically increasing; bumped on reconfigure
  uint64_t generation = 0;

  GreenContextSlot() = default;
  ~GreenContextSlot();
  GreenContextSlot(GreenContextSlot&& other) noexcept;
  GreenContextSlot& operator=(GreenContextSlot&& other) noexcept;
  GreenContextSlot(const GreenContextSlot&) = delete;
  GreenContextSlot& operator=(const GreenContextSlot&) = delete;
};

struct SwapStats {
  int64_t switch_count = 0;
  int64_t count = 0;
  int64_t total_overhead_us = 0;
  double avg_overhead_us() const {
    return switch_count > 0
               ? static_cast<double>(total_overhead_us) / switch_count
               : 0.0;
  }
};

// ---------------------------------------------------------------------------
// GreenContextRuntime — main controller
// ---------------------------------------------------------------------------
// Pre-creates green context slots for each valid SM count on a GPU
// partition. Provides stream table lookup, trace-driven scheduling,
// and thread-level activation via setCurrentCUDAStream.
//
// Thread safety: mutable state protected by mutex. Activation methods
// must be called from the owning thread (the one that will dispatch
// PyTorch ops).

class GreenContextRuntime {
 public:
  struct Options {
    int gpu_id = 0;
    int num_partitions = 1;
    int partition_idx = 0;
    std::vector<StreamRole> roles = {StreamRole::kCompute, StreamRole::kRecv,
                                     StreamRole::kSend, StreamRole::kDp};
    int stream_priority = -1;
    bool strict = false;
    SwitchSyncMode switch_sync = SwitchSyncMode::kEventChain;
  };

  explicit GreenContextRuntime(const Options& opts);
  ~GreenContextRuntime();

  // Non-copyable, non-movable (owns CUDA resources)
  GreenContextRuntime(const GreenContextRuntime&) = delete;
  GreenContextRuntime& operator=(const GreenContextRuntime&) = delete;

  // ── Capability / Introspection ──────────────────────────

  // Returns false if green contexts are not supported (e.g.,
  // old CUDA driver < 12.5, or resource allocation failure).
  bool IsSupported() const { return supported_; }
  std::string UnsupportedReason() const { return unsupported_reason_; }

  // Hardware SM granularity (CC 8.x = 2, CC 9.0+ = 8)
  int SmStep() const { return sm_step_; }

  // Total SMs available to this partition
  int PartitionSmCount() const { return partition_sm_count_; }

  // Currently active SM count
  int ActiveSmCount() const;

  // All valid SM counts (sorted ascending)
  std::vector<int> AvailableSmCounts() const;

  int GpuId() const { return opts_.gpu_id; }

  // ── Stream Table ────────────────────────────────────────

  // Returns the CUstream handle for a given (sm_count, role).
  // Returns nullptr if sm_count is invalid or role not configured.
  // The returned handle is valid for the lifetime of this runtime
  // (until Close() or destruction). Callers should wrap once as
  // torch.cuda.ExternalStream and cache.
  cudaStream_t GetStream(int sm_count, StreamRole role) const;

  // Returns stream as uintptr_t for pybind exposure
  uintptr_t GetStreamPtr(int sm_count, StreamRole role) const {
    return reinterpret_cast<uintptr_t>(GetStream(sm_count, role));
  }

  // ── Control ─────────────────────────────────────────────

  // Switch the active SM count. Future Activate* calls will
  // use streams from this partition. Returns false if num_sms
  // is not a valid SM count.
  bool SetActiveSmCount(int num_sms);

  // ── Trace ───────────────────────────────────────────────

  // Load a trace file and validate against available SM counts.
  bool LoadTrace(const std::string& path);

  // Get the SM count that should be active at the given elapsed
  // time (in the trace's time unit, converted to us internally).
  // Returns default (partition max) if no trace loaded or time
  // is before first entry.
  int SmCountAtTime(int64_t elapsed_us) const;

  // Get the SM count for a given step index (for step clock mode).
  int SmCountAtStep(int64_t step) const;

  // ── Activation (thread-level stream swap) ───────────────

  // Activate the current SM partition's streams on the calling
  // thread. Sets c10::cuda::setCurrentCUDAStream for each role.
  // Returns the previous active SM count (for restore).
  // Must be called from the thread that will dispatch PyTorch ops.
  int ActivateForThread();

  // Activate a specific SM count on the calling thread.
  // Returns previous active SM count.
  int ActivateSmForThread(int num_sms);

  // Restore previous streams. Called with the value returned
  // by ActivateForThread / ActivateSmForThread.
  void DeactivateForThread(int prev_sm_count);

  // ── Lifecycle ───────────────────────────────────────────

  // Explicitly release all CUDA resources. Safe to call multiple
  // times. Destructor calls this automatically.
  void Close();

  // ── Stats ───────────────────────────────────────────────

  uint64_t SwitchCount() const {
    return switch_count_.load(std::memory_order_relaxed);
  }

  uint64_t Generation() const {
    return generation_.load(std::memory_order_relaxed);
  }

  SwapStats GetAndResetSwapStats();

 private:
  void InitAllContexts();
  GreenContextSlot CreateContextSlot(CUdevResource* groups, int num_groups,
                                     int sm_count);

  Options opts_;
  bool supported_ = false;
  std::string unsupported_reason_;
  bool closed_ = false;

  CUdevice cu_device_ = 0;
  int sm_step_ = 0;
  int partition_sm_count_ = 0;

  // SM group resources (from cuDevSmResourceSplitByCount)
  std::vector<CUdevResource> sm_groups_;

  // sm_count → GreenContextSlot
  std::unordered_map<int, GreenContextSlot> slots_;

  // Currently active SM count
  std::atomic<int> active_sm_count_{0};

  // Trace data
  std::vector<GreenTraceEntry> trace_entries_;
  GreenTraceConfig trace_config_;

  // Stats
  std::atomic<uint64_t> switch_count_{0};
  std::atomic<uint64_t> generation_{0};
  std::atomic<int64_t> swap_count_{0};
  std::atomic<int64_t> swap_overhead_us_{0};

  mutable std::mutex mutex_;
};
