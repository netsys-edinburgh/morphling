#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "../bench_common_utils.h"
#include "../cuda/bench_cuda_utils.h"
#include "core/types_and_defs.h"

class CudaPinnedMemoryPool {
 public:
  static CudaPinnedMemoryPool& Instance() {
    static CudaPinnedMemoryPool pool;
    return pool;
  }

  explicit CudaPinnedMemoryPool(size_t max_buffers_per_bucket = 16)
      : max_per_bucket_(max_buffers_per_bucket) {}

  ~CudaPinnedMemoryPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : free_lists_) {
      auto& free_list = kv.second;
      for (auto* ptr : free_list) {
        cudaFreeHost(ptr);
      }
    }
  }

  std::pair<void*, size_t> Acquire(size_t size) {
    size_t bucket = BucketSize(size);
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket];
    if (!free_list.empty()) {
      void* ptr = free_list.back();
      free_list.pop_back();
      return {ptr, bucket};
    }
    void* ptr = nullptr;
    cudaError_t err =
        cudaHostAlloc(&ptr, bucket, cudaHostAllocDefault | cudaHostAllocMapped);
    if (err != cudaSuccess || ptr == nullptr) {
      throw std::runtime_error("CudaPinnedMemoryPool: cudaHostAlloc failed");
    }
    return {ptr, bucket};
  }

  void Release(void* ptr, size_t bucket_size) {
    if (ptr == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_lists_[bucket_size];
    if (free_list.size() < max_per_bucket_) {
      free_list.push_back(ptr);
    } else {
      cudaFreeHost(ptr);
    }
  }

 private:
  static size_t BucketSize(size_t size) {
    static constexpr size_t kMinBucket = 4096;
    if (size <= kMinBucket) {
      return kMinBucket;
    }
    size_t bucket = kMinBucket;
    while (bucket < size) {
      bucket <<= 1;
    }
    return bucket;
  }

  size_t max_per_bucket_;
  std::mutex mutex_;
  std::unordered_map<size_t, std::deque<void*>> free_lists_;
};

constexpr int kWarmupIterations = 3;

static bool EnsureCudaReady(benchmark::State& state) {
  if (!CheckCudaAvailable()) {
    state.SkipWithMessage("No GPU");
    return false;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    state.SkipWithMessage("No GPU");
    return false;
  }
  return true;
}

struct DeviceBytes {
  void* ptr = nullptr;

  ~DeviceBytes() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }

  bool Allocate(size_t bytes) { return cudaMalloc(&ptr, bytes) == cudaSuccess; }
};

static double AverageUs(benchmark::State& state, double total_us) {
  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  return total_us / iters;
}

static void BM_CudaMallocManaged_Latency(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  const size_t bytes = static_cast<size_t>(state.range(0));
  for (int i = 0; i < kWarmupIterations; ++i) {
    void* ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMallocManaged(&ptr, bytes));
    CHECK_CUDA_ERROR(cudaFree(ptr));
  }

  double total_us = 0.0;
  for (auto _ : state) {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&start_evt, cudaEventDefault));
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&stop_evt, cudaEventDefault));

    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, 0));
    void* ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMallocManaged(&ptr, bytes));
    CHECK_CUDA_ERROR(cudaFree(ptr));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    total_us += elapsed_us;
    state.SetIterationTime(elapsed_us / 1e6);

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
  }

  state.counters["latency_us"] = AverageUs(state, total_us);
}

BENCHMARK(BM_CudaMallocManaged_Latency)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576)
    ->Arg(16777216)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void BM_CudaMemcpy_Latency(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  const size_t bytes = static_cast<size_t>(state.range(0));
  DeviceBytes d_buf;
  if (!d_buf.Allocate(bytes)) {
    state.SkipWithMessage("cudaMalloc failed");
    return;
  }

  void* h_buf = nullptr;
  if (cudaHostAlloc(&h_buf, bytes, cudaHostAllocDefault) != cudaSuccess) {
    state.SkipWithMessage("cudaHostAlloc failed");
    return;
  }
  std::memset(h_buf, 0xAB, bytes);

  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_buf.ptr, h_buf, bytes, cudaMemcpyHostToDevice));
  }

  double total_us = 0.0;
  for (auto _ : state) {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&start_evt, cudaEventDefault));
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&stop_evt, cudaEventDefault));

    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, 0));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_buf.ptr, h_buf, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    total_us += elapsed_us;
    state.SetIterationTime(elapsed_us / 1e6);

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    benchmark::DoNotOptimize(d_buf.ptr);
  }

  cudaFreeHost(h_buf);

  const double avg_us = AverageUs(state, total_us);
  const double avg_s = avg_us / 1e6;
  const double bandwidth =
      (avg_s > 0.0) ? (static_cast<double>(bytes) / avg_s / 1e9) : 0.0;
  state.counters["latency_us"] = avg_us;
  state.counters["bandwidth_GBs"] = bandwidth;
}

BENCHMARK(BM_CudaMemcpy_Latency)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576)
    ->Arg(16777216)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void BM_CudaHostAlloc_Latency(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  const size_t bytes = static_cast<size_t>(state.range(0));
  for (int i = 0; i < kWarmupIterations; ++i) {
    void* ptr = nullptr;
    CHECK_CUDA_ERROR(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaFreeHost(ptr));
  }

  double total_us = 0.0;
  for (auto _ : state) {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&start_evt, cudaEventDefault));
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&stop_evt, cudaEventDefault));

    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, 0));
    void* ptr = nullptr;
    CHECK_CUDA_ERROR(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    total_us += elapsed_us;
    state.SetIterationTime(elapsed_us / 1e6);

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
  }

  state.counters["latency_us"] = AverageUs(state, total_us);
}

BENCHMARK(BM_CudaHostAlloc_Latency)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576)
    ->Arg(16777216)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void BM_PinnedPool_Acquire_Latency(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  const size_t bytes = static_cast<size_t>(state.range(0));
  auto& pool = CudaPinnedMemoryPool::Instance();

  for (int i = 0; i < kWarmupIterations; ++i) {
    auto acquired = pool.Acquire(bytes);
    pool.Release(acquired.first, acquired.second);
  }

  double total_us = 0.0;
  for (auto _ : state) {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&start_evt, cudaEventDefault));
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&stop_evt, cudaEventDefault));

    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, 0));
    auto acquired = pool.Acquire(bytes);
    pool.Release(acquired.first, acquired.second);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    total_us += elapsed_us;
    state.SetIterationTime(elapsed_us / 1e6);

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
  }

  state.counters["latency_us"] = AverageUs(state, total_us);
}

BENCHMARK(BM_PinnedPool_Acquire_Latency)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576)
    ->Arg(16777216)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
