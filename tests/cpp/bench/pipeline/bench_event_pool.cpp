#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include "../bench_common_utils.h"
#include "../cuda/bench_cuda_utils.h"

constexpr int kEventPoolSize = 64;

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

static void BM_EventCreateDestroy(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  for (auto _ : state) {
    cudaEvent_t evt = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt));
  }

  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  benchmark::DoNotOptimize(iters);
  state.counters["latency_ns"] = benchmark::Counter(
      1e-9, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

BENCHMARK(BM_EventCreateDestroy)->Unit(benchmark::kNanosecond);

static void BM_EventCreateDestroy_Timing(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  for (auto _ : state) {
    cudaEvent_t evt = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&evt, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt));
  }

  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  benchmark::DoNotOptimize(iters);
  state.counters["latency_ns"] = benchmark::Counter(
      1e-9, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

BENCHMARK(BM_EventCreateDestroy_Timing)->Unit(benchmark::kNanosecond);

class EventPool {
 public:
  explicit EventPool(size_t size) {
    pool_.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      cudaEvent_t evt = nullptr;
      CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
      pool_.push_back(evt);
      free_.push_back(evt);
    }
  }

  ~EventPool() {
    for (auto evt : pool_) {
      cudaEventDestroy(evt);
    }
  }

  cudaEvent_t Acquire() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_.empty()) {
      return nullptr;
    }
    cudaEvent_t evt = free_.front();
    free_.pop_front();
    return evt;
  }

  void Release(cudaEvent_t evt) {
    std::lock_guard<std::mutex> lock(mu_);
    free_.push_back(evt);
  }

 private:
  std::mutex mu_;
  std::vector<cudaEvent_t> pool_;
  std::deque<cudaEvent_t> free_;
};

static void BM_EventPool_Acquire_Release(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  EventPool pool(kEventPoolSize);
  for (auto _ : state) {
    cudaEvent_t evt = pool.Acquire();
    if (evt == nullptr) {
      state.SkipWithMessage("Event pool underflow");
      return;
    }
    pool.Release(evt);
  }

  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  benchmark::DoNotOptimize(iters);
  state.counters["latency_ns"] = benchmark::Counter(
      1e-9, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

BENCHMARK(BM_EventPool_Acquire_Release)->Unit(benchmark::kNanosecond);
