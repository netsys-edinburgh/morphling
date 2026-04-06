#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "../bench_common_utils.h"
#include "../cuda/bench_cuda_utils.h"

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

struct StreamGuard {
  cudaStream_t stream = nullptr;

  ~StreamGuard() {
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }

  bool Create() {
    return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
           cudaSuccess;
  }
};

static void SetTransferCounters(benchmark::State& state, size_t bytes,
                                double latency_us_sum) {
  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  const double avg_us = latency_us_sum / iters;
  const double avg_s = avg_us / 1e6;
  const double bandwidth =
      (avg_s > 0.0) ? (static_cast<double>(bytes) / avg_s / 1e9) : 0.0;
  state.counters["latency_us"] = avg_us;
  state.counters["bandwidth_GBs"] = bandwidth;
}

static void BM_MemcpyAsync_Pinned(benchmark::State& state) {
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

  StreamGuard stream;
  if (!stream.Create()) {
    cudaFreeHost(h_buf);
    state.SkipWithMessage("cudaStreamCreateWithFlags failed");
    return;
  }

  cudaEvent_t start_evt = nullptr;
  cudaEvent_t stop_evt = nullptr;
  if (cudaEventCreateWithFlags(&start_evt, cudaEventDefault) != cudaSuccess ||
      cudaEventCreateWithFlags(&stop_evt, cudaEventDefault) != cudaSuccess) {
    cudaFreeHost(h_buf);
    if (start_evt) {
      cudaEventDestroy(start_evt);
    }
    if (stop_evt) {
      cudaEventDestroy(stop_evt);
    }
    state.SkipWithMessage("cudaEventCreateWithFlags failed");
    return;
  }

  std::memset(h_buf, 0xAB, bytes);
  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf.ptr, h_buf, bytes,
                                     cudaMemcpyHostToDevice, stream.stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream.stream));
  }

  double latency_us_sum = 0.0;
  for (auto _ : state) {
    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf.ptr, h_buf, bytes,
                                     cudaMemcpyHostToDevice, stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, stream.stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    latency_us_sum += elapsed_us;
    state.SetIterationTime(elapsed_us / 1e6);
    benchmark::DoNotOptimize(d_buf.ptr);
  }

  cudaEventDestroy(start_evt);
  cudaEventDestroy(stop_evt);
  cudaFreeHost(h_buf);
  SetTransferCounters(state, bytes, latency_us_sum);
}

BENCHMARK(BM_MemcpyAsync_Pinned)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576)
    ->Arg(16777216)
    ->Arg(67108864)
    ->Arg(268435456)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void BM_MemcpyAsync_Managed(benchmark::State& state) {
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
  if (cudaMallocManaged(&h_buf, bytes) != cudaSuccess) {
    state.SkipWithMessage("cudaMallocManaged failed");
    return;
  }

  StreamGuard stream;
  if (!stream.Create()) {
    cudaFree(h_buf);
    state.SkipWithMessage("cudaStreamCreateWithFlags failed");
    return;
  }

  cudaEvent_t start_evt = nullptr;
  cudaEvent_t stop_evt = nullptr;
  if (cudaEventCreateWithFlags(&start_evt, cudaEventDefault) != cudaSuccess ||
      cudaEventCreateWithFlags(&stop_evt, cudaEventDefault) != cudaSuccess) {
    cudaFree(h_buf);
    if (start_evt) {
      cudaEventDestroy(start_evt);
    }
    if (stop_evt) {
      cudaEventDestroy(stop_evt);
    }
    state.SkipWithMessage("cudaEventCreateWithFlags failed");
    return;
  }

  std::memset(h_buf, 0xCD, bytes);
  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf.ptr, h_buf, bytes,
                                     cudaMemcpyHostToDevice, stream.stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream.stream));
  }

  double latency_us_sum = 0.0;
  for (auto _ : state) {
    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf.ptr, h_buf, bytes,
                                     cudaMemcpyHostToDevice, stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, stream.stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    latency_us_sum += elapsed_us;
    state.SetIterationTime(elapsed_us / 1e6);
    benchmark::DoNotOptimize(d_buf.ptr);
  }

  cudaEventDestroy(start_evt);
  cudaEventDestroy(stop_evt);
  cudaFree(h_buf);
  SetTransferCounters(state, bytes, latency_us_sum);
}

BENCHMARK(BM_MemcpyAsync_Managed)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576)
    ->Arg(16777216)
    ->Arg(67108864)
    ->Arg(268435456)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void BM_MemcpyAsync_Malloc(benchmark::State& state) {
  if (!EnsureCudaReady(state)) {
    return;
  }

  const size_t bytes = static_cast<size_t>(state.range(0));
  DeviceBytes d_buf;
  if (!d_buf.Allocate(bytes)) {
    state.SkipWithMessage("cudaMalloc failed");
    return;
  }

  void* h_buf = std::malloc(bytes);
  if (h_buf == nullptr) {
    state.SkipWithMessage("malloc failed");
    return;
  }

  StreamGuard stream;
  if (!stream.Create()) {
    std::free(h_buf);
    state.SkipWithMessage("cudaStreamCreateWithFlags failed");
    return;
  }

  cudaEvent_t start_evt = nullptr;
  cudaEvent_t stop_evt = nullptr;
  if (cudaEventCreateWithFlags(&start_evt, cudaEventDefault) != cudaSuccess ||
      cudaEventCreateWithFlags(&stop_evt, cudaEventDefault) != cudaSuccess) {
    std::free(h_buf);
    if (start_evt) {
      cudaEventDestroy(start_evt);
    }
    if (stop_evt) {
      cudaEventDestroy(stop_evt);
    }
    state.SkipWithMessage("cudaEventCreateWithFlags failed");
    return;
  }

  std::memset(h_buf, 0xEF, bytes);
  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf.ptr, h_buf, bytes,
                                     cudaMemcpyHostToDevice, stream.stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream.stream));
  }

  double latency_us_sum = 0.0;
  for (auto _ : state) {
    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf.ptr, h_buf, bytes,
                                     cudaMemcpyHostToDevice, stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, stream.stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    latency_us_sum += elapsed_us;
    state.SetIterationTime(elapsed_us / 1e6);
    benchmark::DoNotOptimize(d_buf.ptr);
  }

  cudaEventDestroy(start_evt);
  cudaEventDestroy(stop_evt);
  std::free(h_buf);
  SetTransferCounters(state, bytes, latency_us_sum);
}

BENCHMARK(BM_MemcpyAsync_Malloc)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1048576)
    ->Arg(16777216)
    ->Arg(67108864)
    ->Arg(268435456)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
