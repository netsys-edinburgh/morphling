#include <benchmark/benchmark.h>

#include <algorithm>
#include <mutex>

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

struct DeviceBuffer {
  float* ptr = nullptr;

  ~DeviceBuffer() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }

  bool Allocate(size_t elems) {
    return cudaMalloc(reinterpret_cast<void**>(&ptr), elems * sizeof(float)) ==
           cudaSuccess;
  }
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

struct CublasGuard {
  cublasHandle_t handle = nullptr;

  ~CublasGuard() {
    if (handle) {
      cublasDestroy(handle);
      handle = nullptr;
    }
  }

  bool Create() { return cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS; }
};

static bool CreateEvent(cudaEvent_t* evt, bool timing = true) {
  const unsigned int flags = timing ? cudaEventDefault : cudaEventDisableTiming;
  return cudaEventCreateWithFlags(evt, flags) == cudaSuccess;
}

static void DestroyEvent(cudaEvent_t* evt) {
  if (*evt) {
    cudaEventDestroy(*evt);
    *evt = nullptr;
  }
}

static inline void LaunchNNGemm(cublasHandle_t handle, float* d_a, float* d_b,
                                float* d_c, int dim) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim,
                                 dim, &alpha, d_a, dim, d_b, dim, &beta, d_c,
                                 dim));
}

static void WarmupSerial(cublasHandle_t handle, float* d_a, float* d_b,
                         float* d_c, const float* h_a, const float* h_b,
                         int dim) {
  const size_t bytes = static_cast<size_t>(dim) * dim * sizeof(float);
  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    LaunchNNGemm(handle, d_a, d_b, d_c, dim);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  }
}

static void WarmupOverlapped(cublasHandle_t handle, cudaStream_t copy_stream,
                             cudaStream_t compute_stream, float* d_a,
                             float* d_b, float* d_c, const float* h_a,
                             const float* h_b, int dim,
                             cudaEvent_t h2d_done_evt) {
  const size_t bytes = static_cast<size_t>(dim) * dim * sizeof(float);
  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, copy_stream));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, copy_stream));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d_done_evt, copy_stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream, h2d_done_evt, 0));
    LaunchNNGemm(handle, d_a, d_b, d_c, dim);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));
  }
}

static void BM_H2D_Then_Gemm_Serial(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  if (!EnsureCudaReady(state)) {
    return;
  }

  const int dim = static_cast<int>(state.range(0));
  const size_t elems = static_cast<size_t>(dim) * dim;
  const size_t bytes = elems * sizeof(float);

  PinnedBuffer h_a(elems);
  PinnedBuffer h_b(elems);
  PinnedBuffer h_c(elems);

  DeviceBuffer d_a;
  DeviceBuffer d_b;
  DeviceBuffer d_c;
  if (!d_a.Allocate(elems) || !d_b.Allocate(elems) || !d_c.Allocate(elems)) {
    state.SkipWithMessage("cudaMalloc failed");
    return;
  }

  StreamGuard compute_stream;
  StreamGuard copy_stream;
  if (!compute_stream.Create() || !copy_stream.Create()) {
    state.SkipWithMessage("cudaStreamCreateWithFlags failed");
    return;
  }

  CublasGuard cublas;
  if (!cublas.Create()) {
    state.SkipWithMessage("cublasCreate failed");
    return;
  }
  CHECK_CUBLAS_ERROR(cublasSetStream(cublas.handle, compute_stream.stream));

  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_stop = nullptr;
  cudaEvent_t gemm_start = nullptr;
  cudaEvent_t gemm_stop = nullptr;
  if (!CreateEvent(&h2d_start) || !CreateEvent(&h2d_stop) ||
      !CreateEvent(&gemm_start) || !CreateEvent(&gemm_stop)) {
    state.SkipWithMessage("cudaEventCreateWithFlags failed");
    DestroyEvent(&h2d_start);
    DestroyEvent(&h2d_stop);
    DestroyEvent(&gemm_start);
    DestroyEvent(&gemm_stop);
    return;
  }

  WarmupSerial(cublas.handle, d_a.ptr, d_b.ptr, d_c.ptr, h_a.ptr, h_b.ptr, dim);

  double total_h2d_us = 0.0;
  double total_gemm_us = 0.0;
  for (auto _ : state) {
    CHECK_CUDA_ERROR(cudaEventRecord(h2d_start, 0));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_a.ptr, h_a.ptr, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_b.ptr, h_b.ptr, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d_stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(h2d_stop));

    CHECK_CUDA_ERROR(cudaEventRecord(gemm_start, compute_stream.stream));
    LaunchNNGemm(cublas.handle, d_a.ptr, d_b.ptr, d_c.ptr, dim);
    CHECK_CUDA_ERROR(cudaEventRecord(gemm_stop, compute_stream.stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(gemm_stop));

    float h2d_ms = 0.0f;
    float gemm_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gemm_ms, gemm_start, gemm_stop));
    total_h2d_us += static_cast<double>(h2d_ms) * 1000.0;
    total_gemm_us += static_cast<double>(gemm_ms) * 1000.0;
    benchmark::DoNotOptimize(d_c.ptr);
  }

  DestroyEvent(&h2d_start);
  DestroyEvent(&h2d_stop);
  DestroyEvent(&gemm_start);
  DestroyEvent(&gemm_stop);

  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  const double flops = 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["H2D_us"] = total_h2d_us / iters;
  state.counters["GEMM_us"] = total_gemm_us / iters;
}

BENCHMARK(BM_H2D_Then_Gemm_Serial)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_H2D_Gemm_Overlapped(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  if (!EnsureCudaReady(state)) {
    return;
  }

  const int dim = static_cast<int>(state.range(0));
  const size_t elems = static_cast<size_t>(dim) * dim;
  const size_t bytes = elems * sizeof(float);

  PinnedBuffer h_a(elems);
  PinnedBuffer h_b(elems);
  PinnedBuffer h_c(elems);

  DeviceBuffer d_a;
  DeviceBuffer d_b;
  DeviceBuffer d_c;
  if (!d_a.Allocate(elems) || !d_b.Allocate(elems) || !d_c.Allocate(elems)) {
    state.SkipWithMessage("cudaMalloc failed");
    return;
  }

  StreamGuard compute_stream;
  StreamGuard copy_stream;
  if (!compute_stream.Create() || !copy_stream.Create()) {
    state.SkipWithMessage("cudaStreamCreateWithFlags failed");
    return;
  }

  CublasGuard cublas;
  if (!cublas.Create()) {
    state.SkipWithMessage("cublasCreate failed");
    return;
  }
  CHECK_CUBLAS_ERROR(cublasSetStream(cublas.handle, compute_stream.stream));

  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_stop = nullptr;
  cudaEvent_t h2d_done = nullptr;
  cudaEvent_t gemm_start = nullptr;
  cudaEvent_t gemm_stop = nullptr;
  if (!CreateEvent(&h2d_start) || !CreateEvent(&h2d_stop) ||
      !CreateEvent(&h2d_done, false) || !CreateEvent(&gemm_start) ||
      !CreateEvent(&gemm_stop)) {
    state.SkipWithMessage("cudaEventCreateWithFlags failed");
    DestroyEvent(&h2d_start);
    DestroyEvent(&h2d_stop);
    DestroyEvent(&h2d_done);
    DestroyEvent(&gemm_start);
    DestroyEvent(&gemm_stop);
    return;
  }

  WarmupOverlapped(cublas.handle, copy_stream.stream, compute_stream.stream,
                   d_a.ptr, d_b.ptr, d_c.ptr, h_a.ptr, h_b.ptr, dim, h2d_done);

  double total_h2d_us = 0.0;
  double total_gemm_us = 0.0;
  for (auto _ : state) {
    CHECK_CUDA_ERROR(cudaEventRecord(h2d_start, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_a.ptr, h_a.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_b.ptr, h_b.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d_stop, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d_done, copy_stream.stream));

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream.stream, h2d_done, 0));
    CHECK_CUDA_ERROR(cudaEventRecord(gemm_start, compute_stream.stream));
    LaunchNNGemm(cublas.handle, d_a.ptr, d_b.ptr, d_c.ptr, dim);
    CHECK_CUDA_ERROR(cudaEventRecord(gemm_stop, compute_stream.stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream.stream));

    float h2d_ms = 0.0f;
    float gemm_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gemm_ms, gemm_start, gemm_stop));
    total_h2d_us += static_cast<double>(h2d_ms) * 1000.0;
    total_gemm_us += static_cast<double>(gemm_ms) * 1000.0;
    benchmark::DoNotOptimize(d_c.ptr);
  }

  DestroyEvent(&h2d_start);
  DestroyEvent(&h2d_stop);
  DestroyEvent(&h2d_done);
  DestroyEvent(&gemm_start);
  DestroyEvent(&gemm_stop);

  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  const double flops = 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["H2D_us"] = total_h2d_us / iters;
  state.counters["GEMM_us"] = total_gemm_us / iters;
}

BENCHMARK(BM_H2D_Gemm_Overlapped)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_PipelinedPair(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  if (!EnsureCudaReady(state)) {
    return;
  }

  const int dim = static_cast<int>(state.range(0));
  const size_t elems = static_cast<size_t>(dim) * dim;
  const size_t bytes = elems * sizeof(float);

  PinnedBuffer h_a0(elems);
  PinnedBuffer h_b0(elems);
  PinnedBuffer h_c0(elems);
  PinnedBuffer h_a1(elems);
  PinnedBuffer h_b1(elems);
  PinnedBuffer h_c1(elems);

  DeviceBuffer d_a0;
  DeviceBuffer d_b0;
  DeviceBuffer d_c0;
  DeviceBuffer d_a1;
  DeviceBuffer d_b1;
  DeviceBuffer d_c1;
  if (!d_a0.Allocate(elems) || !d_b0.Allocate(elems) || !d_c0.Allocate(elems) ||
      !d_a1.Allocate(elems) || !d_b1.Allocate(elems) || !d_c1.Allocate(elems)) {
    state.SkipWithMessage("cudaMalloc failed");
    return;
  }

  StreamGuard compute_stream;
  StreamGuard copy_stream;
  if (!compute_stream.Create() || !copy_stream.Create()) {
    state.SkipWithMessage("cudaStreamCreateWithFlags failed");
    return;
  }

  CublasGuard cublas;
  if (!cublas.Create()) {
    state.SkipWithMessage("cublasCreate failed");
    return;
  }
  CHECK_CUBLAS_ERROR(cublasSetStream(cublas.handle, compute_stream.stream));

  cudaEvent_t h2d0_start = nullptr;
  cudaEvent_t h2d0_stop = nullptr;
  cudaEvent_t h2d1_start = nullptr;
  cudaEvent_t h2d1_stop = nullptr;
  cudaEvent_t gemm0_start = nullptr;
  cudaEvent_t gemm0_stop = nullptr;
  cudaEvent_t gemm1_start = nullptr;
  cudaEvent_t gemm1_stop = nullptr;
  if (!CreateEvent(&h2d0_start) || !CreateEvent(&h2d0_stop) ||
      !CreateEvent(&h2d1_start) || !CreateEvent(&h2d1_stop) ||
      !CreateEvent(&gemm0_start) || !CreateEvent(&gemm0_stop) ||
      !CreateEvent(&gemm1_start) || !CreateEvent(&gemm1_stop)) {
    state.SkipWithMessage("cudaEventCreateWithFlags failed");
    DestroyEvent(&h2d0_start);
    DestroyEvent(&h2d0_stop);
    DestroyEvent(&h2d1_start);
    DestroyEvent(&h2d1_stop);
    DestroyEvent(&gemm0_start);
    DestroyEvent(&gemm0_stop);
    DestroyEvent(&gemm1_start);
    DestroyEvent(&gemm1_stop);
    return;
  }

  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_CUDA_ERROR(cudaEventRecord(h2d0_start, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_a0.ptr, h_a0.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_b0.ptr, h_b0.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d0_stop, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream.stream, h2d0_stop, 0));
    LaunchNNGemm(cublas.handle, d_a0.ptr, d_b0.ptr, d_c0.ptr, dim);
    CHECK_CUDA_ERROR(cudaEventRecord(gemm0_stop, compute_stream.stream));

    CHECK_CUDA_ERROR(cudaEventRecord(h2d1_start, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_a1.ptr, h_a1.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_b1.ptr, h_b1.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d1_stop, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream.stream, h2d1_stop, 0));
    LaunchNNGemm(cublas.handle, d_a1.ptr, d_b1.ptr, d_c1.ptr, dim);
    CHECK_CUDA_ERROR(cudaEventRecord(gemm1_stop, compute_stream.stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream.stream));
  }

  double total_h2d_us = 0.0;
  double total_gemm_us = 0.0;
  double total_overlap_pct = 0.0;
  for (auto _ : state) {
    CHECK_CUDA_ERROR(cudaEventRecord(h2d0_start, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_a0.ptr, h_a0.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_b0.ptr, h_b0.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d0_stop, copy_stream.stream));

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream.stream, h2d0_stop, 0));
    CHECK_CUDA_ERROR(cudaEventRecord(gemm0_start, compute_stream.stream));
    LaunchNNGemm(cublas.handle, d_a0.ptr, d_b0.ptr, d_c0.ptr, dim);
    CHECK_CUDA_ERROR(cudaEventRecord(gemm0_stop, compute_stream.stream));

    CHECK_CUDA_ERROR(cudaEventRecord(h2d1_start, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_a1.ptr, h_a1.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        d_b1.ptr, h_b1.ptr, bytes, cudaMemcpyHostToDevice, copy_stream.stream));
    CHECK_CUDA_ERROR(cudaEventRecord(h2d1_stop, copy_stream.stream));

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream.stream, h2d1_stop, 0));
    CHECK_CUDA_ERROR(cudaEventRecord(gemm1_start, compute_stream.stream));
    LaunchNNGemm(cublas.handle, d_a1.ptr, d_b1.ptr, d_c1.ptr, dim);
    CHECK_CUDA_ERROR(cudaEventRecord(gemm1_stop, compute_stream.stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream.stream));

    float h2d0_ms = 0.0f;
    float h2d1_ms = 0.0f;
    float gemm0_ms = 0.0f;
    float gemm1_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2d0_ms, h2d0_start, h2d0_stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2d1_ms, h2d1_start, h2d1_stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gemm0_ms, gemm0_start, gemm0_stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gemm1_ms, gemm1_start, gemm1_stop));

    const double h2d0_us = static_cast<double>(h2d0_ms) * 1000.0;
    const double h2d1_us = static_cast<double>(h2d1_ms) * 1000.0;
    const double gemm0_us = static_cast<double>(gemm0_ms) * 1000.0;
    const double gemm1_us = static_cast<double>(gemm1_ms) * 1000.0;

    total_h2d_us += (h2d0_us + h2d1_us);
    total_gemm_us += (gemm0_us + gemm1_us);
    if (h2d1_us > 0.0) {
      total_overlap_pct += 100.0 * std::min(h2d1_us, gemm0_us) / h2d1_us;
    }
    benchmark::DoNotOptimize(d_c0.ptr);
    benchmark::DoNotOptimize(d_c1.ptr);
  }

  DestroyEvent(&h2d0_start);
  DestroyEvent(&h2d0_stop);
  DestroyEvent(&h2d1_start);
  DestroyEvent(&h2d1_stop);
  DestroyEvent(&gemm0_start);
  DestroyEvent(&gemm0_stop);
  DestroyEvent(&gemm1_start);
  DestroyEvent(&gemm1_stop);

  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  const double flops = 2.0 * dim * dim * dim;
  state.counters["GFLOPS"] = benchmark::Counter(
      2.0 * flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["H2D_us"] = total_h2d_us / iters;
  state.counters["GEMM_us"] = total_gemm_us / iters;
  state.counters["overlap_pct"] = total_overlap_pct / iters;
}

BENCHMARK(BM_PipelinedPair)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);
