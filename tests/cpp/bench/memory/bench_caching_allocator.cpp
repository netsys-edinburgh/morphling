#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "bench_common_utils.h"

#ifdef HAVE_CUDA
#include "memory/caching_allocator.h"

constexpr int64_t kMemTypeShm = 0;
constexpr int64_t kMemTypePin = 1;
constexpr int64_t kMemTypeCuda = 2;

constexpr size_t k4KB = 4ULL * 1024ULL;
constexpr size_t k64KB = 64ULL * 1024ULL;
constexpr size_t k1MB = 1024ULL * 1024ULL;
constexpr size_t k64MB = 64ULL * 1024ULL * 1024ULL;
constexpr size_t k256MB = 256ULL * 1024ULL * 1024ULL;
constexpr double kCudaHeadroomRatio = 0.8;

constexpr std::array<int64_t, 5> kBenchmarkSizes = {
    static_cast<int64_t>(k4KB), static_cast<int64_t>(k64KB),
    static_cast<int64_t>(k1MB), static_cast<int64_t>(k64MB),
    static_cast<int64_t>(k256MB)};

inline void SetCommonCounters(benchmark::State& state, double allocations,
                              double bytes) {
  state.counters["allocations_per_second"] =
      benchmark::Counter(allocations, benchmark::Counter::kIsRate);
  state.counters["bytes_per_second"] =
      benchmark::Counter(bytes, benchmark::Counter::kIsRate);
}

inline bool EnsureCudaDevice(benchmark::State& state) {
  if (!CheckCudaAvailable()) {
    state.SkipWithMessage("No CUDA device");
    return false;
  }

  if (cudaSetDevice(0) != cudaSuccess) {
    state.SkipWithMessage("Failed to select CUDA device");
    return false;
  }
  return true;
}

inline bool EnsureCudaHeadroom(benchmark::State& state, size_t bytes) {
  size_t free_mem = 0;
  size_t total_mem = 0;
  const cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
  if (err != cudaSuccess) {
    state.SkipWithMessage("cudaMemGetInfo failed");
    return false;
  }

  if (bytes > static_cast<size_t>(free_mem * kCudaHeadroomRatio)) {
    state.SkipWithMessage("Insufficient GPU memory");
    return false;
  }
  return true;
}

inline void WarmPool(CachingAllocator& allocator, size_t bytes,
                     int warmups = 1) {
  for (int i = 0; i < warmups; ++i) {
    void* ptr = allocator.Allocate(bytes);
    benchmark::DoNotOptimize(ptr);
    allocator.Free(ptr);
  }
}

inline int DeviceIdForType(MemoryType type) {
  return type == MemoryType::CUDA ? 0 : -1;
}

inline bool TypeNeedsCuda(MemoryType type) {
  return type == MemoryType::CUDA || type == MemoryType::PIN;
}

inline size_t PoolBytesForSize(size_t bench_size) {
  return std::max(bench_size, k1MB);
}

static void BM_CachingAllocator_CudaMalloc(benchmark::State& state) {
  const size_t bench_size = static_cast<size_t>(state.range(0));
  if (!EnsureCudaDevice(state) || !EnsureCudaHeadroom(state, bench_size)) {
    return;
  }

  CachingAllocator allocator(PoolBytesForSize(bench_size), MemoryType::CUDA, 0);
  WarmPool(allocator, bench_size, kGpuWarmupIterations);

  for (auto _ : state) {
    void* ptr = allocator.Allocate(bench_size);
    benchmark::DoNotOptimize(ptr);
    allocator.Free(ptr);
  }

  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bench_size));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(state.iterations() * bench_size));
}

static void BM_CachingAllocator_PinnedAlloc(benchmark::State& state) {
  const size_t bench_size = static_cast<size_t>(state.range(0));
  if (!EnsureCudaDevice(state)) {
    return;
  }

  CachingAllocator allocator(PoolBytesForSize(bench_size), MemoryType::PIN);
  WarmPool(allocator, bench_size, kGpuWarmupIterations);

  for (auto _ : state) {
    void* ptr = allocator.Allocate(bench_size);
    benchmark::DoNotOptimize(ptr);
    allocator.Free(ptr);
  }

  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bench_size));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(state.iterations() * bench_size));
}

static void BM_CachingAllocator_ShmAlloc(benchmark::State& state) {
  const size_t bench_size = static_cast<size_t>(state.range(0));
  CachingAllocator allocator(PoolBytesForSize(bench_size), MemoryType::SHM);
  WarmPool(allocator, bench_size);

  for (auto _ : state) {
    void* ptr = allocator.Allocate(bench_size);
    benchmark::DoNotOptimize(ptr);
    allocator.Free(ptr);
  }

  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bench_size));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(state.iterations() * bench_size));
}

static void BM_CachingAllocator_ColdAlloc(benchmark::State& state) {
  const size_t bench_size = static_cast<size_t>(state.range(0));
  const auto mem_type = static_cast<MemoryType>(state.range(1));

  if (TypeNeedsCuda(mem_type) && !EnsureCudaDevice(state)) {
    return;
  }
  if (mem_type == MemoryType::CUDA && !EnsureCudaHeadroom(state, bench_size)) {
    return;
  }

  for (auto _ : state) {
    CachingAllocator fresh_allocator(PoolBytesForSize(bench_size), mem_type,
                                     DeviceIdForType(mem_type));
    void* ptr = fresh_allocator.Allocate(bench_size);
    benchmark::DoNotOptimize(ptr);
    fresh_allocator.Free(ptr);
  }

  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bench_size));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(state.iterations() * bench_size));
}

static void BM_CachingAllocator_Contention(benchmark::State& state) {
  constexpr size_t kBenchSize = k64KB;
  static CachingAllocator shared_allocator(k64MB, MemoryType::SHM);
  static std::once_flag warm_once;

  std::call_once(warm_once, []() {
    void* warm = shared_allocator.Allocate(kBenchSize);
    shared_allocator.Free(warm);
  });

  for (auto _ : state) {
    void* ptr = shared_allocator.Allocate(kBenchSize);
    benchmark::DoNotOptimize(ptr);
    shared_allocator.Free(ptr);
  }

  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * kBenchSize));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(state.iterations() * kBenchSize));
}

static void BM_CachingAllocator_MixedSizes(benchmark::State& state) {
  const std::array<size_t, 5> sizes = {k4KB, k64KB, k1MB, k64MB, k256MB};
  CachingAllocator allocator(512ULL * 1024ULL * 1024ULL, MemoryType::SHM);

  for (size_t size : sizes) {
    WarmPool(allocator, size);
  }

  size_t index = 0;
  size_t total_bytes = 0;
  for (auto _ : state) {
    const size_t size = sizes[index % sizes.size()];
    ++index;
    void* ptr = allocator.Allocate(size);
    benchmark::DoNotOptimize(ptr);
    allocator.Free(ptr);
    total_bytes += size;
  }

  state.SetBytesProcessed(static_cast<int64_t>(total_bytes));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(total_bytes));
}

static void BM_RawCudaMalloc_Baseline(benchmark::State& state) {
  const size_t bench_size = static_cast<size_t>(state.range(0));
  if (!EnsureCudaDevice(state) || !EnsureCudaHeadroom(state, bench_size)) {
    return;
  }

  for (auto _ : state) {
    void* ptr = nullptr;
    const cudaError_t alloc_err = cudaMalloc(&ptr, bench_size);
    if (alloc_err != cudaSuccess) {
      state.SkipWithMessage(cudaGetErrorString(alloc_err));
      return;
    }
    benchmark::DoNotOptimize(ptr);
    cudaFree(ptr);
  }

  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bench_size));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(state.iterations() * bench_size));
}

static void BM_RawCudaHostAlloc_Baseline(benchmark::State& state) {
  const size_t bench_size = static_cast<size_t>(state.range(0));
  if (!EnsureCudaDevice(state)) {
    return;
  }

  for (auto _ : state) {
    void* ptr = nullptr;
    const cudaError_t alloc_err =
        cudaHostAlloc(&ptr, bench_size, cudaHostAllocDefault);
    if (alloc_err != cudaSuccess) {
      state.SkipWithMessage(cudaGetErrorString(alloc_err));
      return;
    }
    benchmark::DoNotOptimize(ptr);
    cudaFreeHost(ptr);
  }

  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * bench_size));
  SetCommonCounters(state, static_cast<double>(state.iterations()),
                    static_cast<double>(state.iterations() * bench_size));
}

#else

static void BM_CachingAllocator_CudaMalloc(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

static void BM_CachingAllocator_PinnedAlloc(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

static void BM_CachingAllocator_ShmAlloc(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

static void BM_CachingAllocator_ColdAlloc(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

static void BM_CachingAllocator_Contention(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

static void BM_CachingAllocator_MixedSizes(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

static void BM_RawCudaMalloc_Baseline(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

static void BM_RawCudaHostAlloc_Baseline(benchmark::State& state) {
  state.SkipWithMessage("Built without CUDA toolkit");
}

constexpr std::array<int64_t, 5> kBenchmarkSizes = {4096, 65536, 1048576,
                                                    67108864, 268435456};
constexpr int64_t kMemTypeShm = 0;
constexpr int64_t kMemTypePin = 1;
constexpr int64_t kMemTypeCuda = 2;

#endif

BENCHMARK(BM_CachingAllocator_CudaMalloc)
    ->Arg(kBenchmarkSizes[0])
    ->Arg(kBenchmarkSizes[1])
    ->Arg(kBenchmarkSizes[2])
    ->Arg(kBenchmarkSizes[3])
    ->Arg(kBenchmarkSizes[4]);

BENCHMARK(BM_CachingAllocator_PinnedAlloc)
    ->Arg(kBenchmarkSizes[0])
    ->Arg(kBenchmarkSizes[1])
    ->Arg(kBenchmarkSizes[2])
    ->Arg(kBenchmarkSizes[3])
    ->Arg(kBenchmarkSizes[4]);

BENCHMARK(BM_CachingAllocator_ShmAlloc)
    ->Arg(kBenchmarkSizes[0])
    ->Arg(kBenchmarkSizes[1])
    ->Arg(kBenchmarkSizes[2])
    ->Arg(kBenchmarkSizes[3])
    ->Arg(kBenchmarkSizes[4]);

BENCHMARK(BM_CachingAllocator_ColdAlloc)
    ->Args({kBenchmarkSizes[0], kMemTypeCuda})
    ->Args({kBenchmarkSizes[1], kMemTypeCuda})
    ->Args({kBenchmarkSizes[2], kMemTypeCuda})
    ->Args({kBenchmarkSizes[3], kMemTypeCuda})
    ->Args({kBenchmarkSizes[4], kMemTypeCuda})
    ->Args({kBenchmarkSizes[0], kMemTypePin})
    ->Args({kBenchmarkSizes[1], kMemTypePin})
    ->Args({kBenchmarkSizes[2], kMemTypePin})
    ->Args({kBenchmarkSizes[3], kMemTypePin})
    ->Args({kBenchmarkSizes[4], kMemTypePin})
    ->Args({kBenchmarkSizes[0], kMemTypeShm})
    ->Args({kBenchmarkSizes[1], kMemTypeShm})
    ->Args({kBenchmarkSizes[2], kMemTypeShm})
    ->Args({kBenchmarkSizes[3], kMemTypeShm})
    ->Args({kBenchmarkSizes[4], kMemTypeShm});

BENCHMARK(BM_CachingAllocator_Contention)->Threads(2)->Threads(4)->Threads(8);

BENCHMARK(BM_CachingAllocator_MixedSizes);

BENCHMARK(BM_RawCudaMalloc_Baseline)
    ->Arg(kBenchmarkSizes[0])
    ->Arg(kBenchmarkSizes[1])
    ->Arg(kBenchmarkSizes[2])
    ->Arg(kBenchmarkSizes[3])
    ->Arg(kBenchmarkSizes[4]);

BENCHMARK(BM_RawCudaHostAlloc_Baseline)
    ->Arg(kBenchmarkSizes[0])
    ->Arg(kBenchmarkSizes[1])
    ->Arg(kBenchmarkSizes[2])
    ->Arg(kBenchmarkSizes[3])
    ->Arg(kBenchmarkSizes[4]);
