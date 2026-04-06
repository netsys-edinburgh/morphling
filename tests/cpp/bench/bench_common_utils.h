#pragma once

#if defined(__has_include)
#if __has_include(<benchmark/benchmark.h>)
#include <benchmark/benchmark.h>
#endif
#endif

#include <cstddef>
#include <thread>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

inline bool CheckCudaAvailable() {
#ifdef HAVE_CUDA
  int device_count = 0;
  return (cudaGetDeviceCount(&device_count) == cudaSuccess) && device_count > 0;
#else
  return false;
#endif
}

inline unsigned int GetHardwareConcurrency() {
  const unsigned int concurrency = std::thread::hardware_concurrency();
  return concurrency == 0 ? 1u : concurrency;
}

static constexpr int kGpuWarmupIterations = 3;
static constexpr int kDefaultRepetitions = 5;

// Run any benchmark binary with JSON output:
//   ./bench_name --benchmark_format=json --benchmark_out=result.json
// Or via the CI runner:
//   ./tests/run_cpp_tests.sh bench --json   (writes to tests/results/*.json)
