// bench_cuda_utils.h — shared utilities for CUDA benchmarks.
#pragma once

#include <cuda.h>

#include <atomic>

#include "base/logging.h"
#include "utils/cuda_utils.h"

/// One-shot logger initialization for benchmark fixtures.
inline void EnsureLoggerInit() {
  static std::atomic<bool> done{false};
  if (!done.exchange(true)) {
    // base::g_logLevel is already extern-defined; nothing
    // extra needed beyond the static init in logging.cc.
  }
}

/// One-shot CUDA driver initialization.
inline void EnsureDriverInit() {
  static std::atomic<bool> done{false};
  if (!done.exchange(true)) {
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(res, &err_str);
      LOG_FATAL << "cuInit failed: "
                << (err_str ? err_str : "unknown");
    }
  }
}

/// Return SM count for device 0.
inline int GetSmCount() {
  CUdevice dev;
  CHECK_CU_RESULT(cuDeviceGet(&dev, 0));
  int sm_count = 0;
  CHECK_CU_RESULT(cuDeviceGetAttribute(
      &sm_count,
      CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  return sm_count;
}
