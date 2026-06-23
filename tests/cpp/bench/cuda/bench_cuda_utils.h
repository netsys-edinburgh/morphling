// bench_cuda_utils.h — shared utilities for CUDA benchmarks.
#pragma once

#include <cuda.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/cuda_utils.h"
#include "muduo_base/logging.h"
#include "scheduler/gpu_worker.h"

#ifdef CHECK_CUBLAS_ERROR
#undef CHECK_CUBLAS_ERROR
#endif
#define CHECK_CUBLAS_ERROR(call)                                   \
  do {                                                             \
    cublasStatus_t _status = (call);                               \
    if (_status != CUBLAS_STATUS_SUCCESS) {                        \
      std::fprintf(stderr, "CUBLAS error (%d) at %s:%d\n",         \
                   static_cast<int>(_status), __FILE__, __LINE__); \
      std::abort();                                                \
    }                                                              \
  } while (0)

#ifdef CHECK_CUDA_ERROR
#undef CHECK_CUDA_ERROR
#endif
#define CHECK_CUDA_ERROR(call)                                       \
  do {                                                               \
    cudaError_t _status = (call);                                    \
    if (_status != cudaSuccess) {                                    \
      std::fprintf(stderr, "CUDA error (%s) at %s:%d\n",             \
                   cudaGetErrorString(_status), __FILE__, __LINE__); \
      std::abort();                                                  \
    }                                                                \
  } while (0)

#ifdef LOG_FATAL_IF
#undef LOG_FATAL_IF
#endif
#define LOG_FATAL_IF(cond, msg)          \
  do {                                   \
    if (cond) {                          \
      std::fprintf(stderr, "%s", (msg)); \
      std::abort();                      \
    }                                    \
  } while (0)

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
      LOG_FATAL << "cuInit failed: " << (err_str ? err_str : "unknown");
    }
  }
}

/// Return SM count for device 0.
inline int GetSmCount() {
  CUdevice dev;
  CHECK_CU_RESULT(cuDeviceGet(&dev, 0));
  int sm_count = 0;
  CHECK_CU_RESULT(cuDeviceGetAttribute(
      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  return sm_count;
}

struct PinnedBuffer {
  float* ptr = nullptr;
  size_t count = 0;

  PinnedBuffer() = default;

  explicit PinnedBuffer(size_t elems) : count(elems) {
    if (count == 0) {
      return;
    }
    CHECK_CUDA_ERROR(
        cudaMallocHost(reinterpret_cast<void**>(&ptr), count * sizeof(float)));
    for (size_t i = 0; i < count; i++) {
      ptr[i] = static_cast<float>(i % 1000) * 0.001f;
    }
  }

  ~PinnedBuffer() {
    if (ptr) {
      cudaFreeHost(ptr);
      ptr = nullptr;
    }
  }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept
      : ptr(other.ptr), count(other.count) {
    other.ptr = nullptr;
    other.count = 0;
  }

  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (ptr) {
      cudaFreeHost(ptr);
    }
    ptr = other.ptr;
    count = other.count;
    other.ptr = nullptr;
    other.count = 0;
    return *this;
  }
};

inline std::shared_ptr<GemmArgs> MakeNNGemmArgs(int m, int n, int k, float* a,
                                                float* b, float* c) {
  auto args = std::make_shared<GemmArgs>();
  args->transa = 'N';
  args->transb = 'N';
  args->m = m;
  args->n = n;
  args->k = k;
  args->alpha = 1.0f;
  args->a = a;
  args->lda = m;
  args->b = b;
  args->ldb = k;
  args->beta = 0.0f;
  args->c = c;
  args->ldc = m;
  return args;
}

struct NonUniformSplit {
  CUdevice dev = 0;
  unsigned int nb_groups = 0;
  unsigned int group_sm_count = 0;
  std::vector<CUdevResource> all_groups;

  bool Init() {
    EnsureDriverInit();
    CHECK_CU_RESULT(cuDeviceGet(&dev, 0));

    int sm_count = GetSmCount();
    if (sm_count <= 0) {
      return false;
    }

    int cc_major = 0;
    CHECK_CU_RESULT(cuDeviceGetAttribute(
        &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));

    if (cc_major >= 9) {
      group_sm_count = 8;
    } else if (cc_major >= 8) {
      group_sm_count = 2;
    } else {
      group_sm_count = 1;
    }
    if (group_sm_count > static_cast<unsigned int>(sm_count)) {
      group_sm_count = static_cast<unsigned int>(sm_count);
    }

    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev, &device_sm, CU_DEV_RESOURCE_TYPE_SM));

    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, group_sm_count));
    if (nb_groups == 0) {
      all_groups.clear();
      return false;
    }

    all_groups.resize(nb_groups);
    CUdevResource remaining = {};
    unsigned int created_groups = nb_groups;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        all_groups.data(), &created_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, group_sm_count));
    nb_groups = created_groups;
    all_groups.resize(nb_groups);
    if (!all_groups.empty()) {
      group_sm_count = all_groups[0].sm.smCount;
    }
    return nb_groups > 0;
  }
};

struct GreenCtxSlot {
  CUgreenCtx green_ctx = nullptr;
  CUcontext cuda_ctx = nullptr;
  CUstream cu_stream = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  CUdevResourceDesc desc = nullptr;
  int num_groups = 0;
  int sm_count = 0;
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  PinnedBuffer h_A;
  PinnedBuffer h_B;
  PinnedBuffer h_C;
  std::shared_ptr<GemmArgs> args;
};

inline void RunGemmDirect(cublasHandle_t handle, cudaStream_t stream,
                          float* d_A, float* d_B, float* d_C,
                          const GemmArgs& args) {
  CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));
  CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, args.m,
                                 args.n, args.k, &args.alpha, d_A, args.lda,
                                 d_B, args.ldb, &args.beta, d_C, args.ldc));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}

inline void RunGemmDirect(cublasHandle_t handle, cudaStream_t stream,
                          float* d_A, float* d_B, float* d_C,
                          const GemmArgs* args) {
  LOG_FATAL_IF(args == nullptr, "RunGemmDirect: null GemmArgs\\n");
  RunGemmDirect(handle, stream, d_A, d_B, d_C, *args);
}
