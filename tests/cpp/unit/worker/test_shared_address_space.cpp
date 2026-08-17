#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

// Guards the load-bearing assumption behind RunChunkedGemm's "resume": a device
// buffer allocated while one green context is current stays valid and usable
// from a cuBLAS kernel launched under a *different* green context on the same
// device. If this ever fails on new hardware, RunChunkedGemm must re-stage its
// device operands after each SwitchContext instead of reusing them.

namespace {

#define CK_CU(call)                                     \
  do {                                                  \
    CUresult s = (call);                                \
    if (s != CUDA_SUCCESS) {                            \
      const char* e = nullptr;                          \
      cuGetErrorString(s, &e);                          \
      GTEST_FAIL() << #call << " -> " << (e ? e : "?"); \
    }                                                   \
  } while (0)

#define CK_RT(call)                                             \
  do {                                                          \
    cudaError_t s = (call);                                     \
    if (s != cudaSuccess)                                       \
      GTEST_FAIL() << #call << " -> " << cudaGetErrorString(s); \
  } while (0)

struct GreenCtx {
  CUgreenCtx green = nullptr;
  CUcontext ctx = nullptr;
  CUstream stream = nullptr;
};

}  // namespace

TEST(SharedAddressSpace, DeviceBufferValidAcrossGreenCtxSwitch) {
  if (cuInit(0) != CUDA_SUCCESS) GTEST_SKIP() << "cuInit failed";
  CUdevice dev = 0;
  CK_CU(cuDeviceGet(&dev, 0));

  CUdevResource sm{};
  CK_CU(cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM));
  unsigned nb = 0;
  if (cuDevSmResourceSplitByCount(
          nullptr, &nb, &sm, nullptr,
          CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2) != CUDA_SUCCESS ||
      nb < 2) {
    GTEST_SKIP() << "green contexts unsupported or <2 SM groups";
  }
  std::vector<CUdevResource> groups(nb);
  CUdevResource rem{};
  CK_CU(cuDevSmResourceSplitByCount(
      groups.data(), &nb, &sm, &rem,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

  GreenCtx a;
  GreenCtx b;
  CUdevResourceDesc da = nullptr;
  CUdevResourceDesc db = nullptr;
  CK_CU(cuDevResourceGenerateDesc(&da, &groups[0], 1));
  CK_CU(cuGreenCtxCreate(&a.green, da, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CK_CU(cuCtxFromGreenCtx(&a.ctx, a.green));
  CK_CU(cuCtxSetCurrent(a.ctx));
  CK_CU(cuGreenCtxStreamCreate(&a.stream, a.green, CU_STREAM_NON_BLOCKING, 0));

  CK_CU(cuDevResourceGenerateDesc(&db, &groups[0], 2));
  CK_CU(cuGreenCtxCreate(&b.green, db, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CK_CU(cuCtxFromGreenCtx(&b.ctx, b.green));
  CK_CU(cuCtxSetCurrent(b.ctx));
  CK_CU(cuGreenCtxStreamCreate(&b.stream, b.green, CU_STREAM_NON_BLOCKING, 0));

  const int n = 1 << 16;
  std::vector<float> host(n, 3.0f);

  // Allocate + fill under context A.
  CK_CU(cuCtxSetCurrent(a.ctx));
  float* d = nullptr;
  CK_RT(cudaMalloc(reinterpret_cast<void**>(&d), n * sizeof(float)));
  CK_RT(cudaMemcpy(d, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  // Scale it by 2 with a cuBLAS kernel launched under context B.
  CK_CU(cuCtxSetCurrent(b.ctx));
  cublasHandle_t h = nullptr;
  ASSERT_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  ASSERT_EQ(cublasSetStream(h, b.stream), CUBLAS_STATUS_SUCCESS);
  const float alpha = 2.0f;
  ASSERT_EQ(cublasSscal(h, n, &alpha, d, 1), CUBLAS_STATUS_SUCCESS);
  CK_RT(cudaStreamSynchronize(b.stream));

  std::vector<float> out(n, 0.0f);
  CK_RT(cudaMemcpy(out.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    ASSERT_FLOAT_EQ(out[i], 6.0f) << "mismatch at index " << i;
  }

  cublasDestroy(h);
  CK_CU(cuCtxSetCurrent(a.ctx));
  cudaFree(d);
  cuStreamDestroy(a.stream);
  cuStreamDestroy(b.stream);
  cuCtxSetCurrent(a.ctx);
  cuGreenCtxDestroy(a.green);
  cuGreenCtxDestroy(b.green);
  cudaSetDevice(0);
}
