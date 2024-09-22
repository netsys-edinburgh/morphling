#include "interceptor.h"

#include <dlfcn.h>
#include <stdio.h>

#include <cstdint>

extern "C" {
// IMPL_INTERCEPTOR(sgemm, float, "libmkl_rt.so")
// IMPL_INTERCEPTOR(sgemm_batch, float, "libmkl_rt.so")

sgemm_type orig_sgemm = NULL;

void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* a, const int* lda,
            const float* b, const int* ldb, const float* beta, float* c,
            const int* ldc) {
  if (!orig_sgemm) {
    void* handle_lib = dlopen("libmkl_rt.so", RTLD_LAZY);
    LOG_FATAL_IF(!handle_lib, "Error loading MKL library: {}", dlerror());
    orig_sgemm = (sgemm_type)dlsym(handle_lib, "sgemm_");
    LOG_FATAL_IF(!orig_sgemm, "Error loading original sgemm_: {}", dlerror());
  }
  LOG_DEBUG(
      "Intercepted sgemm; args transa: {}, transb: {}, m: {}, n: {}, k: "
      "{}, alpha: {}, lda: {}, ldb: {}, beta: {}, ldc: {}",
      *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc);
  GemmArgsPtr args = std::make_unique<GemmArgs>();
  args->group_size = 1;
  args->transa[0] = *transa;
  args->transb[0] = *transb;
  args->m[0] = *m;
  args->n[0] = *n;
  args->k[0] = *k;
  args->alpha[0] = *alpha;
  args->a[0] = a;
  args->lda[0] = *lda;
  args->b[0] = b;
  args->ldb[0] = *ldb;
  args->beta[0] = *beta;
  args->c[0] = c;
  args->ldc[0] = *ldc;
  TaskExecution(args);
}
}

void TaskExecution(const GemmArgsPtr& args) {
  InitMemoryManagerClient();
  size_t task_size = sizeof(GemmArgs);
  void* task_buffer = kCachingAllocator->Allocate(task_size);
  auto* buffer_args = reinterpret_cast<GemmArgs*>(task_buffer);
  *buffer_args = *args;
  kMemoryManagerClient->ScheduleGemmSync(args->a, args->b, args->c,
                                         task_buffer);
}

std::tuple<size_t, size_t, size_t> CalculateTaskSizes(const GemmArgsPtr& args) {
  if (args->group_size == 1) {
    auto size_a = (*args->transa == 'N' || *args->transa == 'n')
                      ? (*args->lda) * (*args->k) * sizeof(float)
                      : (*args->lda) * (*args->m) * sizeof(float);
    auto size_b = (*args->transb == 'N' || *args->transb == 'n')
                      ? (*args->ldb) * (*args->n) * sizeof(float)
                      : (*args->ldb) * (*args->k) * sizeof(float);
    auto size_c = (*args->ldc) * (*args->n) * sizeof(float);
    return {size_a, size_b, size_c};
  }
  return {0, 0, 0};
}

// bool CheckBufferOffloaded(const void* buffer, size_t size) {
//   // when the buffer is offloaded, the first and last uint32_t is the same
//   const uint32_t* buffer_uint32 = reinterpret_cast<const uint32_t*>(buffer);
//   const uint32_t first_num = buffer_uint32[0];
//   const uint32_t last_num = buffer_uint32[size / sizeof(uint32_t) - 1];
//   return (first_num == last_num) && (first_num < 0xFFFF) && (last_num <
//   0xFFFF);
// }