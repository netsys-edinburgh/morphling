#pragma once

#include "memory/caching_allocator.h"
#include "utils/logger.h"

#define DECLARE_INTERCEPTOR_TYPE(name, T)                                     \
  typedef void (*name##_type)(const char*, const char*, const int*,           \
                              const int*, const int*, const float*, const T*, \
                              const int*, const T*, const int*, const float*, \
                              T*, const int*)
#define DECLARE_INTERCEPTOR_PTR(name) extern name##_type orig_##name
#define DECLARE_INTERCEPTOR_FUNC(name, T)                                     \
  void name##_(const char* transa, const char* transb, const int* m,          \
               const int* n, const int* k, const float* alpha, const T* a,    \
               const int* lda, const T* b, const int* ldb, const float* beta, \
               T* c, const int* ldc)

template <typename T>
struct InterceptedArgs {
  char transa;
  char transb;
  int m;
  int n;
  int k;
  float alpha;
  const T* a;
  int lda;
  const T* b;
  int ldb;
  float beta;
  T* c;
  int ldc;
};

#define DECLARE_INTERCEPTOR(name, T) \
  DECLARE_INTERCEPTOR_TYPE(name, T); \
  DECLARE_INTERCEPTOR_PTR(name);     \
  DECLARE_INTERCEPTOR_FUNC(name, T);

#define IMPL_INTERCEPTOR_PTR(name) name##_type orig_##name = NULL
#define IMPL_INTERCEPTOR_FUNC(name, T, LIB)                                   \
  void name##_(const char* transa, const char* transb, const int* m,          \
               const int* n, const int* k, const float* alpha, const T* a,    \
               const int* lda, const T* b, const int* ldb, const float* beta, \
               T* c, const int* ldc) {                                        \
    if (!orig_##name) {                                                       \
      void* handle_lib = dlopen(LIB, RTLD_LAZY);                              \
      LOG_FATAL_IF(!handle_lib, "Error loading MKL library: {}", dlerror());  \
      orig_##name = (name##_type)dlsym(handle_lib, #name "_");                \
      LOG_FATAL_IF(!orig_##name, "Error loading original " #name "_: {}",     \
                   dlerror());                                                \
    }                                                                         \
    InterceptedArgs<T> args = {.transa = *transa,                             \
                               .transb = *transb,                             \
                               .m = *m,                                       \
                               .n = *n,                                       \
                               .k = *k,                                       \
                               .alpha = *alpha,                               \
                               .a = a,                                        \
                               .lda = *lda,                                   \
                               .b = b,                                        \
                               .ldb = *ldb,                                   \
                               .beta = *beta,                                 \
                               .c = c,                                        \
                               .ldc = *ldc};                                  \
    NotifyTaskExecution(args);                                                \
    WaitTaskExecution(args);                                                  \
  }

#define IMPL_INTERCEPTOR(name, T, LIB) \
  IMPL_INTERCEPTOR_PTR(name);          \
  IMPL_INTERCEPTOR_FUNC(name, T, LIB);

template <typename T>
std::tuple<size_t, size_t, size_t> CalculateTaskSizes(
    const InterceptedArgs<T>& args) {
  size_t size_a = (args.transa == 'N' || args.transa == 'n')
                      ? args.lda * args.k * sizeof(T)
                      : args.lda * args.m * sizeof(T);
  size_t size_b = (args.transb == 'N' || args.transb == 'n')
                      ? args.ldb * args.n * sizeof(T)
                      : args.ldb * args.k * sizeof(T);
  size_t size_c = args.ldc * args.n * sizeof(T);

  return {size_a, size_b, size_c};
}

bool CheckBufferOffloaded(const void* buffer, size_t size);

template <typename T>
void NotifyTaskExecution(const InterceptedArgs<T>& args) {
  auto [size_a, size_b, size_c] = CalculateTaskSizes(args);
  size_t task_size = sizeof(InterceptedArgs<T>) + size_a + size_b + size_c;
  if (CheckBufferOffloaded(args.a, size_a)) {
    task_size -= size_a;
  }
  if (CheckBufferOffloaded(args.b, size_b)) {
    task_size -= size_b;
  }
  if (CheckBufferOffloaded(args.c, size_c)) {
    task_size -= size_c;
  }

  // InitCachingAllocator();
}

template <typename T>
void WaitTaskExecution(const InterceptedArgs<T>& args) {}

template <typename T>
void SerializeInterceptedArgs(const InterceptedArgs<T>& args, void* buffer) {
  InterceptedArgs<T>* buffer_args =
      reinterpret_cast<InterceptedArgs<T>*>(buffer);
  *buffer_args = args;
  // deal with pointers a, b, c
}

template <typename T>
void DeserializeInterceptedArgs(const void* buffer, InterceptedArgs<T>* args) {}

DECLARE_INTERCEPTOR(sgemm, float)
DECLARE_INTERCEPTOR(sgemm_batch, float)

#if 0
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// Updated includes for shared memory management
#include "/home/eren/DeviceEmulator/csrc/memory/shared_memory_initializer.h"
#include "/home/eren/DeviceEmulator/csrc/memory/shared_memory_manager.h"

#define SHM_NAME "/sgemm_shm"
#define LOG_DIR "/home/eren/DeviceEmulator/csrc/intercept/logs/"
#define EMPTY 0
#define WRITING 1
#define WRITTEN 2
#define READING 3
#define EXECUTING 4
#define COMPLETE 5
#define MAX_TASKS 1024
#define QUEUE_SIZE 1024

typedef struct {
  char transa;
  char transb;
  int m;
  int n;
  int k;
  float alpha;
  const float* a;
  int lda;
  const float* b;
  int ldb;
  float beta;
  float* c;
  int ldc;
} sgemm_args_t;

typedef void (*sgemm_type)(const char*, const char*, const int*, const int*,
                           const int*, const float*, const float*, const int*,
                           const float*, const int*, const float*, float*,
                           const int*);

typedef void (*sgemm_batch_type)(
    const char* transa_array, const char* transb_array, const int* m_array,
    const int* n_array, const int* k_array, const float* alpha_array,
    const float* a_array[], const int* lda_array, const float* b_array[],
    const int* ldb_array, const float* beta_array, float* c_array[],
    const int* ldc_array, const int* group_count, const int* group_size);

extern FILE* log_file;
extern pthread_mutex_t shm_mutex;
extern sgemm_type orig_sgemm;

// Function declarations
void initialize_logging();
void get_timestamp(char* buffer, size_t size);
void log_message(const char* message);
void lock_memory();
void unlock_memory();
size_t calculate_matrix_size(char trans, int rows, int cols, int leading_dim);

size_t get_shared_memory_size();
int enqueue_task(task_queue_t* task_queue, int task_index);
void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* a, const int* lda,
            const float* b, const int* ldb, const float* beta, float* c,
            const int* ldc);
void process_task_from_queue(shared_memory_t* shared_mem_ptr);
#endif