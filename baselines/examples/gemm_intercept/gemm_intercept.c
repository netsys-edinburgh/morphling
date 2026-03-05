#define _GNU_SOURCE

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if __has_include(<cublas_v2.h>)
#include <cublas_v2.h>
#else
typedef int cublasStatus_t;
typedef void* cublasHandle_t;
typedef int cublasOperation_t;
typedef struct {
  unsigned short x;
} __half;
typedef int cudaDataType;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;
enum { CUBLAS_STATUS_INTERNAL_ERROR = 14 };
#endif

#if __has_include(<cublasLt.h>)
#include <cublasLt.h>
#else
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulAlgo_t;
typedef void* cudaStream_t;
typedef enum {
  CUBLASLT_MATRIX_LAYOUT_ROWS = 0,
  CUBLASLT_MATRIX_LAYOUT_COLS = 1
} cublasLtMatrixLayoutAttribute_t;
extern cublasStatus_t cublasLtMatrixLayoutGetAttribute(
    cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);
#endif

typedef cublasStatus_t (*cublasSgemm_v2_fn)(cublasHandle_t, cublasOperation_t,
                                            cublasOperation_t, int, int, int,
                                            const float*, const float*, int,
                                            const float*, int, const float*,
                                            float*, int);

typedef cublasStatus_t (*cublasDgemm_v2_fn)(cublasHandle_t, cublasOperation_t,
                                            cublasOperation_t, int, int, int,
                                            const double*, const double*, int,
                                            const double*, int, const double*,
                                            double*, int);

typedef cublasStatus_t (*cublasHgemm_fn)(cublasHandle_t, cublasOperation_t,
                                         cublasOperation_t, int, int, int,
                                         const void*, const void*, int,
                                         const void*, int, const void*, void*,
                                         int);

typedef cublasStatus_t (*cublasGemmEx_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType, int, const void*, cudaDataType, int,
    const void*, void*, cudaDataType, int, cublasComputeType_t,
    cublasGemmAlgo_t);

typedef cublasStatus_t (*cublasLtMatmul_fn)(
    cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*,
    cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t, const void*,
    const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
    const cublasLtMatmulAlgo_t*, void*, size_t, cudaStream_t);

static cublasSgemm_v2_fn g_real_cublasSgemm_v2 = NULL;
static cublasDgemm_v2_fn g_real_cublasDgemm_v2 = NULL;
static cublasHgemm_fn g_real_cublasHgemm = NULL;
static cublasGemmEx_fn g_real_cublasGemmEx = NULL;
static cublasLtMatmul_fn g_real_cublasLtMatmul = NULL;

static FILE* g_log_fp = NULL;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline long long now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static void log_gemm(const char* name, long long start, long long end, int m,
                     int n, int k) {
  pthread_mutex_lock(&g_mutex);
  if (g_log_fp != NULL) {
    fprintf(g_log_fp, "%s,%lld,%lld,%lld,%d,%d,%d\n", name, start, end,
            end - start, m, n, k);
    fflush(g_log_fp);
  }
  pthread_mutex_unlock(&g_mutex);
}

static void* resolve_symbol(const char* symbol_name) {
  dlerror();
  void* sym = dlsym(RTLD_NEXT, symbol_name);
  const char* err = dlerror();
  if (err != NULL) {
    fprintf(stderr, "[gemm_intercept] dlsym(%s) failed: %s\n", symbol_name,
            err);
    return NULL;
  }
  return sym;
}

static cublasSgemm_v2_fn get_real_cublasSgemm_v2(void) {
  pthread_mutex_lock(&g_mutex);
  if (g_real_cublasSgemm_v2 == NULL) {
    g_real_cublasSgemm_v2 = (cublasSgemm_v2_fn)resolve_symbol("cublasSgemm_v2");
  }
  cublasSgemm_v2_fn fn = g_real_cublasSgemm_v2;
  pthread_mutex_unlock(&g_mutex);
  return fn;
}

static cublasDgemm_v2_fn get_real_cublasDgemm_v2(void) {
  pthread_mutex_lock(&g_mutex);
  if (g_real_cublasDgemm_v2 == NULL) {
    g_real_cublasDgemm_v2 = (cublasDgemm_v2_fn)resolve_symbol("cublasDgemm_v2");
  }
  cublasDgemm_v2_fn fn = g_real_cublasDgemm_v2;
  pthread_mutex_unlock(&g_mutex);
  return fn;
}

static cublasHgemm_fn get_real_cublasHgemm(void) {
  pthread_mutex_lock(&g_mutex);
  if (g_real_cublasHgemm == NULL) {
    g_real_cublasHgemm = (cublasHgemm_fn)resolve_symbol("cublasHgemm");
  }
  cublasHgemm_fn fn = g_real_cublasHgemm;
  pthread_mutex_unlock(&g_mutex);
  return fn;
}

static cublasGemmEx_fn get_real_cublasGemmEx(void) {
  pthread_mutex_lock(&g_mutex);
  if (g_real_cublasGemmEx == NULL) {
    g_real_cublasGemmEx = (cublasGemmEx_fn)resolve_symbol("cublasGemmEx");
  }
  cublasGemmEx_fn fn = g_real_cublasGemmEx;
  pthread_mutex_unlock(&g_mutex);
  return fn;
}

static cublasLtMatmul_fn get_real_cublasLtMatmul(void) {
  pthread_mutex_lock(&g_mutex);
  if (g_real_cublasLtMatmul == NULL) {
    g_real_cublasLtMatmul = (cublasLtMatmul_fn)resolve_symbol("cublasLtMatmul");
  }
  cublasLtMatmul_fn fn = g_real_cublasLtMatmul;
  pthread_mutex_unlock(&g_mutex);
  return fn;
}

__attribute__((constructor)) static void gemm_intercept_init(void) {
  const char* log_path = getenv("GEMM_LOG_PATH");
  if (log_path == NULL || log_path[0] == '\0') {
    return;
  }

  pthread_mutex_lock(&g_mutex);
  g_log_fp = fopen(log_path, "a");
  if (g_log_fp != NULL) {
    fprintf(g_log_fp, "function_name,start_ns,end_ns,duration_ns,m,n,k\n");
    fflush(g_log_fp);
  }
  pthread_mutex_unlock(&g_mutex);
}

__attribute__((destructor)) static void gemm_intercept_fini(void) {
  pthread_mutex_lock(&g_mutex);
  if (g_log_fp != NULL) {
    fflush(g_log_fp);
    fclose(g_log_fp);
    g_log_fp = NULL;
  }
  pthread_mutex_unlock(&g_mutex);
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta,
                              float* C, int ldc) {
  cublasSgemm_v2_fn real_fn = get_real_cublasSgemm_v2();
  if (real_fn == NULL) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  long long start = now_ns();
  cublasStatus_t st = real_fn(handle, transa, transb, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  long long end = now_ns();
  log_gemm("cublasSgemm_v2", start, end, m, n, k);
  return st;
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const double* alpha, const double* A, int lda,
                              const double* B, int ldb, const double* beta,
                              double* C, int ldc) {
  cublasDgemm_v2_fn real_fn = get_real_cublasDgemm_v2();
  if (real_fn == NULL) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  long long start = now_ns();
  cublasStatus_t st = real_fn(handle, transa, transb, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  long long end = now_ns();
  log_gemm("cublasDgemm_v2", start, end, m, n, k);
  return st;
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const void* alpha, const void* A, int lda,
                           const void* B, int ldb, const void* beta, void* C,
                           int ldc) {
  cublasHgemm_fn real_fn = get_real_cublasHgemm();
  if (real_fn == NULL) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  long long start = now_ns();
  cublasStatus_t st = real_fn(handle, transa, transb, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc);
  long long end = now_ns();
  log_gemm("cublasHgemm", start, end, m, n, k);
  return st;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A,
                            cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta,
                            void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo) {
  cublasGemmEx_fn real_fn = get_real_cublasGemmEx();
  if (real_fn == NULL) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  long long start = now_ns();
  cublasStatus_t st =
      real_fn(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype,
              ldb, beta, C, Ctype, ldc, computeType, algo);
  long long end = now_ns();
  log_gemm("cublasGemmEx", start, end, m, n, k);
  return st;
}

static void get_layout_rows_cols(cublasLtMatrixLayout_t layout, int* rows,
                                 int* cols) {
  unsigned long long r = 0;
  unsigned long long c = 0;
  size_t written = 0;

  if (layout != NULL) {
    (void)cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ROWS,
                                           &r, sizeof(r), &written);
    (void)cublasLtMatrixLayoutGetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_COLS,
                                           &c, sizeof(c), &written);
  }

  *rows = (int)r;
  *cols = (int)c;
}

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  cublasLtMatmul_fn real_fn = get_real_cublasLtMatmul();
  if (real_fn == NULL) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  int a_rows = -1, a_cols = -1;
  int b_rows = -1, b_cols = -1;
  int c_rows = -1, c_cols = -1;
  int m = -1, n = -1, k = -1;

  get_layout_rows_cols(Adesc, &a_rows, &a_cols);
  get_layout_rows_cols(Bdesc, &b_rows, &b_cols);
  get_layout_rows_cols(Cdesc, &c_rows, &c_cols);

  m = (c_rows > 0) ? c_rows : a_rows;
  n = (c_cols > 0) ? c_cols : b_cols;
  k = (a_cols > 0) ? a_cols : b_rows;

  long long start = now_ns();
  cublasStatus_t st =
      real_fn(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C,
              Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
  long long end = now_ns();
  log_gemm("cublasLtMatmul", start, end, m, n, k);
  return st;
}
