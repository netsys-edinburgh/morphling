#pragma once

#include "core/logger.h"

#define MAX_GEMM_DIM \
  5  // FIXME(#48): dim cap, dtype, and group-size assumptions.
struct GemmArgs {
  char transa[MAX_GEMM_DIM];
  char transb[MAX_GEMM_DIM];
  int m[MAX_GEMM_DIM];
  int n[MAX_GEMM_DIM];
  int k[MAX_GEMM_DIM];
  float alpha[MAX_GEMM_DIM];
  const float* a[MAX_GEMM_DIM];  // FIXME(#48): float-only operand assumption.
  int lda[MAX_GEMM_DIM];
  const float* b[MAX_GEMM_DIM];
  int ldb[MAX_GEMM_DIM];
  float beta[MAX_GEMM_DIM];
  float* c[MAX_GEMM_DIM];
  int ldc[MAX_GEMM_DIM];
  int group_size;  // FIXME(#48): heterogeneous group sizes not supported.

  std::string DebugString() const {
    std::string str = "GemmArgs: ";
    for (int i = 0; i < group_size; i++) {
      str += "transa: " + std::string(1, transa[i]) + ", ";
      str += "transb: " + std::string(1, transb[i]) + ", ";
      str += "m: " + std::to_string(m[i]) + ", ";
      str += "n: " + std::to_string(n[i]) + ", ";
      str += "k: " + std::to_string(k[i]) + ", ";
      str += "alpha: " + std::to_string(alpha[i]) + ", ";
      str += "lda: " + std::to_string(lda[i]) + ", ";
      str += "ldb: " + std::to_string(ldb[i]) + ", ";
      str += "beta: " + std::to_string(beta[i]) + ", ";
      str += "ldc: " + std::to_string(ldc[i]) + ", ";
    }
    return str;
  }
};

typedef std::unique_ptr<GemmArgs> GemmArgsPtr;

inline std::tuple<size_t, size_t, size_t> CalculateTaskSizes(
    const GemmArgs* args) {
  if (args->group_size == 1) {
    auto size_a = (args->transa[0] == 'N' || args->transa[0] == 'n')
                      ? (args->lda[0]) * (args->k[0]) * sizeof(float)
                      : (args->lda[0]) * (args->m[0]) * sizeof(float);
    auto size_b = (args->transb[0] == 'N' || args->transb[0] == 'n')
                      ? (args->ldb[0]) * (args->n[0]) * sizeof(float)
                      : (args->ldb[0]) * (args->k[0]) * sizeof(float);
    auto size_c = (args->ldc[0]) * (args->n[0]) * sizeof(float);
    LOG_FATAL_IF(size_a == 0 || size_b == 0 || size_c == 0)
        << "Invalid task sizes: A: " << size_a << ", B: " << size_b
        << ", C: " << size_c;
    return {size_a, size_b, size_c};
  }
  LOG_FATAL << "Grouped gemm not supported yet, group size: "
            << args->group_size;
  return {0, 0, 0};
}
