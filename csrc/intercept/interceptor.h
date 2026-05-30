#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "core/logger.h"

// GemmArgs is single-shape, fp32-only by design (#48). Grouped-GEMM and
// other dtypes are future extensions; reopen #48 (or a successor) when a
// consumer requires them.
enum class GemmDtype : uint8_t { kFloat32 = 0 };

struct GemmArgs {
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
  GemmDtype dtype = GemmDtype::kFloat32;

  std::string DebugString() const {
    std::string str = "GemmArgs: ";
    str += "transa=" + std::string(1, transa) + ", ";
    str += "transb=" + std::string(1, transb) + ", ";
    str += "m=" + std::to_string(m) + ", ";
    str += "n=" + std::to_string(n) + ", ";
    str += "k=" + std::to_string(k) + ", ";
    str += "alpha=" + std::to_string(alpha) + ", ";
    str += "lda=" + std::to_string(lda) + ", ";
    str += "ldb=" + std::to_string(ldb) + ", ";
    str += "beta=" + std::to_string(beta) + ", ";
    str += "ldc=" + std::to_string(ldc);
    return str;
  }
};

static_assert(sizeof(float) == 4, "GemmArgs assumes 32-bit float operands");

typedef std::unique_ptr<GemmArgs> GemmArgsPtr;

inline std::tuple<size_t, size_t, size_t> CalculateTaskSizes(
    const GemmArgs* args) {
  auto size_a = (args->transa == 'N' || args->transa == 'n')
                    ? static_cast<size_t>(args->lda) * args->k * sizeof(float)
                    : static_cast<size_t>(args->lda) * args->m * sizeof(float);
  auto size_b = (args->transb == 'N' || args->transb == 'n')
                    ? static_cast<size_t>(args->ldb) * args->n * sizeof(float)
                    : static_cast<size_t>(args->ldb) * args->k * sizeof(float);
  auto size_c = static_cast<size_t>(args->ldc) * args->n * sizeof(float);
  LOG_FATAL_IF(size_a == 0 || size_b == 0 || size_c == 0)
      << "Invalid task sizes: A: " << size_a << ", B: " << size_b
      << ", C: " << size_c;
  return {size_a, size_b, size_c};
}
