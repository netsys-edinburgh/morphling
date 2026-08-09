#pragma once

#include <atomic>
#include <stdexcept>
#include <string>

namespace morphling {
namespace backend {

inline constexpr int kMaxLifetimeOperationCount = 65'536;

inline void ValidateOperationId(int oid) {
  if (oid < 0 || oid >= kMaxLifetimeOperationCount) {
    throw std::out_of_range("matmul operation id " + std::to_string(oid) +
                            " is outside the valid range [0, " +
                            std::to_string(kMaxLifetimeOperationCount) + ")");
  }
}

inline int ReserveOperationId(std::atomic_int& next_oid) noexcept {
  int oid = next_oid.load(std::memory_order_relaxed);
  while (oid >= 0 && oid < kMaxLifetimeOperationCount) {
    if (next_oid.compare_exchange_weak(oid, oid + 1,
                                       std::memory_order_relaxed)) {
      return oid;
    }
  }
  return -1;
}

}  // namespace backend
}  // namespace morphling
