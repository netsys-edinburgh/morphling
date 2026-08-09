#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>

#include "backend/operation_id.h"

namespace {

using morphling::backend::kMaxLifetimeOperationCount;
using morphling::backend::ReserveOperationId;
using morphling::backend::ValidateOperationId;

TEST(OperationIdTest, WaitMatMulRejectsNegativeOid) {
  EXPECT_THROW(ValidateOperationId(-1), std::out_of_range);
}

TEST(OperationIdTest, WaitMatMulRejectsOidAtCapacity) {
  EXPECT_THROW(ValidateOperationId(kMaxLifetimeOperationCount),
               std::out_of_range);
}

TEST(OperationIdTest, ReservesLastAvailableOidWithoutExceedingCapacity) {
  std::atomic_int next_oid{kMaxLifetimeOperationCount - 1};

  EXPECT_EQ(ReserveOperationId(next_oid), kMaxLifetimeOperationCount - 1);
  EXPECT_EQ(ReserveOperationId(next_oid), -1);
  EXPECT_EQ(next_oid.load(), kMaxLifetimeOperationCount);
}

}
