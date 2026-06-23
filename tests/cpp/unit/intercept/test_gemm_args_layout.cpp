// Issue #48: GemmArgs collapse — single-shape, fp32-only struct.
// Compiles ONLY against the post-collapse layout (scalar fields, no
// group_size).

#include <gtest/gtest.h>

#include "intercept/interceptor.h"

namespace {

TEST(GemmArgsLayout, FieldsAreScalarFp32) {
  GemmArgs args{};
  args.transa = 'T';
  args.transb = 'N';
  args.m = 64;
  args.n = 128;
  args.k = 32;
  args.alpha = 1.0f;
  args.beta = 0.0f;
  args.lda = 32;
  args.ldb = 32;
  args.ldc = 64;
  args.a = nullptr;
  args.b = nullptr;
  args.c = nullptr;
  EXPECT_EQ(args.dtype, GemmDtype::kFloat32);
  EXPECT_EQ(args.m, 64);
  EXPECT_EQ(args.alpha, 1.0f);
}

TEST(GemmArgsLayout, CalculateTaskSizesUsesScalarFields) {
  GemmArgs args{};
  args.transa = 'N';
  args.transb = 'N';
  args.m = 64;
  args.n = 128;
  args.k = 32;
  args.alpha = 1.0f;
  args.beta = 0.0f;
  args.lda = 64;
  args.ldb = 32;
  args.ldc = 64;
  args.a = reinterpret_cast<const float*>(0xdead);
  args.b = reinterpret_cast<const float*>(0xbeef);
  args.c = reinterpret_cast<float*>(0xcafe);

  auto [sa, sb, sc] = CalculateTaskSizes(&args);
  EXPECT_EQ(sa, static_cast<size_t>(64) * 32 * sizeof(float));
  EXPECT_EQ(sb, static_cast<size_t>(32) * 128 * sizeof(float));
  EXPECT_EQ(sc, static_cast<size_t>(64) * 128 * sizeof(float));
}

TEST(GemmArgsLayout, DebugStringIncludesShape) {
  GemmArgs args{};
  args.transa = 'N';
  args.transb = 'T';
  args.m = 8;
  args.n = 4;
  args.k = 2;
  const auto s = args.DebugString();
  EXPECT_NE(s.find("m=8"), std::string::npos);
  EXPECT_NE(s.find("n=4"), std::string::npos);
  EXPECT_NE(s.find("k=2"), std::string::npos);
}

}  // namespace
