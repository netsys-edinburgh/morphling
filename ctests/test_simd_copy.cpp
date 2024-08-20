// gtest for parallel copy

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

#include "csrc/memory/simd_copy.cpp"
#include "csrc/memory/simd_copy.h"

TEST(TestSimdCopy, TestNonSIMDCopy) {
  size_t size = 1024 * 1024 * 1024;
  float* src = new float[size];
  float* dest = new float[size];
  for (size_t i = 0; i < size; i++) {
    src[i] = i;
  }
  auto start = std::chrono::high_resolution_clock::now();
  memcpy(dest, src, size * sizeof(float));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(dest[i], src[i]);
  }
  delete[] src;
  delete[] dest;
}

TEST(TestSimdCopy, TestSimdCopy1) {
  size_t size = 1024 * 1024 * 1024;
  float* src = new float[size];
  float* dest = new float[size];
  for (size_t i = 0; i < size; i++) {
    src[i] = i;
  }
  auto start = std::chrono::high_resolution_clock::now();
  helper_memcpy_1(dest, src, size);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(dest[i], src[i]);
  }
  delete[] src;
  delete[] dest;
}

TEST(TestSimdCopy, TestSimdCopy4) {
  size_t size = 1024 * 1024 * 1024;
  float* src = new float[size];
  float* dest = new float[size];
  for (size_t i = 0; i < size; i++) {
    src[i] = i;
  }
  auto start = std::chrono::high_resolution_clock::now();
  helper_memcpy_4(dest, src, size);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(dest[i], src[i]);
  }
  delete[] src;
  delete[] dest;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}