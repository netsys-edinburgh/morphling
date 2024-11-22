#pragma once

#include <torch/torch.h>

#include <future>

#include "utils/logger.h"

template <typename T>
std::vector<std::vector<T>> CartesianProduct(const std::vector<T>& list) {
  std::vector<std::vector<T>> ranges;
  for (auto i : list) {
    std::vector<T> range;
    for (T j = 0; j < i; ++j) {
      range.push_back(j);
    }
    ranges.push_back(range);
  }

  std::vector<std::vector<T>> result;
  if (ranges.empty()) return result;

  // Initialize with the first list
  result.push_back({});
  for (const auto& list : ranges) {
    std::vector<std::vector<T>> temp;
    for (const auto& res : result) {
      for (const auto& elem : list) {
        std::vector<T> new_combination = res;
        new_combination.push_back(elem);
        temp.push_back(new_combination);
      }
    }
    result = std::move(temp);
  }
  return result;
}

torch::Tensor CreateOutputMatrix(const torch::Tensor& mat_a,
                                 const torch::Tensor& mat_b);

void UpdateMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                       int64_t c, int64_t pivot, int64_t block_size);
void IndexPutMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                         int64_t c, int64_t pivot, int64_t block_size);
typedef std::tuple<void*, int64_t> MatData;

struct MatrixPartition {
  int64_t row;
  int64_t col;
  int64_t pivot;
  int64_t h_dim;
  int64_t oid;
  std::vector<MatData> mat;
  // int64_t block_size;

  std::tuple<void*, int64_t> Serialize() const;
  void Deserialize(const void* data, int64_t size);
};

enum TimerType { kTimerGet, kTimerPut };

struct Timer {
  uint8_t type;
  uint64_t time;
};

MatrixPartition CalculateMatrixPartition(const torch::Tensor& mat_a,
                                         const torch::Tensor& mat_b, int64_t r,
                                         int64_t c, int64_t pivot,
                                         int64_t block_size);

// The computation must be AB^T
// Every matrix is partitioned into blocks on rows since torch is row major
std::vector<MatrixPartition> PartitionMatrices(const torch::Tensor& mat_a,
                                               const torch::Tensor& mat_b,
                                               int64_t block_size);
