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

typedef std::tuple<uint64_t, int64_t, int64_t, bool>
    TensorKey;  // version, pivot, r/c, is_row

// define custom hash function for TensorKey
namespace std {
template <>
struct hash<TensorKey> {
  std::size_t operator()(const TensorKey& k) const {
    return std::hash<uint64_t>{}(std::get<0>(k)) ^
           std::hash<int64_t>{}(std::get<1>(k)) ^
           std::hash<int64_t>{}(std::get<2>(k)) ^
           std::hash<bool>{}(std::get<3>(k));
  }
};
}  // namespace std

struct MatrixPartition {
  uint64_t version;
  int64_t row;
  int64_t col;
  int64_t pivot;             // leading dimension
  int64_t h_dim;             // hidden dimension
  int64_t dev_id;            // device id
  int64_t oid;               // output matrix id for parallel matmul
  std::vector<MatData> mat;  // if ptr is null and size is 0, means that this
                             // entry need to be fetched from cache first mat is
                             // row block, second mat is col block
  std::tuple<void*, int64_t> Serialize() const;
  void Deserialize(const void* data, int64_t size);

  TensorKey GetRowKey() const {
    return std::make_tuple(version, pivot, row, true);
  }

  TensorKey GetColKey() const {
    return std::make_tuple(version, pivot, col, false);
  }

  std::string GetPartitionKey() const {
    // [version:pivot:row:col]
    return "[" + std::to_string(version) + ":" + std::to_string(pivot) + ":" +
           std::to_string(row) + ":" + std::to_string(col) + "]";
  }

  std::string DebugString() const;
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
