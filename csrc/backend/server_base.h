#pragma once

#include <torch/torch.h>

#include <future>

#include "common/types_and_defs.h"
#include "global_api.pb.h"
#include "morphling.pb.h"

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

typedef std::tuple<void*, int64_t> PtrData;

// define hash function for PtrData
namespace std {
template <>
struct hash<PtrData> {
  std::size_t operator()(const PtrData& p) const {
    return std::hash<void*>{}(std::get<0>(p)) ^
           std::hash<int64_t>{}(std::get<1>(p));
  }
};
}  // namespace std

typedef std::tuple<uint64_t, int64_t, int64_t, bool>
    TensorKey;  // version, pivot, r/c, is_row

// create a spdlog fmt for TensorKey

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
  uint64_t timestamp;        // timestamp on sending the message
  std::vector<PtrData> mat;  // if ptr is null and size is 0, means that this
                             // entry need to be fetched from cache first mat is
                             // row block, second mat is col block
  void* ptr_ = nullptr;      // pointer to the data
  size_t size_ = 0;          // size of the data
  std::tuple<void*, size_t> Serialize() const;
  void Deserialize(const void* data, size_t size);

  // Protobuf serialization/deserialization
  std::tuple<void*, size_t> SerializeToProto() const;
  void DeserializeFromProto(const void* data, size_t size);

  TensorKey GetRowKey() const {
    return std::make_tuple(version, pivot, row, true);
  }

  TensorKey GetColKey() const {
    return std::make_tuple(version, pivot, col, false);
  }

  int64_t Size() const {
    int64_t size = 0;
    for (const auto& m : mat) {
      size += std::get<1>(m);
    }
    return size;
  }

  std::string GetPartitionKey() const {
    // [version:pivot:row:col]
    return "[" + std::to_string(version) + ":" + std::to_string(pivot) + ":" +
           std::to_string(row) + ":" + std::to_string(col) + "]";
  }

  std::string DebugString() const;
};

typedef std::shared_ptr<MatrixPartition> MatrixPartitionPtr;

// enum TimerType { kTimerGet, kTimerPut };

// struct Timer {
//   uint8_t type;
//   uint64_t time;
// };

MatrixPartition CalculateMatrixPartition(const torch::Tensor& mat_a,
                                         const torch::Tensor& mat_b, int64_t r,
                                         int64_t c, int64_t pivot,
                                         int64_t block_size);

// The computation must be AB^T
// Every matrix is partitioned into blocks on rows since torch is row major
std::vector<MatrixPartition> PartitionMatrices(const torch::Tensor& mat_a,
                                               const torch::Tensor& mat_b,
                                               int64_t block_size);

class DataCodec {
 public:
  static std::tuple<void*, int64_t> Encode(const MatrixPartition& partition);
  static MatrixPartition Decode(const void* data, int64_t size);
};
