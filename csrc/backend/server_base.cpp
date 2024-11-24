#include "server_base.h"

#include <sys/mman.h>

#include "common/generator.h"

void IndexPutMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                         int64_t c, int64_t pivot, int64_t block_size) {
  // implement torch::index_put_ using memory copy
  void* target_ptr = target.data_ptr();
  void* mat_ptr = mat.data_ptr();

  auto mat_shape = mat.sizes().vec();
  auto target_shape = target.sizes().vec();

  int64_t mat_n_rows = mat_shape[mat_shape.size() - 2];
  int64_t mat_n_cols = mat_shape[mat_shape.size() - 1];

  int64_t elem_size = mat.element_size();

  int64_t in_dim = target_shape[target_shape.size() - 2];
  int64_t out_dim = target_shape[target_shape.size() - 1];

  int64_t offset_r = r * block_size * out_dim * elem_size;
  int64_t offset_c = c * block_size * elem_size;
  int64_t target_offset =
      pivot * in_dim * out_dim * elem_size + offset_r + offset_c;

  for (int64_t i = 0; i < mat_n_rows; ++i) {
    int64_t mat_row_offset = i * mat_n_cols * elem_size;
    int64_t target_row_offset = target_offset + i * out_dim * elem_size;
    // LOG_DEBUG("IndexPutMatrixBlock, target_row_offset: {}, mat_row_offset:
    // {}",
    //           target_row_offset, mat_row_offset);
    memcpy((char*)target_ptr + target_row_offset,
           (char*)mat_ptr + mat_row_offset, mat_n_cols * elem_size);
  }
}

torch::Tensor CreateOutputMatrix(const torch::Tensor& mat_a,
                                 const torch::Tensor& mat_b) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  // assume b needs to be transposed
  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t out_dim = b_shape[b_shape.size() - 2];

  std::vector<int64_t> c_shape;
  if (a_shape.size() == 2 && b_shape.size() == 2) {
    c_shape = {in_dim, out_dim};
  } else if (a_shape.size() > 2 && b_shape.size() == 2) {
    c_shape.insert(c_shape.end(), a_shape.begin(), a_shape.end() - 2);
    c_shape.push_back(in_dim);
    c_shape.push_back(out_dim);
  } else {
    auto lda_shape = std::vector<int64_t>(a_shape.begin(), a_shape.end() - 2);
    auto ldb_shape = std::vector<int64_t>(b_shape.begin(), b_shape.end() - 2);
    if (lda_shape != ldb_shape) {
      throw std::runtime_error("Input dimensions must be the same");
    }
    c_shape = lda_shape;
    c_shape.push_back(in_dim);
    c_shape.push_back(out_dim);
  }

  LOG_DEBUG(
      "Creating output matrix, A shape: {}, B shape: {}, Output shape: {}",
      a_shape, b_shape, c_shape);

  auto output_matrix = torch::empty(c_shape);
  // fill with nan
  output_matrix.fill_(std::nan(""));

  return output_matrix;
}

void UpdateMatrixBlock(torch::Tensor& target, torch::Tensor& mat, int64_t r,
                       int64_t c, int64_t pivot, int64_t block_size) {
  auto target_shape = target.sizes().vec();
  auto mat_shape = mat.sizes().vec();

  // LOG_DEBUG("Updating matrix block, target shape: {}, mat shape: {}",
  //           target_shape, mat_shape);

  IndexPutMatrixBlock(target, mat, r, c, pivot, block_size);

  // if (target_shape.size() == 2) {
  //   // no need to reshape
  //   auto offset_r = r * block_size;
  //   auto offset_c = c * block_size;

  //   auto end_r = std::min(offset_r + block_size, target_shape[0]);
  //   auto end_c = std::min(offset_c + block_size, target_shape[1]);

  //   target.index_put_({torch::indexing::Slice(offset_r, end_r),
  //                      torch::indexing::Slice(offset_c, end_c)},
  //                     mat);
  // } else {
  //   auto num_ld = target_shape.size() - 2;
  //   auto offset_r = r * block_size;
  //   auto offset_c = c * block_size;

  //   auto end_r =
  //       std::min(offset_r + block_size, target_shape[target_shape.size() -
  //       2]);
  //   auto end_c =
  //       std::min(offset_c + block_size, target_shape[target_shape.size() -
  //       1]);

  //   if (num_ld == 1) {
  //     LOG_DEBUG(
  //         "Updating matrix block, offset_r: {}, end_r: {}, offset_c: {}, "
  //         "end_c: {}",
  //         offset_r, end_r, offset_c, end_c);
  //     target.index_put_({pivot, torch::indexing::Slice(offset_r, end_r),
  //                        torch::indexing::Slice(offset_c, end_c)},
  //                       mat);
  //   } else {
  //     auto ld_combinations = CartesianProduct(
  //         std::vector<int64_t>(target_shape.begin(), target_shape.end() -
  //         2));
  //     auto ld_vec = torch::tensor(ld_combinations[pivot]);
  //     target.index_put_({ld_vec, torch::indexing::Slice(offset_r, end_r),
  //                        torch::indexing::Slice(offset_c, end_c)},
  //                       mat);
  //   }
  // }
  // LOG_DEBUG("Updated matrix block, r: {}, c: {}, pivot: {}", r, c, pivot);
}

MatrixPartition CalculateMatrixPartition(const torch::Tensor& mat_a,
                                         const torch::Tensor& mat_b, int64_t r,
                                         int64_t c, int64_t pivot,
                                         int64_t block_size) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t h_dim = a_shape[a_shape.size() - 1];
  int64_t out_dim = b_shape[b_shape.size() - 2];

  int64_t offset_r =
      (r * block_size + pivot * in_dim) * h_dim * mat_a.element_size();
  int64_t offset_c = (c * block_size +
                      (mat_b.sizes().vec().size() > 2 ? pivot : 0) * out_dim) *
                     h_dim * mat_b.element_size();

  // fprintf(stderr, "offset_r: %ld, offset_c: %ld, r: %ld, c: %ld, pivot:
  // %ld\n",
  //         offset_r, offset_c, r, c, pivot);
  void* offset_r_ptr = (char*)mat_a.data_ptr() + offset_r;
  void* offset_c_ptr = (char*)mat_b.data_ptr() + offset_c;

  // int64_t a_bytes = in_dim * h_dim * mat_a.element_size();
  // int64_t b_bytes = h_dim * out_dim * mat_b.element_size();

  int64_t size_r = std::min(block_size, in_dim - r * block_size) * h_dim *
                   mat_a.element_size();
  int64_t size_c = std::min(block_size, out_dim - c * block_size) * h_dim *
                   mat_b.element_size();

  // fprintf(stderr, "size_r: %ld, size_c: %ld, a_bytes: %ld, b_bytes: %ld\n",
  // size_r, size_c, a_bytes, b_bytes);

  MatrixPartition partition;
  partition.version = 0;  // need to set version
  partition.oid = -1;     // need to set oid
  partition.row = r;
  partition.col = c;
  partition.h_dim = h_dim;
  partition.pivot = pivot;
  partition.dev_id = -1;
  partition.mat.push_back({offset_r_ptr, size_r});
  partition.mat.push_back({offset_c_ptr, size_c});
  // partition.block_size = block_size;

  return partition;
}

std::vector<MatrixPartition> PartitionMatrices(const torch::Tensor& mat_a,
                                               const torch::Tensor& mat_b,
                                               int64_t block_size) {
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  assert(a_shape.size() >= b_shape.size());

  int64_t in_dim = a_shape[a_shape.size() - 2];
  int64_t h_dim = a_shape[a_shape.size() - 1];
  int64_t out_dim = b_shape[b_shape.size() - 2];

  std::vector<MatrixPartition> partitions;
  auto uuid64 = GenUUID64();

  int64_t num_block_rows = in_dim / block_size + (in_dim % block_size != 0);
  int64_t num_block_cols = out_dim / block_size + (out_dim % block_size != 0);

  for (int r = 0; r < num_block_rows; ++r) {
    for (int c = 0; c < num_block_cols; ++c) {
      int64_t pivot = 0;
      if (mat_a.dim() == 2 && mat_b.dim() == 2) {
        partitions.push_back(
            CalculateMatrixPartition(mat_a, mat_b, r, c, pivot, block_size));
      } else {
        auto ld_combinations = CartesianProduct(
            std::vector<int64_t>(a_shape.begin(), a_shape.end() - 2));
        for (const auto ld : ld_combinations) {
          partitions.push_back(
              CalculateMatrixPartition(mat_a, mat_b, r, c, pivot, block_size));
          // fprintf(stderr, "r: %d, c: %d, pivot: %ld\n", r, c, pivot);
          pivot++;
        }
      }
    }
  }

  // set version to same uuid64
  for (auto& partition : partitions) {
    partition.version = uuid64;
  }

  return partitions;
}

std::tuple<void*, int64_t> MatrixPartition::Serialize() const {
  int64_t size = sizeof(int64_t) * 6 + sizeof(uint64_t);
  for (const auto& mat : mat) {
    size += std::get<1>(mat) + sizeof(int64_t);
  }

  // fprintf(stderr, "Size: %ld\n", size);

  uint8_t* ptr = (uint8_t*)malloc(size);
  // // pinning the pointer
  // int ret = mlock(ptr, size);
  // LOG_FATAL_IF(ret != 0, "Failed to pin memory in serialization, error code:
  // {}, msg: {}", ret, strerror(errno));

  int64_t offset = 0;
  // fprintf(stderr, "Serializing partition: %ld, %ld, %ld\n", row, col, h_dim);
  memcpy(ptr + offset, &version, sizeof(uint64_t));
  offset += sizeof(uint64_t);
  memcpy(ptr + offset, &row, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(ptr + offset, &col, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(ptr + offset, &pivot, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(ptr + offset, &h_dim, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(ptr + offset, &dev_id, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(ptr + offset, &oid, sizeof(int64_t));
  offset += sizeof(int64_t);

  for (const auto& mat : mat) {
    memcpy(ptr + offset, &std::get<1>(mat), sizeof(int64_t));
    offset += sizeof(int64_t);
    // fprintf(stderr, "ptr: %p, offset: %ld, size: %ld\n", ptr, offset,
    // std::get<1>(mat));
    if (std::get<1>(mat) == 0) {
      continue;
    }
    memcpy(ptr + offset, std::get<0>(mat), std::get<1>(mat));
    offset += std::get<1>(mat);
    // fprintf(stderr, "Mat size: %ld\n", std::get<1>(mat));
  }

  return std::make_tuple(ptr, size);
}

void MatrixPartition::Deserialize(const void* data, int64_t size) {
  uint8_t* ptr = (uint8_t*)data;
  int64_t offset = 0;

  // if (mlock(ptr, size) != 0) {
  //   LOG_FATAL("Failed to pin memory in deserialization");
  // }

  memcpy(&version, ptr + offset, sizeof(uint64_t));
  offset += sizeof(uint64_t);
  memcpy(&row, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(&col, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(&pivot, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(&h_dim, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(&dev_id, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  memcpy(&oid, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);

  // fprintf(stderr, "Deserialized partition: %ld, %ld, %ld, %ld, %ld, %ld\n",
  // version, row,
  //         col, pivot, h_dim, dev_id);

  while (offset < size) {
    int64_t mat_size;
    memcpy(&mat_size, ptr + offset, sizeof(int64_t));
    offset += sizeof(int64_t);
    if (mat_size == 0) {
      mat.push_back({nullptr, 0});
      continue;
    }

    mat.push_back({ptr + offset, mat_size});
    offset += mat_size;
  }

  // fprintf(stderr, "Deserialized partition, size: %ld\n", mat.size());
}

std::string MatrixPartition::DebugString() const {
  std::ostringstream oss;
  oss << "v: " << version << ", r: " << row << ", c: " << col
      << ", p: " << pivot << ", h: " << h_dim << ", dev_id: " << dev_id
      << ", oid: " << oid;

  // add mat data
  for (const auto& mat_data : mat) {
    oss << ", m_size: " << std::get<1>(mat_data);
  }
  return oss.str();
}
