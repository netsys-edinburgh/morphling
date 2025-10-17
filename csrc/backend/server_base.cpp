#include "server_base.h"

#include <arpa/inet.h>
#include <sys/mman.h>

#include "common/generator.h"
#include "global_api.pb.h"
#include "utils/logging.h"

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

  // LOG_DEBUG(
  //     "Creating output matrix, A shape: {}, B shape: {}, Output shape: {}",
  //     a_shape, b_shape, c_shape);

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
  // partition.version = 0;  // need to set version
  // partition.oid = -1;     // need to set oid
  partition.row = r;
  partition.col = c;
  partition.h_dim = h_dim;
  partition.pivot = pivot;
  // partition.dev_id = -1;
  partition.timestamp = CurrentTimeMicros();
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

std::tuple<void*, size_t> MatrixPartition::Serialize() const {
  uint32_t size = sizeof(int64_t) * 6 + sizeof(uint64_t) * 2;
  for (const auto& mat : mat) {
    size += std::get<1>(mat) + sizeof(int64_t);
  }
  // fprintf(stderr, "Size: %ld\n", size);

  uint8_t* ptr = (uint8_t*)malloc(size + sizeof(uint32_t));
  // // pinning the pointer
  // int ret = mlock(ptr, size);
  // LOG_FATAL_IF(ret != 0, "Failed to pin memory in serialization, error code:
  // {}, msg: {}", ret, strerror(errno));

  // write payload size
  size_t nl_size = htonl(size);
  memcpy(ptr, &nl_size, sizeof(uint32_t));

  int64_t offset = sizeof(uint32_t);
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
  memcpy(ptr + offset, &timestamp, sizeof(uint64_t));
  offset += sizeof(uint64_t);

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

  return std::make_tuple(ptr, size + sizeof(uint32_t));
}

void MatrixPartition::Deserialize(const void* data, size_t size) {
  LOG_INFO << "Deserialize called: data=" << data << ", size=" << size;

  if (data == nullptr) {
    LOG_ERROR << "Deserialize: data pointer is nullptr!";
    return;
  }

  if (size <= 0) {
    LOG_ERROR << "Deserialize: invalid size=" << size;
    return;
  }

  uint8_t* ptr = (uint8_t*)data;
  int64_t offset = sizeof(uint32_t);

  ptr_ = ptr;
  size_ = size;

  LOG_DEBUG << "Reading version at offset=" << offset;
  memcpy(&version, ptr + offset, sizeof(uint64_t));
  offset += sizeof(uint64_t);
  LOG_DEBUG << "version=" << version;

  LOG_DEBUG << "Reading row at offset=" << offset;
  memcpy(&row, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  LOG_DEBUG << "row=" << row;

  LOG_DEBUG << "Reading col at offset=" << offset;
  memcpy(&col, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  LOG_DEBUG << "col=" << col;

  LOG_DEBUG << "Reading pivot at offset=" << offset;
  memcpy(&pivot, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  LOG_DEBUG << "pivot=" << pivot;

  LOG_DEBUG << "Reading h_dim at offset=" << offset;
  memcpy(&h_dim, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  LOG_DEBUG << "h_dim=" << h_dim;

  LOG_DEBUG << "Reading dev_id at offset=" << offset;
  memcpy(&dev_id, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  LOG_DEBUG << "dev_id=" << dev_id;

  LOG_DEBUG << "Reading oid at offset=" << offset;
  memcpy(&oid, ptr + offset, sizeof(int64_t));
  offset += sizeof(int64_t);
  LOG_DEBUG << "oid=" << oid;

  LOG_DEBUG << "Reading timestamp at offset=" << offset;
  memcpy(&timestamp, ptr + offset, sizeof(uint64_t));
  offset += sizeof(uint64_t);
  LOG_DEBUG << "timestamp=" << timestamp;

  LOG_DEBUG << "Starting matrix data loop, offset=" << offset
            << ", size=" << size;
  int mat_count = 0;
  while (offset < size) {
    LOG_DEBUG << "Reading mat_size at offset=" << offset
              << ", remaining=" << (size - offset);

    if (offset + sizeof(int64_t) > size) {
      LOG_ERROR << "Not enough data for mat_size at offset=" << offset;
      break;
    }

    int64_t mat_size;
    memcpy(&mat_size, ptr + offset, sizeof(int64_t));
    offset += sizeof(int64_t);
    LOG_DEBUG << "mat[" << mat_count << "] size=" << mat_size;

    if (mat_size == 0) {
      mat.push_back({nullptr, 0});
      mat_count++;
      continue;
    }

    if (mat_size < 0) {
      LOG_ERROR << "Invalid mat_size=" << mat_size << " at mat[" << mat_count
                << "]";
      break;
    }

    if (offset + mat_size > size) {
      LOG_ERROR << "Not enough data for matrix at offset=" << offset
                << ", mat_size=" << mat_size
                << ", remaining=" << (size - offset);
      break;
    }

    LOG_DEBUG << "Adding matrix data: ptr=" << (void*)(ptr + offset)
              << ", size=" << mat_size;
    mat.push_back({ptr + offset, mat_size});
    offset += mat_size;
    mat_count++;
  }

  LOG_INFO << "Deserialize completed: parsed " << mat_count
           << " matrices, final offset=" << offset;
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

std::tuple<void*, size_t> MatrixPartition::SerializeToProto() const {
  // Create UMessage with ComputeGemmRequest
  morphling::UMessage umsg;

  // Set header fields
  auto* head = umsg.mutable_head();
  head->set_version(1);
  head->set_magic_flag(0x12340987);
  head->set_random_num(0);
  head->set_flow_no(0);
  head->set_session_no("");
  head->set_message_type(morphling::global_api::COMPUTE_GEMM_DATA);

  // Set body with ComputeGemmRequest extension
  auto* body = umsg.mutable_body();
  auto* gemm_data =
      body->MutableExtension(morphling::global_api::compute_gemm_data);

  gemm_data->set_version(version);
  gemm_data->set_row(row);
  gemm_data->set_col(col);
  gemm_data->set_pivot(pivot);
  gemm_data->set_h_dim(h_dim);
  gemm_data->set_dev_id(dev_id);
  gemm_data->set_oid(oid);
  gemm_data->set_timestamp(timestamp);

  for (const auto& m : mat) {
    auto* mat_payload = gemm_data->add_matrices();  // use matrix_a for all mats
    mat_payload->set_offset(0);  // actual offset handled in payload
    mat_payload->set_size(std::get<1>(m));
  }

  // Serialize proto
  std::string proto_str = umsg.SerializeAsString();
  uint32_t proto_size = proto_str.size();

  // Calculate total tensor size (sum of all matrix data)
  uint64_t tensor_size = 0;
  for (const auto& m : mat) {
    tensor_size += std::get<1>(m);
  }

  // Layout: [4B: payload_size] [4B: proto_size] [8B: tensor_size] [proto...]
  // [tensors...] payload_size does NOT include the 4-byte size field itself
  // (compatible with network protocol)
  uint32_t payload_size = sizeof(proto_size) + sizeof(tensor_size) +
                          proto_size +
                          tensor_size;  // proto_size + tensor_size + header
  uint64_t total_size =
      sizeof(payload_size) + payload_size;  // 8 + payload_size
  uint8_t* ptr = (uint8_t*)malloc(total_size);

  // Write payload size in network byte order (compatible with receiver's
  // protocol) [0-3]: payload_size (4 bytes, network byte order - BIG_ENDIAN with htonl)
  uint32_t payload_size_be = htonl(payload_size);
  memcpy(ptr, &payload_size_be, sizeof(payload_size_be));
  uint32_t offset = sizeof(payload_size_be);

  // Write header in NATIVE byte order (LITTLE_ENDIAN on x86/ARM, NO byte order conversion!)
  // [4-7]: proto_size (4 bytes, LITTLE_ENDIAN - native byte order, NO htonl!)
  // [8-15]: tensor_size (8 bytes, LITTLE_ENDIAN - native byte order, NO conversion!)
  memcpy(ptr + offset, &proto_size, sizeof(uint32_t));
  offset += sizeof(proto_size);
  memcpy(ptr + offset, &tensor_size, sizeof(uint64_t));
  offset += sizeof(tensor_size);

  // Write proto data at offset 16
  // [16-...]: serialized protobuf message
  memcpy(ptr + offset, proto_str.data(), proto_size);
  offset += proto_size;

  // Write tensor data starting at offset 16 + proto_size
  // [16+proto_size-...]: matrix data (A, then B, etc.)
  for (const auto& m : mat) {
    if (std::get<1>(m) > 0 && std::get<0>(m) != nullptr) {
      memcpy(ptr + offset, std::get<0>(m), std::get<1>(m));
      offset += std::get<1>(m);
    }
  }

  LOG_DEBUG << "SerializeToProto completed: total_size=" << total_size
            << ", payload_size=" << payload_size
            << ", proto_size=" << proto_size << ", tensor_size=" << tensor_size;
  return std::make_tuple(ptr, total_size);
}

void MatrixPartition::DeserializeFromProto(const void* data, size_t size) {
  LOG_INFO << "DeserializeFromProto called: data=" << data << ", size=" << size;

  if (data == nullptr) {
    LOG_ERROR << "DeserializeFromProto: data pointer is nullptr!";
    throw std::runtime_error("DeserializeFromProto: data pointer is nullptr");
  }

  if (size < 16) {
    LOG_ERROR << "DeserializeFromProto: size too small, size=" << size;
    throw std::runtime_error("DeserializeFromProto: size too small");
  }

  uint8_t* ptr = (uint8_t*)data;
  uint32_t offset = sizeof(uint32_t);

  // Read header in MIXED byte order (matching SerializeToProto format):
  // [0-3]: payload_size (already skipped, was in BIG_ENDIAN with htonl)
  // [4-7]: proto_size (4 bytes, LITTLE_ENDIAN - native byte order, NO conversion!)
  // [8-15]: tensor_size (8 bytes, LITTLE_ENDIAN - native byte order, NO conversion!)
  uint32_t proto_size;
  size_t tensor_size;

  memcpy(&proto_size, ptr + offset, sizeof(proto_size));
  offset += sizeof(proto_size);
  memcpy(&tensor_size, ptr + offset, sizeof(tensor_size));
  offset += sizeof(tensor_size);

  LOG_DEBUG << "Read header: proto_size=" << proto_size
            << ", tensor_size=" << tensor_size;

  // Validate sizes
  if (proto_size == 0 ||
      proto_size > 100 * 1024 * 1024)  // 100MB sanity check
  {
    LOG_ERROR << "Invalid proto_size=" << proto_size;
    throw std::runtime_error("Invalid proto_size");
  }

  if (tensor_size > 1ull * 1024 * 1024 * 1024) {  // 1GB sanity check
    LOG_ERROR << "Invalid tensor_size=" << tensor_size;
    throw std::runtime_error("Invalid tensor_size");
  }

  // Validate: payload_size + proto + tensors should fit in buffer
  if (offset + proto_size + tensor_size > size) {
    LOG_ERROR << "Total size mismatch: " << (offset + proto_size + tensor_size)
              << " > " << size;
    throw std::runtime_error("Total size exceeds provided buffer");
  }

  // Parse proto
  LOG_DEBUG << "Parsing protobuf at offset 16, proto_size=" << proto_size;
  morphling::UMessage umsg;
  if (!umsg.ParseFromArray(ptr + offset, proto_size)) {
    LOG_ERROR << "Failed to parse UMessage from proto";
    throw std::runtime_error("Failed to parse UMessage from proto");
  }
  offset += proto_size;
  LOG_DEBUG << "Protobuf parsed successfully";

  // Extract ComputeGemmRequest
  const auto& body = umsg.body();
  if (!body.HasExtension(morphling::global_api::compute_gemm_data)) {
    LOG_ERROR << "UMessage does not contain compute_gemm_data";
    throw std::runtime_error("UMessage does not contain compute_gemm_data");
  }

  const auto& gemm_data =
      body.GetExtension(morphling::global_api::compute_gemm_data);
  LOG_DEBUG << "ComputeGemmRequest extracted";

  // Fill MatrixPartition fields
  version = gemm_data.version();
  row = gemm_data.row();
  col = gemm_data.col();
  pivot = gemm_data.pivot();
  h_dim = gemm_data.h_dim();
  dev_id = gemm_data.dev_id();
  oid = gemm_data.oid();
  timestamp = gemm_data.timestamp();

  LOG_INFO << "Partition fields: version=" << version << ", row=" << row
           << ", col=" << col << ", pivot=" << pivot << ", h_dim=" << h_dim
           << ", dev_id=" << dev_id << ", oid=" << oid
           << ", timestamp=" << timestamp;

  // // Extract tensor data pointers
  // uint64_t tensor_offset = offset + proto_size;
  // LOG_DEBUG << "Tensor data starts at offset=" << tensor_offset;

  mat.clear();
  for (const auto& payload : gemm_data.matrices()) {
    LOG_DEBUG << "Matrix payload: size=" << payload.size();
    mat.push_back({ptr + offset, payload.size()});
    offset += payload.size();
  }

  ptr_ = ptr;
  size_ = size;

  LOG_INFO << "DeserializeFromProto completed: parsed " << mat.size()
           << " matrices";
}
