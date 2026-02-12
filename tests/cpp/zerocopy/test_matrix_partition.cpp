#include <gtest/gtest.h>

#include <cstring>
#include <numeric>

#include "backend/server_base.h"

class MatrixPartitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple partition with known data
    partition_.version = 12345;
    partition_.row = 1;
    partition_.col = 2;
    partition_.pivot = 0;
    partition_.h_dim = 4;
    partition_.dev_id = 3;
    partition_.oid = 7;
    partition_.gemm_id = 99;
    partition_.timestamp = 1000000;

    // Create tensor data (simulate two matrix blocks)
    tensor_a_.resize(64);  // 64 bytes of data
    tensor_b_.resize(128);
    for (size_t i = 0; i < tensor_a_.size(); ++i) {
      tensor_a_[i] = static_cast<uint8_t>(i);
    }
    for (size_t i = 0; i < tensor_b_.size(); ++i) {
      tensor_b_[i] = static_cast<uint8_t>(255 - i);
    }

    partition_.mat.push_back({tensor_a_.data(), (int64_t)tensor_a_.size()});
    partition_.mat.push_back({tensor_b_.data(), (int64_t)tensor_b_.size()});
  }

  MatrixPartition partition_;
  std::vector<uint8_t> tensor_a_;
  std::vector<uint8_t> tensor_b_;
};

TEST_F(MatrixPartitionTest, SerializeProto_DeserializeProto_Roundtrip) {
  auto serialized = partition_.Serialize();
  ASSERT_NE(serialized, nullptr);
  ASSERT_GT(serialized->GetSize(), 0u);

  MatrixPartition deserialized;
  deserialized.Deserialize(serialized->GetBuffer(), serialized->GetSize());

  // Verify metadata
  EXPECT_EQ(deserialized.version, partition_.version);
  EXPECT_EQ(deserialized.row, partition_.row);
  EXPECT_EQ(deserialized.col, partition_.col);
  EXPECT_EQ(deserialized.pivot, partition_.pivot);
  EXPECT_EQ(deserialized.h_dim, partition_.h_dim);
  EXPECT_EQ(deserialized.dev_id, partition_.dev_id);
  EXPECT_EQ(deserialized.oid, partition_.oid);
  EXPECT_EQ(deserialized.gemm_id, partition_.gemm_id);
  EXPECT_EQ(deserialized.timestamp, partition_.timestamp);

  // Verify tensor data
  ASSERT_EQ(deserialized.mat.size(), 2u);
  auto [ptr_a, size_a] = deserialized.mat[0];
  auto [ptr_b, size_b] = deserialized.mat[1];

  EXPECT_EQ(size_a, (int64_t)tensor_a_.size());
  EXPECT_EQ(size_b, (int64_t)tensor_b_.size());

  EXPECT_EQ(memcmp(ptr_a, tensor_a_.data(), tensor_a_.size()), 0);
  EXPECT_EQ(memcmp(ptr_b, tensor_b_.data(), tensor_b_.size()), 0);
}

TEST_F(MatrixPartitionTest, SerializeZeroCopy_TotalSizeMatchesSerializeProto) {
  auto proto_buf = partition_.Serialize();
  auto sg_buf = partition_.SerializeZeroCopy();

  ASSERT_NE(proto_buf, nullptr);
  ASSERT_NE(sg_buf, nullptr);

  EXPECT_EQ(sg_buf->GetTotalSize(), proto_buf->GetSize());
}

TEST_F(MatrixPartitionTest, SerializeZeroCopy_Flatten_Deserialize_Roundtrip) {
  auto sg = partition_.SerializeZeroCopy();
  ASSERT_NE(sg, nullptr);

  // Flatten scatter-gather into contiguous buffer
  size_t total = sg->GetTotalSize();
  std::vector<uint8_t> flat(total);
  size_t offset = 0;
  for (const auto& seg : sg->GetSegments()) {
    memcpy(flat.data() + offset, seg.data, seg.size);
    offset += seg.size;
  }
  ASSERT_EQ(offset, total);

  // Deserialize from flattened buffer
  MatrixPartition deserialized;
  deserialized.Deserialize(flat.data(), flat.size());

  EXPECT_EQ(deserialized.version, partition_.version);
  EXPECT_EQ(deserialized.row, partition_.row);
  EXPECT_EQ(deserialized.col, partition_.col);
  EXPECT_EQ(deserialized.pivot, partition_.pivot);
  EXPECT_EQ(deserialized.h_dim, partition_.h_dim);
  EXPECT_EQ(deserialized.dev_id, partition_.dev_id);
  EXPECT_EQ(deserialized.oid, partition_.oid);
  EXPECT_EQ(deserialized.gemm_id, partition_.gemm_id);
  EXPECT_EQ(deserialized.timestamp, partition_.timestamp);

  ASSERT_EQ(deserialized.mat.size(), 2u);
  EXPECT_EQ(std::get<1>(deserialized.mat[0]), (int64_t)tensor_a_.size());
  EXPECT_EQ(std::get<1>(deserialized.mat[1]), (int64_t)tensor_b_.size());

  EXPECT_EQ(memcmp(std::get<0>(deserialized.mat[0]), tensor_a_.data(),
                   tensor_a_.size()),
            0);
  EXPECT_EQ(memcmp(std::get<0>(deserialized.mat[1]), tensor_b_.data(),
                   tensor_b_.size()),
            0);
}

TEST_F(MatrixPartitionTest, ZeroCopy_TensorReferencePointsToOriginalData) {
  auto sg = partition_.SerializeZeroCopy();
  ASSERT_NE(sg, nullptr);

  const auto& segments = sg->GetSegments();
  // First segment is header (owned), remaining are tensor references
  ASSERT_GE(segments.size(), 3u);  // header + 2 tensor segments

  // Tensor segments should reference original data directly
  EXPECT_EQ(segments[1].data, tensor_a_.data());
  EXPECT_EQ(segments[1].size, tensor_a_.size());
  EXPECT_FALSE(segments[1].owned);

  EXPECT_EQ(segments[2].data, tensor_b_.data());
  EXPECT_EQ(segments[2].size, tensor_b_.size());
  EXPECT_FALSE(segments[2].owned);
}

TEST_F(MatrixPartitionTest, ZeroCopy_HeaderSegmentOwned) {
  auto sg = partition_.SerializeZeroCopy();
  const auto& segments = sg->GetSegments();
  ASSERT_GE(segments.size(), 1u);

  // Header segment should be owned
  EXPECT_TRUE(segments[0].owned);
}

TEST_F(MatrixPartitionTest, EmptyMatrices_Serialize_Deserialize) {
  MatrixPartition empty;
  empty.version = 1;
  empty.row = 0;
  empty.col = 0;
  empty.pivot = 0;
  empty.h_dim = 0;
  empty.dev_id = 0;
  empty.oid = 0;
  empty.gemm_id = 0;
  empty.timestamp = 0;
  // No matrices

  auto serialized = empty.Serialize();
  ASSERT_NE(serialized, nullptr);

  MatrixPartition deserialized;
  deserialized.Deserialize(serialized->GetBuffer(), serialized->GetSize());

  EXPECT_EQ(deserialized.version, 1u);
  EXPECT_EQ(deserialized.mat.size(), 0u);
}

TEST_F(MatrixPartitionTest, UtilityMethods) {
  // DebugString
  std::string debug = partition_.DebugString();
  EXPECT_FALSE(debug.empty());
  EXPECT_NE(debug.find("12345"), std::string::npos);  // version

  // GetPartitionKey
  std::string key = partition_.GetPartitionKey();
  EXPECT_EQ(key, "[12345:0:1:2]");

  // GetRowKey
  auto row_key = partition_.GetRowKey();
  EXPECT_EQ(std::get<0>(row_key), 12345u);
  EXPECT_EQ(std::get<1>(row_key), 0);  // pivot
  EXPECT_EQ(std::get<2>(row_key), 1);  // row
  EXPECT_EQ(std::get<3>(row_key), true);

  // GetColKey
  auto col_key = partition_.GetColKey();
  EXPECT_EQ(std::get<0>(col_key), 12345u);
  EXPECT_EQ(std::get<1>(col_key), 0);  // pivot
  EXPECT_EQ(std::get<2>(col_key), 2);  // col
  EXPECT_EQ(std::get<3>(col_key), false);

  // Size
  EXPECT_EQ(partition_.Size(), (int64_t)(tensor_a_.size() + tensor_b_.size()));
}

TEST_F(MatrixPartitionTest, GetMessageType) {
  EXPECT_EQ(partition_.GetMessageType(),
            morphling::global_api::COMPUTE_GEMM_DATA);
}

TEST_F(MatrixPartitionTest, DeserializeProto_InvalidData_Throws) {
  MatrixPartition mp;

  // Null pointer
  EXPECT_THROW(mp.Deserialize(nullptr, 100), std::runtime_error);

  // Size too small
  uint8_t small_buf[8] = {};
  EXPECT_THROW(mp.Deserialize(small_buf, sizeof(small_buf)),
               std::runtime_error);
}
