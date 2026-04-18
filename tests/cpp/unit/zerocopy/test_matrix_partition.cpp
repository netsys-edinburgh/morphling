#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "backend/server_base.h"

namespace {

MatrixPartition CreateBasePartition() {
  MatrixPartition partition;
  partition.version = 123;
  partition.row = 1;
  partition.col = 2;
  partition.pivot = 3;
  partition.h_dim = 128;
  partition.dev_id = 4;
  partition.oid = 5;
  partition.gemm_id = 6;
  partition.timestamp = 777;
  return partition;
}

std::vector<uint8_t> FlattenScatterGather(const ScatterGatherBuffer& sg) {
  std::vector<uint8_t> bytes;
  bytes.reserve(sg.GetTotalSize());
  for (const auto& seg : sg.GetSegments()) {
    const auto* ptr = static_cast<const uint8_t*>(seg.data);
    bytes.insert(bytes.end(), ptr, ptr + seg.size);
  }
  return bytes;
}

}  // namespace

TEST(MatrixPartitionSerializationTest, ProtoRoundTripWithShmReferences) {
  MatrixPartition partition = CreateBasePartition();
  partition.mat.push_back({nullptr, 0});
  partition.mat.push_back({nullptr, 0});

  partition.shm_refs_.push_back({"/morphling_shm_a", 4096, 64, 1024});
  partition.shm_refs_.push_back({"/morphling_shm_b", 8192, 128, 2048});

  auto serialized = partition.Serialize();

  MatrixPartition decoded;
  decoded.Deserialize(serialized->GetBuffer(), serialized->GetSize());

  ASSERT_EQ(decoded.shm_refs_.size(), 2u);
  EXPECT_EQ(decoded.shm_refs_[0].shm_name, "/morphling_shm_a");
  EXPECT_EQ(decoded.shm_refs_[0].segment_size, 4096u);
  EXPECT_EQ(decoded.shm_refs_[0].offset, 64u);
  EXPECT_EQ(decoded.shm_refs_[0].tensor_size, 1024u);

  EXPECT_EQ(decoded.shm_refs_[1].shm_name, "/morphling_shm_b");
  EXPECT_EQ(decoded.shm_refs_[1].segment_size, 8192u);
  EXPECT_EQ(decoded.shm_refs_[1].offset, 128u);
  EXPECT_EQ(decoded.shm_refs_[1].tensor_size, 2048u);

  ASSERT_EQ(decoded.mat.size(), 2u);
  EXPECT_EQ(std::get<0>(decoded.mat[0]), nullptr);
  EXPECT_EQ(std::get<1>(decoded.mat[0]), 0);
  EXPECT_EQ(std::get<0>(decoded.mat[1]), nullptr);
  EXPECT_EQ(std::get<1>(decoded.mat[1]), 0);
}

TEST(MatrixPartitionSerializationTest,
     ZeroCopyRoundTripWithShmReferencesInProtoHeader) {
  MatrixPartition partition = CreateBasePartition();
  partition.mat.push_back({nullptr, 0});
  partition.mat.push_back({nullptr, 0});
  partition.shm_refs_.push_back({"/morphling_shm_zero_copy", 16384, 256, 4096});

  auto sg = partition.SerializeZeroCopy();
  auto flattened = FlattenScatterGather(*sg);

  MatrixPartition decoded;
  decoded.Deserialize(flattened.data(), flattened.size());

  ASSERT_EQ(decoded.shm_refs_.size(), 1u);
  EXPECT_EQ(decoded.shm_refs_[0].shm_name, "/morphling_shm_zero_copy");
  EXPECT_EQ(decoded.shm_refs_[0].segment_size, 16384u);
  EXPECT_EQ(decoded.shm_refs_[0].offset, 256u);
  EXPECT_EQ(decoded.shm_refs_[0].tensor_size, 4096u);

  ASSERT_EQ(decoded.mat.size(), 1u);
  EXPECT_EQ(std::get<0>(decoded.mat[0]), nullptr);
  EXPECT_EQ(std::get<1>(decoded.mat[0]), 0);
}

TEST(MatrixPartitionSerializationTest, LegacyPayloadRoundTripUnaffected) {
  MatrixPartition partition = CreateBasePartition();

  std::vector<uint8_t> tensor_a = {0x01, 0x02, 0x03, 0x04};
  std::vector<uint8_t> tensor_b = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE};
  partition.mat.push_back(
      {tensor_a.data(), static_cast<int64_t>(tensor_a.size())});
  partition.mat.push_back(
      {tensor_b.data(), static_cast<int64_t>(tensor_b.size())});

  auto serialized = partition.Serialize();

  MatrixPartition decoded;
  decoded.Deserialize(serialized->GetBuffer(), serialized->GetSize());

  EXPECT_TRUE(decoded.shm_refs_.empty());
  ASSERT_EQ(decoded.mat.size(), 2u);
  ASSERT_EQ(std::get<1>(decoded.mat[0]), static_cast<int64_t>(tensor_a.size()));
  ASSERT_EQ(std::get<1>(decoded.mat[1]), static_cast<int64_t>(tensor_b.size()));

  EXPECT_EQ(std::memcmp(std::get<0>(decoded.mat[0]), tensor_a.data(),
                        tensor_a.size()),
            0);
  EXPECT_EQ(std::memcmp(std::get<0>(decoded.mat[1]), tensor_b.data(),
                        tensor_b.size()),
            0);
}
