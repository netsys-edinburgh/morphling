// Issue #45 Step 1: DeviceProfileData carries server-measured fields.
//
// Locks in two contracts:
//  - Legacy payloads (without measured_* fields) deserialize cleanly,
//    leaving the new optional fields at their default value 0.
//  - Round-tripping a struct with measured_* set preserves all 12 fields.

#include <gtest/gtest.h>

#include <memory>

#include "backend/serialization_buffer.h"
#include "backend/server_base.h"

namespace {

DeviceProfileData LegacyProfile() {
  DeviceProfileData p{};
  p.uuid = 0x1122334455667788ULL;
  p.flops = 100000000000ULL;
  p.memory = 16ULL * 1024 * 1024 * 1024;
  p.ul_bw = 10ULL * 1024 * 1024 * 1024;
  p.dl_bw = 10ULL * 1024 * 1024 * 1024;
  p.ul_lat = 1000;
  p.dl_lat = 1000;
  return p;
}

TEST(DeviceProfileDataRoundtrip, LegacyFieldsOnly) {
  DeviceProfileData src = LegacyProfile();
  auto buf = src.Serialize();
  ASSERT_TRUE(buf != nullptr);

  DeviceProfileData dst{};
  dst.Deserialize(buf->GetBuffer(), buf->GetSize());

  EXPECT_EQ(dst.uuid, src.uuid);
  EXPECT_EQ(dst.flops, src.flops);
  EXPECT_EQ(dst.memory, src.memory);
  EXPECT_EQ(dst.ul_bw, src.ul_bw);
  EXPECT_EQ(dst.dl_bw, src.dl_bw);
  EXPECT_EQ(dst.ul_lat, src.ul_lat);
  EXPECT_EQ(dst.dl_lat, src.dl_lat);

  // New optional fields default to 0 when sender omits them.
  EXPECT_EQ(dst.measured_flops, 0ULL);
  EXPECT_EQ(dst.measured_lat_ns, 0ULL);
  EXPECT_EQ(dst.measured_ul_bw_bps, 0ULL);
  EXPECT_EQ(dst.measured_dl_bw_bps, 0ULL);
  EXPECT_FALSE(dst.measured_flops_verified);
}

TEST(DeviceProfileDataRoundtrip, LegacyPlusMeasured) {
  DeviceProfileData src = LegacyProfile();
  src.measured_flops = 95000000000ULL;
  src.measured_flops_verified = true;
  src.measured_lat_ns = 250000ULL;
  src.measured_ul_bw_bps = 9ULL * 1024 * 1024 * 1024;
  src.measured_dl_bw_bps = 8ULL * 1024 * 1024 * 1024;

  auto buf = src.Serialize();
  ASSERT_TRUE(buf != nullptr);

  DeviceProfileData dst{};
  dst.Deserialize(buf->GetBuffer(), buf->GetSize());

  EXPECT_EQ(dst.uuid, src.uuid);
  EXPECT_EQ(dst.measured_flops, src.measured_flops);
  EXPECT_TRUE(dst.measured_flops_verified);
  EXPECT_EQ(dst.measured_lat_ns, src.measured_lat_ns);
  EXPECT_EQ(dst.measured_ul_bw_bps, src.measured_ul_bw_bps);
  EXPECT_EQ(dst.measured_dl_bw_bps, src.measured_dl_bw_bps);
}

TEST(DeviceProfileDataRoundtrip, DebugStringMentionsMeasuredWhenSet) {
  DeviceProfileData src = LegacyProfile();
  src.measured_flops = 95000000000ULL;
  src.measured_flops_verified = true;

  const auto s = src.DebugString();
  EXPECT_NE(s.find("measured_flops"), std::string::npos);
  EXPECT_NE(s.find("95000000000"), std::string::npos);
}

}  // namespace
