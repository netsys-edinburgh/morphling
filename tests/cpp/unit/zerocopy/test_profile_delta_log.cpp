// Issue #60: PROFILE_DELTA row formatting (policy-neutral observability).
//
// Locks in the column order, the measured/reported ratio computation, the
// -1 sentinel when a reported field is zero, and the raw us/ns latency
// columns (no latency ratio is emitted because reported is us, measured ns).

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "backend/profile_delta_format.h"
#include "backend/server_base.h"

namespace {

using morphling::backend::FormatProfileDeltaRow;

std::vector<std::string> Split(const std::string& row) {
  std::vector<std::string> out;
  std::stringstream ss(row);
  std::string field;
  while (std::getline(ss, field, ',')) {
    out.push_back(field);
  }
  return out;
}

DeviceProfileData MakeProfile() {
  DeviceProfileData p{};
  p.uuid = 0x1122334455667788ULL;
  p.flops = 1000;
  p.memory = 16ULL * 1024 * 1024 * 1024;
  p.ul_bw = 2000;
  p.dl_bw = 4000;
  p.ul_lat = 1000;
  p.dl_lat = 1000;
  p.measured_flops = 500;
  p.measured_flops_verified = true;
  p.measured_lat_ns = 250000;
  p.measured_ul_bw_bps = 1000;
  p.measured_dl_bw_bps = 6000;
  return p;
}

TEST(ProfileDeltaRow, SchemaAndRatios) {
  DeviceProfileData p = MakeProfile();
  std::string row = FormatProfileDeltaRow(123456, 7, p);

  auto f = Split(row);
  ASSERT_EQ(f.size(), 17u);

  EXPECT_EQ(f[0], "PROFILE_DELTA");
  EXPECT_EQ(f[1], "123456");
  EXPECT_EQ(f[2], "7");
  EXPECT_EQ(f[3], std::to_string(p.uuid));

  EXPECT_EQ(f[4], "1000");
  EXPECT_EQ(f[5], "500");
  EXPECT_EQ(f[6], "1");
  EXPECT_DOUBLE_EQ(std::stod(f[7]), 0.5);

  EXPECT_EQ(f[8], "2000");
  EXPECT_EQ(f[9], "1000");
  EXPECT_DOUBLE_EQ(std::stod(f[10]), 0.5);

  EXPECT_EQ(f[11], "4000");
  EXPECT_EQ(f[12], "6000");
  EXPECT_DOUBLE_EQ(std::stod(f[13]), 1.5);

  EXPECT_EQ(f[14], "1000");
  EXPECT_EQ(f[15], "1000");
  EXPECT_EQ(f[16], "250000");
}

TEST(ProfileDeltaRow, SentinelWhenReportedZero) {
  DeviceProfileData p{};
  p.uuid = 42;
  p.flops = 0;
  p.measured_flops = 900;
  p.ul_bw = 0;
  p.measured_ul_bw_bps = 800;
  p.dl_bw = 0;
  p.measured_dl_bw_bps = 700;

  auto f = Split(FormatProfileDeltaRow(1, 2, p));
  ASSERT_EQ(f.size(), 17u);

  EXPECT_DOUBLE_EQ(std::stod(f[7]), -1.0);
  EXPECT_DOUBLE_EQ(std::stod(f[10]), -1.0);
  EXPECT_DOUBLE_EQ(std::stod(f[13]), -1.0);
}

TEST(ProfileDeltaRow, EmptyMeasuredIsZeroRatio) {
  DeviceProfileData p{};
  p.flops = 1000;
  p.ul_bw = 1000;
  p.dl_bw = 1000;

  auto f = Split(FormatProfileDeltaRow(1, 2, p));
  ASSERT_EQ(f.size(), 17u);

  EXPECT_DOUBLE_EQ(std::stod(f[7]), 0.0);
  EXPECT_DOUBLE_EQ(std::stod(f[10]), 0.0);
  EXPECT_DOUBLE_EQ(std::stod(f[13]), 0.0);
  EXPECT_EQ(f[6], "0");
}

}  // namespace
