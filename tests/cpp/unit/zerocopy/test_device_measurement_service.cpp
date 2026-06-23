// Issue #45 Step 2: DeviceMeasurementService skeleton.
//
// Verifies the env-var-gated control surface:
//  - All probe flags default off; service is a no-op.
//  - Setting MORPHLING_MEASURE_LAT/BW/FLOPS flips the corresponding
//    enable bit at construction time.
//  - MeasureLatency/Bandwidth/Flops are safe to call when disabled
//    (return false, leave the profile untouched).
//
// Real probe wire protocol + integration tests come in step 3.

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "backend/device_measurement.h"
#include "backend/server_base.h"

namespace {

class EnvVarScope {
 public:
  EnvVarScope(const char* name, const char* value)
      : name_(name), had_prev_(false) {
    if (const char* prev = std::getenv(name)) {
      prev_ = prev;
      had_prev_ = true;
    }
    if (value == nullptr) {
      unsetenv(name);
    } else {
      setenv(name, value, 1);
    }
  }
  ~EnvVarScope() {
    if (had_prev_) {
      setenv(name_, prev_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::string prev_;
  bool had_prev_;
};

TEST(DeviceMeasurementService, AllFlagsDefaultOff) {
  EnvVarScope clear_lat("MORPHLING_MEASURE_LAT", nullptr);
  EnvVarScope clear_bw("MORPHLING_MEASURE_BW", nullptr);
  EnvVarScope clear_flops("MORPHLING_MEASURE_FLOPS", nullptr);

  DeviceMeasurementService svc;
  EXPECT_FALSE(svc.LatencyEnabled());
  EXPECT_FALSE(svc.BandwidthEnabled());
  EXPECT_FALSE(svc.FlopsEnabled());
}

TEST(DeviceMeasurementService, EnvVarsEnableProbes) {
  EnvVarScope lat("MORPHLING_MEASURE_LAT", "1");
  EnvVarScope bw("MORPHLING_MEASURE_BW", "1");
  EnvVarScope flops("MORPHLING_MEASURE_FLOPS", "1");

  DeviceMeasurementService svc;
  EXPECT_TRUE(svc.LatencyEnabled());
  EXPECT_TRUE(svc.BandwidthEnabled());
  EXPECT_TRUE(svc.FlopsEnabled());
}

TEST(DeviceMeasurementService, NoOpWhenDisabled) {
  EnvVarScope clear_lat("MORPHLING_MEASURE_LAT", nullptr);
  EnvVarScope clear_bw("MORPHLING_MEASURE_BW", nullptr);
  EnvVarScope clear_flops("MORPHLING_MEASURE_FLOPS", nullptr);

  DeviceMeasurementService svc;
  DeviceProfileData profile{};
  profile.uuid = 42;

  EXPECT_FALSE(svc.MeasureLatency(profile));
  EXPECT_FALSE(svc.MeasureBandwidth(profile));
  EXPECT_FALSE(svc.MeasureFlops(profile));

  EXPECT_EQ(profile.measured_flops, 0ULL);
  EXPECT_EQ(profile.measured_lat_ns, 0ULL);
  EXPECT_EQ(profile.measured_ul_bw_bps, 0ULL);
  EXPECT_EQ(profile.measured_dl_bw_bps, 0ULL);
  EXPECT_FALSE(profile.measured_flops_verified);
}

}  // namespace
