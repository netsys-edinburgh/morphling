#pragma once

#include <cstdint>

#include "server_base.h"

// Coordinates server-side measurement of device-reported FLOPS, bandwidth,
// and latency (#45). All probes default off; enabled via env vars at
// construction time:
//   MORPHLING_MEASURE_LAT   - server-measured round-trip latency
//   MORPHLING_MEASURE_BW    - server-measured echo bandwidth
//   MORPHLING_MEASURE_FLOPS - seeded challenge-GEMM with verification
//
// Step 2 ships the control surface only: Measure* returns false when the
// corresponding flag is off and leaves the profile untouched. The actual
// probe wire protocol (3 new opcodes) lands in step 3.
class DeviceMeasurementService {
 public:
  DeviceMeasurementService();

  bool LatencyEnabled() const { return latency_enabled_; }
  bool BandwidthEnabled() const { return bandwidth_enabled_; }
  bool FlopsEnabled() const { return flops_enabled_; }

  bool MeasureLatency(DeviceProfileData& profile);
  bool MeasureBandwidth(DeviceProfileData& profile);
  bool MeasureFlops(DeviceProfileData& profile);

 private:
  bool latency_enabled_;
  bool bandwidth_enabled_;
  bool flops_enabled_;
};
