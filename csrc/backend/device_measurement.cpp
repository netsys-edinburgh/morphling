#include "device_measurement.h"

#include <cstdlib>
#include <cstring>

namespace {

bool ReadBoolEnv(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return false;
  if (std::strcmp(v, "0") == 0) return false;
  if (std::strcmp(v, "false") == 0) return false;
  if (std::strcmp(v, "False") == 0) return false;
  if (std::strcmp(v, "FALSE") == 0) return false;
  return true;
}

}  // namespace

DeviceMeasurementService::DeviceMeasurementService()
    : latency_enabled_(ReadBoolEnv("MORPHLING_MEASURE_LAT")),
      bandwidth_enabled_(ReadBoolEnv("MORPHLING_MEASURE_BW")),
      flops_enabled_(ReadBoolEnv("MORPHLING_MEASURE_FLOPS")) {}

bool DeviceMeasurementService::MeasureLatency(DeviceProfileData& /*profile*/) {
  if (!latency_enabled_) return false;
  // Step 3 wires the probe; until then, enabling without infrastructure is
  // a no-op that returns true so callers can sequence M1->M2->M3.
  return false;
}

bool DeviceMeasurementService::MeasureBandwidth(
    DeviceProfileData& /*profile*/) {
  if (!bandwidth_enabled_) return false;
  return false;
}

bool DeviceMeasurementService::MeasureFlops(DeviceProfileData& /*profile*/) {
  if (!flops_enabled_) return false;
  return false;
}
