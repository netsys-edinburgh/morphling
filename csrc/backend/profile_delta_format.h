#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "server_base.h"

namespace morphling {
namespace backend {

inline std::string FormatProfileDeltaRow(uint64_t now_us, int64_t device_id,
                                         const DeviceProfileData& p) {
  auto ratio = [](uint64_t reported, uint64_t measured) -> double {
    return reported > 0 ? static_cast<double>(measured) / reported : -1.0;
  };

  char buf[512];
  int len = snprintf(
      buf, sizeof(buf),
      "PROFILE_DELTA,%lu,%ld,%lu,%lu,%lu,%d,%.6f,%lu,%lu,%.6f,%lu,%lu,%.6f,%lu,"
      "%lu,%lu",
      now_us, device_id, p.uuid, p.flops, p.measured_flops,
      p.measured_flops_verified ? 1 : 0, ratio(p.flops, p.measured_flops),
      p.ul_bw, p.measured_ul_bw_bps, ratio(p.ul_bw, p.measured_ul_bw_bps),
      p.dl_bw, p.measured_dl_bw_bps, ratio(p.dl_bw, p.measured_dl_bw_bps),
      p.ul_lat, p.dl_lat, p.measured_lat_ns);
  if (len > 0 && len < static_cast<int>(sizeof(buf))) {
    return std::string(buf, len);
  }
  return std::string();
}

}  // namespace backend
}  // namespace morphling
