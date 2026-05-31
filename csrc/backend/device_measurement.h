#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "server_base.h"

namespace morphling {
namespace backend {

class DeviceMeasurementService {
 public:
  DeviceMeasurementService();

  bool LatencyEnabled() const { return latency_enabled_; }
  bool BandwidthEnabled() const { return bandwidth_enabled_; }
  bool FlopsEnabled() const { return flops_enabled_; }
  bool AnyEnabled() const {
    return latency_enabled_ || bandwidth_enabled_ || flops_enabled_;
  }

  uint32_t LatencyPayloadBytes() const { return latency_payload_bytes_; }
  uint32_t BandwidthPayloadBytes() const { return bandwidth_payload_bytes_; }
  uint32_t FlopsMatrixDim() const { return flops_matrix_dim_; }
  double ProbeTimeoutSec() const { return probe_timeout_sec_; }
  double FlopsTolerance() const { return flops_tolerance_; }

  bool MeasureLatency(DeviceProfileData& profile);
  bool MeasureBandwidth(DeviceProfileData& profile);
  bool MeasureFlops(DeviceProfileData& profile);

 private:
  bool latency_enabled_;
  bool bandwidth_enabled_;
  bool flops_enabled_;
  uint32_t latency_payload_bytes_;
  uint32_t bandwidth_payload_bytes_;
  uint32_t flops_matrix_dim_;
  double probe_timeout_sec_;
  double flops_tolerance_;
};

// Build payloads (request bodies) for the three probes. Reused by tests.
class ProbeMessageCodec {
 public:
  static SerializationBufferPtr SerializeLatencyRequest(uint64_t probe_id,
                                                        const void* payload,
                                                        size_t payload_size);
  static SerializationBufferPtr SerializeLatencyResponse(uint64_t probe_id,
                                                         const void* payload,
                                                         size_t payload_size);
  static SerializationBufferPtr SerializeBandwidthRequest(uint64_t probe_id,
                                                          const void* payload,
                                                          size_t payload_size);
  static SerializationBufferPtr SerializeBandwidthResponse(uint64_t probe_id,
                                                           const void* payload,
                                                           size_t payload_size);
  static SerializationBufferPtr SerializeFlopsRequest(uint64_t probe_id,
                                                      uint64_t seed, uint32_t m,
                                                      uint32_t n, uint32_t k,
                                                      const float* matrix_a,
                                                      const float* matrix_b);
  static SerializationBufferPtr SerializeFlopsResponse(uint64_t probe_id,
                                                       const float* matrix_c,
                                                       size_t m, size_t n);

  // Parse helpers: caller supplies pointers to framed UMessage wire bytes
  // (starting with payload_size). The returned views own their byte buffers
  // (moved out of the parsed protobuf) so they remain valid after the
  // input bytes are drained.
  struct EchoView {
    uint64_t probe_id = 0;
    std::string payload;
  };
  struct FlopsRequestView {
    uint64_t probe_id = 0;
    uint64_t seed = 0;
    uint32_t m = 0, n = 0, k = 0;
    std::string matrix_a;
    std::string matrix_b;
  };
  struct FlopsResponseView {
    uint64_t probe_id = 0;
    std::string matrix_c;
  };

  static bool ParseLatencyRequest(const void* data, size_t size, EchoView& out);
  static bool ParseLatencyResponse(const void* data, size_t size,
                                   EchoView& out);
  static bool ParseBandwidthRequest(const void* data, size_t size,
                                    EchoView& out);
  static bool ParseBandwidthResponse(const void* data, size_t size,
                                     EchoView& out);
  static bool ParseFlopsRequest(const void* data, size_t size,
                                FlopsRequestView& out);
  static bool ParseFlopsResponse(const void* data, size_t size,
                                 FlopsResponseView& out);
};

}  // namespace backend
}  // namespace morphling

// ============================================================================
// Compatibility alias for step-2 test (tests/cpp/unit/zerocopy/
// test_device_measurement_service.cpp). The test still uses the unqualified
// name `DeviceMeasurementService`; keep it available without forcing a
// migration of that test alongside this header.
// ============================================================================
using DeviceMeasurementService = morphling::backend::DeviceMeasurementService;
