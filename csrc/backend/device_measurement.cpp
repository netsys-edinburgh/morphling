#include "device_measurement.h"

#include <arpa/inet.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

#include "core/logger.h"
#include "global_api.pb.h"
#include "morphling.pb.h"
#include "serialization_buffer.h"

namespace morphling {
namespace backend {

// ============================================================================
// DeviceMeasurementService — env-var-gated configuration
// ============================================================================

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

uint32_t ReadUInt32Env(const char* name, uint32_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return fallback;
  return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
}

double ReadDoubleEnv(const char* name, double fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return fallback;
  return std::strtod(v, nullptr);
}

// Build the framed UMessage wire envelope around a serialized protobuf body.
SerializationBufferPtr FrameUMessage(
    int32_t message_type,
    std::function<void(morphling::UMessage&)> fill_extension) {
  morphling::UMessage umsg;
  auto* head = umsg.mutable_head();
  head->set_version(1);
  head->set_magic_flag(0x12340987);
  head->set_random_num(0);
  head->set_flow_no(0);
  head->set_session_no("");
  head->set_message_type(message_type);

  fill_extension(umsg);

  std::string proto_str = umsg.SerializeAsString();
  uint32_t proto_size = static_cast<uint32_t>(proto_str.size());
  uint64_t tensor_size = 0;

  uint32_t payload_size = static_cast<uint32_t>(
      sizeof(proto_size) + sizeof(tensor_size) + proto_size);
  uint64_t total_size = sizeof(payload_size) + payload_size;

  SerializationBufferPtr buffer = std::make_shared<SerializationBuffer>();
  buffer->Allocate(total_size);
  buffer->WriteUInt32(payload_size, true);
  buffer->WriteUInt32(proto_size, false);
  buffer->WriteUInt64(tensor_size);
  buffer->WriteBytes(proto_str.data(), proto_size);
  return buffer;
}

bool ExtractUMessage(const void* data, size_t size, morphling::UMessage& umsg) {
  if (data == nullptr || size < 16) return false;
  SerializationBuffer buffer(data, size, false);
  uint32_t /*payload_size*/ _payload_size = buffer.ReadUInt32(true);
  (void)_payload_size;
  uint32_t proto_size = buffer.ReadUInt32(false);
  uint64_t /*tensor_size*/ _tensor_size = buffer.ReadUInt64();
  (void)_tensor_size;
  if (proto_size == 0 || proto_size > 100 * 1024 * 1024) return false;
  return umsg.ParseFromArray(buffer.GetCurrentPtr(), proto_size);
}

}  // namespace

DeviceMeasurementService::DeviceMeasurementService()
    : latency_enabled_(ReadBoolEnv("MORPHLING_MEASURE_LAT")),
      bandwidth_enabled_(ReadBoolEnv("MORPHLING_MEASURE_BW")),
      flops_enabled_(ReadBoolEnv("MORPHLING_MEASURE_FLOPS")),
      latency_payload_bytes_(ReadUInt32Env("MORPHLING_MEASURE_LAT_BYTES", 64u)),
      bandwidth_payload_bytes_(
          ReadUInt32Env("MORPHLING_MEASURE_BW_BYTES", 4u * 1024u * 1024u)),
      flops_matrix_dim_(ReadUInt32Env("MORPHLING_MEASURE_FLOPS_DIM", 256u)),
      probe_timeout_sec_(ReadDoubleEnv("MORPHLING_MEASURE_TIMEOUT_S", 5.0)),
      flops_tolerance_(ReadDoubleEnv("MORPHLING_MEASURE_FLOPS_TOL", 1e-3)) {}

DeviceMeasurementService::DeviceMeasurementService(const Config& cfg)
    : latency_enabled_(cfg.latency_enabled),
      bandwidth_enabled_(cfg.bandwidth_enabled),
      flops_enabled_(cfg.flops_enabled),
      latency_payload_bytes_(cfg.latency_payload_bytes),
      bandwidth_payload_bytes_(cfg.bandwidth_payload_bytes),
      flops_matrix_dim_(cfg.flops_matrix_dim),
      probe_timeout_sec_(cfg.probe_timeout_sec),
      flops_tolerance_(cfg.flops_tolerance) {}

bool DeviceMeasurementService::MeasureLatency(DeviceProfileData& /*profile*/) {
  return false;
}
bool DeviceMeasurementService::MeasureBandwidth(
    DeviceProfileData& /*profile*/) {
  return false;
}
bool DeviceMeasurementService::MeasureFlops(DeviceProfileData& /*profile*/) {
  return false;
}

// ============================================================================
// ProbeMessageCodec — Serialize / Parse wire messages
// ============================================================================

SerializationBufferPtr ProbeMessageCodec::SerializeLatencyRequest(
    uint64_t probe_id, const void* payload, size_t payload_size) {
  return FrameUMessage(morphling::global_api::PROBE_LATENCY_REQUEST,
                       [&](morphling::UMessage& u) {
                         auto* m = u.mutable_body()->MutableExtension(
                             morphling::global_api::probe_latency_request);
                         m->set_probe_id(probe_id);
                         m->set_payload(static_cast<const char*>(payload),
                                        payload_size);
                       });
}

SerializationBufferPtr ProbeMessageCodec::SerializeLatencyResponse(
    uint64_t probe_id, const void* payload, size_t payload_size) {
  return FrameUMessage(morphling::global_api::PROBE_LATENCY_RESPONSE,
                       [&](morphling::UMessage& u) {
                         auto* m = u.mutable_body()->MutableExtension(
                             morphling::global_api::probe_latency_response);
                         m->set_probe_id(probe_id);
                         m->set_payload(static_cast<const char*>(payload),
                                        payload_size);
                       });
}

SerializationBufferPtr ProbeMessageCodec::SerializeBandwidthRequest(
    uint64_t probe_id, const void* payload, size_t payload_size) {
  return FrameUMessage(morphling::global_api::PROBE_BANDWIDTH_REQUEST,
                       [&](morphling::UMessage& u) {
                         auto* m = u.mutable_body()->MutableExtension(
                             morphling::global_api::probe_bandwidth_request);
                         m->set_probe_id(probe_id);
                         m->set_payload(static_cast<const char*>(payload),
                                        payload_size);
                       });
}

SerializationBufferPtr ProbeMessageCodec::SerializeBandwidthResponse(
    uint64_t probe_id, const void* payload, size_t payload_size) {
  return FrameUMessage(morphling::global_api::PROBE_BANDWIDTH_RESPONSE,
                       [&](morphling::UMessage& u) {
                         auto* m = u.mutable_body()->MutableExtension(
                             morphling::global_api::probe_bandwidth_response);
                         m->set_probe_id(probe_id);
                         m->set_payload(static_cast<const char*>(payload),
                                        payload_size);
                       });
}

SerializationBufferPtr ProbeMessageCodec::SerializeFlopsRequest(
    uint64_t probe_id, uint64_t seed, uint32_t m, uint32_t n, uint32_t k,
    const float* matrix_a, const float* matrix_b) {
  size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  return FrameUMessage(
      morphling::global_api::PROBE_FLOPS_REQUEST, [&](morphling::UMessage& u) {
        auto* msg = u.mutable_body()->MutableExtension(
            morphling::global_api::probe_flops_request);
        msg->set_probe_id(probe_id);
        msg->set_seed(seed);
        msg->set_m(m);
        msg->set_n(n);
        msg->set_k(k);
        msg->set_matrix_a(reinterpret_cast<const char*>(matrix_a), a_bytes);
        msg->set_matrix_b(reinterpret_cast<const char*>(matrix_b), b_bytes);
      });
}

SerializationBufferPtr ProbeMessageCodec::SerializeFlopsResponse(
    uint64_t probe_id, const float* matrix_c, size_t m, size_t n) {
  size_t c_bytes = m * n * sizeof(float);
  return FrameUMessage(
      morphling::global_api::PROBE_FLOPS_RESPONSE, [&](morphling::UMessage& u) {
        auto* msg = u.mutable_body()->MutableExtension(
            morphling::global_api::probe_flops_response);
        msg->set_probe_id(probe_id);
        msg->set_matrix_c(reinterpret_cast<const char*>(matrix_c), c_bytes);
      });
}

namespace {

template <typename ExtId>
bool ParseEcho(const void* data, size_t size, const ExtId& ext_id,
               ProbeMessageCodec::EchoView& out) {
  morphling::UMessage umsg;
  if (!ExtractUMessage(data, size, umsg)) return false;
  const auto& body = umsg.body();
  if (!body.HasExtension(ext_id)) return false;
  const auto& msg = body.GetExtension(ext_id);
  out.probe_id = msg.probe_id();
  out.payload = msg.payload();
  return true;
}

}  // namespace

bool ProbeMessageCodec::ParseLatencyRequest(const void* data, size_t size,
                                            EchoView& out) {
  return ParseEcho(data, size, morphling::global_api::probe_latency_request,
                   out);
}
bool ProbeMessageCodec::ParseLatencyResponse(const void* data, size_t size,
                                             EchoView& out) {
  return ParseEcho(data, size, morphling::global_api::probe_latency_response,
                   out);
}
bool ProbeMessageCodec::ParseBandwidthRequest(const void* data, size_t size,
                                              EchoView& out) {
  return ParseEcho(data, size, morphling::global_api::probe_bandwidth_request,
                   out);
}
bool ProbeMessageCodec::ParseBandwidthResponse(const void* data, size_t size,
                                               EchoView& out) {
  return ParseEcho(data, size, morphling::global_api::probe_bandwidth_response,
                   out);
}

bool ProbeMessageCodec::ParseFlopsRequest(const void* data, size_t size,
                                          FlopsRequestView& out) {
  morphling::UMessage umsg;
  if (!ExtractUMessage(data, size, umsg)) return false;
  const auto& body = umsg.body();
  if (!body.HasExtension(morphling::global_api::probe_flops_request)) {
    return false;
  }
  const auto& msg =
      body.GetExtension(morphling::global_api::probe_flops_request);
  out.probe_id = msg.probe_id();
  out.seed = msg.seed();
  out.m = msg.m();
  out.n = msg.n();
  out.k = msg.k();
  out.matrix_a = msg.matrix_a();
  out.matrix_b = msg.matrix_b();
  return true;
}

bool ProbeMessageCodec::ParseFlopsResponse(const void* data, size_t size,
                                           FlopsResponseView& out) {
  morphling::UMessage umsg;
  if (!ExtractUMessage(data, size, umsg)) return false;
  const auto& body = umsg.body();
  if (!body.HasExtension(morphling::global_api::probe_flops_response)) {
    return false;
  }
  const auto& msg =
      body.GetExtension(morphling::global_api::probe_flops_response);
  out.probe_id = msg.probe_id();
  out.matrix_c = msg.matrix_c();
  return true;
}

}  // namespace backend
}  // namespace morphling
