#include "device_measurement_session.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "core/logger.h"

namespace morphling {
namespace backend {

const char* MeasurementStateName(MeasurementState s) {
  switch (s) {
    case MeasurementState::IDLE:
      return "IDLE";
    case MeasurementState::LAT_PENDING:
      return "LAT_PENDING";
    case MeasurementState::BW_PENDING:
      return "BW_PENDING";
    case MeasurementState::FLOPS_PENDING:
      return "FLOPS_PENDING";
    case MeasurementState::DONE:
      return "DONE";
    case MeasurementState::FAILED:
      return "FAILED";
  }
  return "?";
}

namespace {

// Deterministically fill an echo payload with the device seed + probe id so
// the server can sanity-check the device echo.
void FillEchoPayload(std::string& buf, uint64_t seed, uint64_t probe_id) {
  std::mt19937_64 rng(seed ^ (probe_id * 0x9E3779B97F4A7C15ULL));
  size_t n = buf.size();
  size_t i = 0;
  while (i + 8 <= n) {
    uint64_t v = rng();
    std::memcpy(&buf[i], &v, 8);
    i += 8;
  }
  while (i < n) {
    uint8_t v = static_cast<uint8_t>(rng() & 0xFFu);
    buf[i++] = static_cast<char>(v);
  }
}

// Deterministically fill a float matrix with values in [-1, 1].
void FillMatrix(std::vector<float>& buf, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < buf.size(); ++i) buf[i] = dist(rng);
}

}  // namespace

DeviceMeasurementSession::DeviceMeasurementSession(int64_t device_id,
                                                   Config cfg, SendFn send,
                                                   TimerOps timer,
                                                   ReferenceGemmFn reference)
    : device_id_(device_id),
      cfg_(cfg),
      send_(std::move(send)),
      timer_(std::move(timer)),
      reference_gemm_(std::move(reference)) {}

DeviceMeasurementSession::~DeviceMeasurementSession() { CancelTimer(); }

void DeviceMeasurementSession::Start(OnDoneFn on_done) {
  on_done_ = std::move(on_done);
  if (state_ != MeasurementState::IDLE) {
    LOG_ERROR << "DeviceMeasurementSession::Start called in state "
              << MeasurementStateName(state_);
    Finish(false);
    return;
  }

  if (!cfg_.enable_latency && !cfg_.enable_bandwidth && !cfg_.enable_flops) {
    state_ = MeasurementState::DONE;
    MeasurementResult r;
    r.ok = true;
    r.terminal_state = state_;
    r.measured_profile = measured_;
    if (on_done_) on_done_(r);
    return;
  }

  // Probes are gated independently but ordered: the M2 RTT subtraction
  // assumes we know measured_lat_ns_, which is only valid if M1 ran. So if
  // M2 is enabled without M1, fall back to assuming zero latency (BW will
  // still be a useful upper bound). Same for M3 (no latency correction
  // needed beyond its 5 s budget).
  if (cfg_.enable_latency) {
    StartLatencyProbe();
  } else if (cfg_.enable_bandwidth) {
    StartBandwidthProbe();
  } else if (cfg_.enable_flops) {
    StartFlopsProbe();
  }
}

void DeviceMeasurementSession::StartLatencyProbe() {
  state_ = MeasurementState::LAT_PENDING;
  ++probe_id_seq_;
  in_flight_probe_id_ = probe_id_seq_;

  latency_payload_.assign(cfg_.latency_payload_bytes, '\0');
  FillEchoPayload(latency_payload_, cfg_.seed, in_flight_probe_id_);

  auto buf = ProbeMessageCodec::SerializeLatencyRequest(
      in_flight_probe_id_, latency_payload_.data(), latency_payload_.size());

  t_send_ = Clock::now();
  ArmTimer();
  if (!send_(buf)) {
    LOG_ERROR << "send PROBE_LATENCY_REQUEST failed for device " << device_id_;
    CancelTimer();
    Finish(false);
  }
}

void DeviceMeasurementSession::StartBandwidthProbe() {
  state_ = MeasurementState::BW_PENDING;
  ++probe_id_seq_;
  in_flight_probe_id_ = probe_id_seq_;

  bandwidth_payload_.assign(cfg_.bandwidth_payload_bytes, '\0');
  FillEchoPayload(bandwidth_payload_, cfg_.seed, in_flight_probe_id_);

  auto buf = ProbeMessageCodec::SerializeBandwidthRequest(
      in_flight_probe_id_, bandwidth_payload_.data(),
      bandwidth_payload_.size());

  t_send_ = Clock::now();
  ArmTimer();
  if (!send_(buf)) {
    LOG_ERROR << "send PROBE_BANDWIDTH_REQUEST failed for device "
              << device_id_;
    CancelTimer();
    Finish(false);
  }
}

void DeviceMeasurementSession::StartFlopsProbe() {
  state_ = MeasurementState::FLOPS_PENDING;
  ++probe_id_seq_;
  in_flight_probe_id_ = probe_id_seq_;

  const uint32_t m = cfg_.flops_matrix_dim;
  const uint32_t n = cfg_.flops_matrix_dim;
  const uint32_t k = cfg_.flops_matrix_dim;
  a_.assign(static_cast<size_t>(m) * k, 0.0f);
  b_.assign(static_cast<size_t>(k) * n, 0.0f);
  FillMatrix(a_, cfg_.seed ^ 0xA11AA11AULL);
  FillMatrix(b_, cfg_.seed ^ 0xB22BB22BULL);

  // Kick the reference GEMM in parallel with the device challenge so the
  // total verification window doesn't double. Capture self via raw pointer
  // — the session outlives in-flight callbacks because proxy_svr owns it
  // and only deletes it after Finish() or on UnregisterDevice (which also
  // tears down the connection).
  ref_c_ready_ = false;
  ref_c_.reset();
  pending_device_c_.reset();
  flops_elapsed_sec_ = 0.0;

  ReferenceGemmRequest req{m, n, k, a_.data(), b_.data()};
  auto* self = this;
  reference_gemm_(req, [self](bool ok, std::shared_ptr<std::vector<float>> c) {
    if (!ok || !c) {
      LOG_ERROR << "reference GEMM failed for device " << self->device_id_;
      self->ref_c_ready_ = false;
      if (self->state_ == MeasurementState::FLOPS_PENDING) {
        self->CancelTimer();
        self->Finish(false);
      }
      return;
    }
    self->ref_c_ = c;
    self->ref_c_ready_ = true;
    // If the device response already arrived, finish verification now.
    if (self->pending_device_c_) {
      auto dev_c = self->pending_device_c_;
      self->pending_device_c_.reset();
      // Reuse the verification path: pretend we just got the response.
      // We bypass parsing — verify in-line.
      const auto& cref = *self->ref_c_;
      const auto& cdev = *dev_c;
      const double atol = self->cfg_.flops_tolerance;
      const double rtol = self->cfg_.flops_tolerance;
      bool verified = (cref.size() == cdev.size());
      if (verified) {
        for (size_t i = 0; i < cref.size(); ++i) {
          double diff = std::fabs(static_cast<double>(cref[i]) - cdev[i]);
          double tol = atol + rtol * std::fabs(static_cast<double>(cref[i]));
          if (diff > tol) {
            verified = false;
            break;
          }
        }
      }
      const uint32_t mm = self->cfg_.flops_matrix_dim;
      double flops =
          2.0 * mm * mm * mm / std::max(self->flops_elapsed_sec_, 1e-9);
      self->measured_.measured_flops = static_cast<uint64_t>(flops);
      self->measured_.measured_flops_verified = verified;
      self->Finish(verified);
    }
  });

  auto buf = ProbeMessageCodec::SerializeFlopsRequest(
      in_flight_probe_id_, cfg_.seed, m, n, k, a_.data(), b_.data());

  t_send_ = Clock::now();
  ArmTimer();
  if (!send_(buf)) {
    LOG_ERROR << "send PROBE_FLOPS_REQUEST failed for device " << device_id_;
    CancelTimer();
    Finish(false);
  }
}

bool DeviceMeasurementSession::OnProbeLatencyResponse(const void* data,
                                                      size_t size) {
  if (state_ != MeasurementState::LAT_PENDING) {
    LOG_DEBUG << "ignoring PROBE_LATENCY_RESPONSE in state "
              << MeasurementStateName(state_);
    return false;
  }
  ProbeMessageCodec::EchoView view;
  if (!ProbeMessageCodec::ParseLatencyResponse(data, size, view) ||
      view.probe_id != in_flight_probe_id_) {
    LOG_DEBUG << "stale/invalid PROBE_LATENCY_RESPONSE for device "
              << device_id_;
    return false;
  }
  CancelTimer();
  auto now = Clock::now();
  uint64_t rtt_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_send_)
          .count();
  measured_lat_ns_ = rtt_ns / 2;
  measured_.measured_lat_ns = measured_lat_ns_;

  if (cfg_.enable_bandwidth) {
    StartBandwidthProbe();
  } else if (cfg_.enable_flops) {
    StartFlopsProbe();
  } else {
    state_ = MeasurementState::DONE;
    Finish(true);
  }
  return true;
}

bool DeviceMeasurementSession::OnProbeBandwidthResponse(const void* data,
                                                        size_t size) {
  if (state_ != MeasurementState::BW_PENDING) {
    LOG_DEBUG << "ignoring PROBE_BANDWIDTH_RESPONSE in state "
              << MeasurementStateName(state_);
    return false;
  }
  ProbeMessageCodec::EchoView view;
  if (!ProbeMessageCodec::ParseBandwidthResponse(data, size, view) ||
      view.probe_id != in_flight_probe_id_) {
    LOG_DEBUG << "stale/invalid PROBE_BANDWIDTH_RESPONSE for device "
              << device_id_;
    return false;
  }
  CancelTimer();
  auto now = Clock::now();
  uint64_t rtt_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_send_)
          .count();
  // Subtract one full RTT of latency (request flight + response flight).
  uint64_t lat_ns = 2 * measured_lat_ns_;
  uint64_t transfer_ns = (rtt_ns > lat_ns) ? (rtt_ns - lat_ns) : rtt_ns;
  // Bytes-on-wire (one direction): payload size. Bandwidth measured here is
  // the round-trip echo throughput, i.e. min(uplink, downlink). We report
  // it as both ul_bw and dl_bw with that limitation noted in the plan.
  double transfer_sec = std::max(1e-9, static_cast<double>(transfer_ns) / 1e9);
  double bytes = static_cast<double>(cfg_.bandwidth_payload_bytes);
  double bps = bytes / transfer_sec;
  measured_.measured_dl_bw_bps = static_cast<uint64_t>(bps);
  measured_.measured_ul_bw_bps = static_cast<uint64_t>(bps);

  if (cfg_.enable_flops) {
    StartFlopsProbe();
  } else {
    state_ = MeasurementState::DONE;
    Finish(true);
  }
  return true;
}

bool DeviceMeasurementSession::OnProbeFlopsResponse(const void* data,
                                                    size_t size) {
  if (state_ != MeasurementState::FLOPS_PENDING) {
    LOG_DEBUG << "ignoring PROBE_FLOPS_RESPONSE in state "
              << MeasurementStateName(state_);
    return false;
  }
  ProbeMessageCodec::FlopsResponseView view;
  if (!ProbeMessageCodec::ParseFlopsResponse(data, size, view) ||
      view.probe_id != in_flight_probe_id_) {
    LOG_DEBUG << "stale/invalid PROBE_FLOPS_RESPONSE for device " << device_id_;
    return false;
  }
  CancelTimer();
  auto now = Clock::now();
  uint64_t rtt_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_send_)
          .count();
  // Subtract the round-trip latency we measured in M1 to recover the
  // device-side compute time.
  uint64_t lat_ns = 2 * measured_lat_ns_;
  uint64_t compute_ns = (rtt_ns > lat_ns) ? (rtt_ns - lat_ns) : rtt_ns;
  flops_elapsed_sec_ = std::max(1e-9, static_cast<double>(compute_ns) / 1e9);

  const uint32_t m = cfg_.flops_matrix_dim;
  const uint32_t n = cfg_.flops_matrix_dim;
  size_t expected_bytes = static_cast<size_t>(m) * n * sizeof(float);
  if (view.matrix_c.size() != expected_bytes) {
    LOG_ERROR << "PROBE_FLOPS_RESPONSE matrix_c size mismatch: got "
              << view.matrix_c.size() << " expected " << expected_bytes;
    Finish(false);
    return true;  // we did consume a response
  }
  auto dev_c = std::make_shared<std::vector<float>>(static_cast<size_t>(m) * n);
  std::memcpy(dev_c->data(), view.matrix_c.data(), expected_bytes);

  if (!ref_c_ready_) {
    // Reference GEMM still in flight; stash and verify when it's ready.
    pending_device_c_ = dev_c;
    return true;
  }

  const auto& cref = *ref_c_;
  const auto& cdev = *dev_c;
  const double atol = cfg_.flops_tolerance;
  const double rtol = cfg_.flops_tolerance;
  bool verified = (cref.size() == cdev.size());
  if (verified) {
    for (size_t i = 0; i < cref.size(); ++i) {
      double diff = std::fabs(static_cast<double>(cref[i]) - cdev[i]);
      double tol = atol + rtol * std::fabs(static_cast<double>(cref[i]));
      if (diff > tol) {
        verified = false;
        break;
      }
    }
  }
  double flops_val = 2.0 * static_cast<double>(m) * n * cfg_.flops_matrix_dim /
                     flops_elapsed_sec_;
  measured_.measured_flops = static_cast<uint64_t>(flops_val);
  measured_.measured_flops_verified = verified;
  Finish(verified);
  return true;
}

void DeviceMeasurementSession::Finish(bool ok) {
  if (state_ == MeasurementState::DONE || state_ == MeasurementState::FAILED) {
    return;  // already terminal
  }
  state_ = ok ? MeasurementState::DONE : MeasurementState::FAILED;
  CancelTimer();
  if (on_done_) {
    MeasurementResult r;
    r.ok = ok;
    r.terminal_state = state_;
    r.measured_profile = measured_;
    auto cb = std::move(on_done_);
    on_done_ = nullptr;
    cb(r);
  }
}

void DeviceMeasurementSession::OnTimeout() {
  LOG_WARN << "DeviceMeasurementSession device=" << device_id_
           << " state=" << MeasurementStateName(state_) << " timed out after "
           << cfg_.probe_timeout_sec << "s";
  timer_armed_ = false;
  Finish(false);
}

void DeviceMeasurementSession::ArmTimer() {
  CancelTimer();
  if (!timer_.Arm) return;  // tests may run without a timer
  auto* self = this;
  timer_id_ =
      timer_.Arm(cfg_.probe_timeout_sec, [self]() { self->OnTimeout(); });
  timer_armed_ = true;
}

void DeviceMeasurementSession::CancelTimer() {
  if (timer_armed_ && timer_.Cancel) {
    timer_.Cancel(timer_id_);
  }
  timer_armed_ = false;
  timer_id_ = 0;
}

}  // namespace backend
}  // namespace morphling
