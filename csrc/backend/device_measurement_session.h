#pragma once

// ============================================================================
// DeviceMeasurementSession — per-device state machine (issue #55 step 3c)
//
// Drives the M1 (latency) → M2 (bandwidth) → M3 (FLOPS) probe sequence on
// the libevent worker thread that owns the device's connection. All state
// transitions happen on that thread: probe response handlers advance state
// and fire the next probe. No threads, no promises, no mutexes.
//
// The transport (wire send, timer, reference-GEMM) is injected via callbacks
// so this class can be exercised by a pure unit test without a real socket
// or a real GPU. The production driver in proxy_svr.cpp binds these to
// ConnectionUevent::SendDataZeroCopy, UeventLoop::RunAfter/CancelTimer, and
// XtGemmWorkerPool::EnqueueGemm respectively.
// ============================================================================

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "device_measurement.h"
#include "server_base.h"

namespace morphling {
namespace backend {

enum class MeasurementState {
  IDLE,
  LAT_PENDING,
  BW_PENDING,
  FLOPS_PENDING,
  DONE,
  FAILED,
};

const char* MeasurementStateName(MeasurementState s);

// Result delivered to the on_done callback. `ok` is true iff all enabled
// probes completed within their timeouts and (for M3) verification passed.
// `measured_profile` carries whatever probes did complete; unmeasured fields
// stay at 0.
struct MeasurementResult {
  bool ok = false;
  MeasurementState terminal_state = MeasurementState::IDLE;
  DeviceProfileData measured_profile;  // only the measured_* fields are set
};

// Reference-C computation (used for M3 verification). Receives matrices A
// and B in row-major float32 plus dims. Must call `done(true|false, C)`
// asynchronously (or synchronously) on the libevent thread. `C` is row-major
// m*n float32; `false` means the worker pool was unavailable.
struct ReferenceGemmRequest {
  uint32_t m, n, k;
  const float*
      a;  // m*k row-major, owned by caller (lives until done is called)
  const float* b;  // k*n row-major
};
using ReferenceGemmDoneFn =
    std::function<void(bool ok, std::shared_ptr<std::vector<float>> c)>;
using ReferenceGemmFn =
    std::function<void(const ReferenceGemmRequest&, ReferenceGemmDoneFn done)>;

// Wire send. Returns true on success (libevent took ownership). Buffer is
// kept alive by the session until the libevent cleanup callback fires, which
// it does via the SerializationBufferPtr keepalive embedded in the cleanup
// argument.
using SendFn = std::function<bool(SerializationBufferPtr buffer)>;

// Timer handle abstraction. `Arm(seconds, cb)` returns an opaque id; `Cancel`
// stops a pending callback. Production uses uevent::TimerId; tests use a
// monotonic counter and manual advance.
struct TimerOps {
  std::function<uint64_t(double seconds, std::function<void()> cb)> Arm;
  std::function<void(uint64_t id)> Cancel;
};

using OnDoneFn = std::function<void(const MeasurementResult&)>;

class DeviceMeasurementSession {
 public:
  struct Config {
    bool enable_latency = false;
    bool enable_bandwidth = false;
    bool enable_flops = false;
    uint32_t latency_payload_bytes = 64;
    uint32_t bandwidth_payload_bytes = 4u * 1024u * 1024u;
    uint32_t flops_matrix_dim = 256;  // square: m == n == k
    double probe_timeout_sec = 5.0;
    double flops_tolerance = 1e-3;
    uint64_t seed = 0;  // device_uuid; drives FLOPS challenge matrices
  };

  DeviceMeasurementSession(int64_t device_id, Config cfg, SendFn send,
                           TimerOps timer, ReferenceGemmFn reference_gemm);

  ~DeviceMeasurementSession();

  DeviceMeasurementSession(const DeviceMeasurementSession&) = delete;
  DeviceMeasurementSession& operator=(const DeviceMeasurementSession&) = delete;

  // Begin the probe sequence. If no probe is enabled, on_done fires
  // synchronously with ok=true and an empty measured_profile.
  void Start(OnDoneFn on_done);

  // Probe-response entry points. Each one MUST be called on the same thread
  // that called Start() (i.e., the libevent worker thread). Returns true if
  // the response advanced the state machine; false if it was unexpected
  // (wrong state, wrong probe_id, parse failure) and was ignored.
  bool OnProbeLatencyResponse(const void* data, size_t size);
  bool OnProbeBandwidthResponse(const void* data, size_t size);
  bool OnProbeFlopsResponse(const void* data, size_t size);

  MeasurementState state() const { return state_; }
  int64_t device_id() const { return device_id_; }

 private:
  void StartLatencyProbe();
  void StartBandwidthProbe();
  void StartFlopsProbe();
  void Finish(bool ok);
  void OnTimeout();
  void ArmTimer();
  void CancelTimer();

  using Clock = std::chrono::steady_clock;

  int64_t device_id_;
  Config cfg_;
  SendFn send_;
  TimerOps timer_;
  ReferenceGemmFn reference_gemm_;
  OnDoneFn on_done_;

  MeasurementState state_ = MeasurementState::IDLE;

  uint64_t probe_id_seq_ = 0;
  uint64_t in_flight_probe_id_ = 0;
  uint64_t timer_id_ = 0;
  bool timer_armed_ = false;

  Clock::time_point t_send_;
  uint64_t measured_lat_ns_ = 0;  // one-way (RTT/2)

  // Cached probe payload reused across probes; lives for the lifetime of the
  // session so the wire buffer references are stable until cleanup callbacks
  // fire.
  std::string latency_payload_;
  std::string bandwidth_payload_;

  // M3 challenge matrices (row-major float32). Held for verification when
  // the response comes back. Same lifetime as the session.
  std::vector<float> a_;
  std::vector<float> b_;
  std::shared_ptr<std::vector<float>> ref_c_;  // populated by reference_gemm_
  bool ref_c_ready_ = false;
  // If the device response arrives before the reference GEMM completes, stash
  // the device-computed C and verify when ref_c_ becomes ready.
  std::shared_ptr<std::vector<float>> pending_device_c_;
  double flops_elapsed_sec_ = 0.0;

  DeviceProfileData measured_;
};

using DeviceMeasurementSessionPtr = std::shared_ptr<DeviceMeasurementSession>;

}  // namespace backend
}  // namespace morphling
