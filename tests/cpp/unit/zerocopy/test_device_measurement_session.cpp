// Issue #55 Step 3c: DeviceMeasurementSession state machine.
//
// Exercises the per-device M1->M2->M3 probe state machine in isolation
// (no socket, no GPU). The driver injects:
//   - send_fn:        records outgoing wire bytes
//   - timer_ops:      manual-advance timer (no libevent)
//   - reference_gemm: synchronous CPU reference C = A * B
//
// Covered cases:
//   - Empty config (no probes enabled): Start finishes immediately, ok=true
//   - Happy path M1+M2+M3: walks through every state, measured_* populated
//   - FLOPS verification mismatch: device echoes wrong bytes -> verified=false
//   - Latency timeout: no response delivered, fires OnTimeout -> FAILED
//   - Stale probe_id is ignored

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "backend/device_measurement.h"
#include "backend/device_measurement_session.h"
#include "backend/serialization_buffer.h"
#include "backend/server_base.h"

namespace mb = morphling::backend;

namespace {

// Recorded outgoing wire frame: a copy of the bytes the session asked us
// to send. Stored as std::string so the test can re-parse it and produce
// the matching response frame.
struct CapturedSend {
  std::vector<uint8_t> bytes;
};

// Manual timer queue with monotonically-increasing ids and an explicit
// Advance() entry point that fires every still-armed callback.
class FakeTimer {
 public:
  uint64_t Arm(double /*seconds*/, std::function<void()> cb) {
    uint64_t id = ++next_id_;
    callbacks_[id] = std::move(cb);
    return id;
  }
  void Cancel(uint64_t id) { callbacks_.erase(id); }
  void FireAll() {
    auto pending = std::move(callbacks_);
    callbacks_.clear();
    for (auto& kv : pending) kv.second();
  }
  size_t pending() const { return callbacks_.size(); }

 private:
  uint64_t next_id_ = 0;
  std::map<uint64_t, std::function<void()>> callbacks_;
};

// Synchronous CPU GEMM, row-major float32, C = A * B with C of size m*n.
void CpuGemm(uint32_t m, uint32_t n, uint32_t k, const float* a, const float* b,
             float* c) {
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (uint32_t p = 0; p < k; ++p) {
        acc += static_cast<double>(a[i * k + p]) * b[p * n + j];
      }
      c[i * n + j] = static_cast<float>(acc);
    }
  }
}

// Helper: rebuild the framed UMessage envelope shape the codec emits, but
// instead pass it back through the codec's parse path on the session. The
// session expects the "framed" bytes (starting with payload_size). The
// serializer already produces that layout, so we forward verbatim.

}  // namespace

TEST(DeviceMeasurementSession, EmptyConfigDoneImmediately) {
  std::vector<CapturedSend> sent;
  FakeTimer timer;
  mb::DeviceMeasurementSession::Config cfg{};  // all probes off

  mb::DeviceMeasurementSession s(
      /*device_id=*/1, cfg,
      [&](SerializationBufferPtr buf) {
        sent.push_back(CapturedSend{std::vector<uint8_t>(
            static_cast<const uint8_t*>(buf->GetBuffer()),
            static_cast<const uint8_t*>(buf->GetBuffer()) + buf->GetSize())});
        return true;
      },
      {[&](double sec, std::function<void()> cb) {
         return timer.Arm(sec, std::move(cb));
       },
       [&](uint64_t id) { timer.Cancel(id); }},
      /*reference_gemm=*/nullptr);

  bool fired = false;
  mb::MeasurementResult result;
  s.Start([&](const mb::MeasurementResult& r) {
    fired = true;
    result = r;
  });

  EXPECT_TRUE(fired);
  EXPECT_TRUE(result.ok);
  EXPECT_EQ(result.terminal_state, mb::MeasurementState::DONE);
  EXPECT_EQ(sent.size(), 0u);
  EXPECT_EQ(s.state(), mb::MeasurementState::DONE);
}

TEST(DeviceMeasurementSession, HappyPathM1M2M3) {
  std::vector<CapturedSend> sent;
  FakeTimer timer;

  mb::DeviceMeasurementSession::Config cfg;
  cfg.enable_latency = true;
  cfg.enable_bandwidth = true;
  cfg.enable_flops = true;
  cfg.latency_payload_bytes = 64;
  cfg.bandwidth_payload_bytes = 4096;  // small, this is a CPU test
  cfg.flops_matrix_dim = 16;
  cfg.probe_timeout_sec = 10.0;
  cfg.flops_tolerance = 1e-3;
  cfg.seed = 0xC0FFEEULL;

  // Reference GEMM: compute C = A * B on CPU and hand the result back.
  auto ref_gemm = [](const mb::ReferenceGemmRequest& req,
                     mb::ReferenceGemmDoneFn done) {
    auto c = std::make_shared<std::vector<float>>(static_cast<size_t>(req.m) *
                                                  req.n);
    CpuGemm(req.m, req.n, req.k, req.a, req.b, c->data());
    done(true, c);
  };

  mb::DeviceMeasurementSession s(
      /*device_id=*/42, cfg,
      [&](SerializationBufferPtr buf) {
        sent.push_back(CapturedSend{std::vector<uint8_t>(
            static_cast<const uint8_t*>(buf->GetBuffer()),
            static_cast<const uint8_t*>(buf->GetBuffer()) + buf->GetSize())});
        return true;
      },
      {[&](double sec, std::function<void()> cb) {
         return timer.Arm(sec, std::move(cb));
       },
       [&](uint64_t id) { timer.Cancel(id); }},
      ref_gemm);

  bool fired = false;
  mb::MeasurementResult result;
  s.Start([&](const mb::MeasurementResult& r) {
    fired = true;
    result = r;
  });

  ASSERT_EQ(s.state(), mb::MeasurementState::LAT_PENDING);
  ASSERT_EQ(sent.size(), 1u);

  // Parse the LAT request, echo it back as a LAT response.
  mb::ProbeMessageCodec::EchoView lat_view;
  ASSERT_TRUE(mb::ProbeMessageCodec::ParseLatencyRequest(
      sent[0].bytes.data(), sent[0].bytes.size(), lat_view));
  // Throttle 1 ms so RTT is measurable.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  auto lat_rsp = mb::ProbeMessageCodec::SerializeLatencyResponse(
      lat_view.probe_id, lat_view.payload.data(), lat_view.payload.size());
  ASSERT_TRUE(
      s.OnProbeLatencyResponse(lat_rsp->GetBuffer(), lat_rsp->GetSize()));

  ASSERT_EQ(s.state(), mb::MeasurementState::BW_PENDING);
  ASSERT_EQ(sent.size(), 2u);

  // Echo BW response.
  mb::ProbeMessageCodec::EchoView bw_view;
  ASSERT_TRUE(mb::ProbeMessageCodec::ParseBandwidthRequest(
      sent[1].bytes.data(), sent[1].bytes.size(), bw_view));
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto bw_rsp = mb::ProbeMessageCodec::SerializeBandwidthResponse(
      bw_view.probe_id, bw_view.payload.data(), bw_view.payload.size());
  ASSERT_TRUE(
      s.OnProbeBandwidthResponse(bw_rsp->GetBuffer(), bw_rsp->GetSize()));

  ASSERT_EQ(s.state(), mb::MeasurementState::FLOPS_PENDING);
  ASSERT_EQ(sent.size(), 3u);

  // Parse the FLOPS request, compute device-side C with the same CPU GEMM,
  // build the response frame.
  mb::ProbeMessageCodec::FlopsRequestView flops_view;
  ASSERT_TRUE(mb::ProbeMessageCodec::ParseFlopsRequest(
      sent[2].bytes.data(), sent[2].bytes.size(), flops_view));
  const uint32_t m = flops_view.m, n = flops_view.n, k = flops_view.k;
  std::vector<float> dev_c(static_cast<size_t>(m) * n);
  CpuGemm(m, n, k, reinterpret_cast<const float*>(flops_view.matrix_a.data()),
          reinterpret_cast<const float*>(flops_view.matrix_b.data()),
          dev_c.data());
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto flops_rsp = mb::ProbeMessageCodec::SerializeFlopsResponse(
      flops_view.probe_id, dev_c.data(), m, n);
  ASSERT_TRUE(
      s.OnProbeFlopsResponse(flops_rsp->GetBuffer(), flops_rsp->GetSize()));

  EXPECT_TRUE(fired);
  EXPECT_TRUE(result.ok);
  EXPECT_EQ(result.terminal_state, mb::MeasurementState::DONE);
  EXPECT_GT(result.measured_profile.measured_lat_ns, 0ULL);
  EXPECT_GT(result.measured_profile.measured_dl_bw_bps, 0ULL);
  EXPECT_GT(result.measured_profile.measured_ul_bw_bps, 0ULL);
  EXPECT_GT(result.measured_profile.measured_flops, 0ULL);
  EXPECT_TRUE(result.measured_profile.measured_flops_verified);
  EXPECT_EQ(s.state(), mb::MeasurementState::DONE);
  EXPECT_EQ(timer.pending(), 0u);
}

TEST(DeviceMeasurementSession, FlopsVerificationMismatch) {
  std::vector<CapturedSend> sent;
  FakeTimer timer;

  mb::DeviceMeasurementSession::Config cfg;
  cfg.enable_flops = true;
  cfg.flops_matrix_dim = 8;
  cfg.flops_tolerance = 1e-3;
  cfg.seed = 0xBADULL;

  auto ref_gemm = [](const mb::ReferenceGemmRequest& req,
                     mb::ReferenceGemmDoneFn done) {
    auto c = std::make_shared<std::vector<float>>(static_cast<size_t>(req.m) *
                                                  req.n);
    CpuGemm(req.m, req.n, req.k, req.a, req.b, c->data());
    done(true, c);
  };

  mb::DeviceMeasurementSession s(
      /*device_id=*/7, cfg,
      [&](SerializationBufferPtr buf) {
        sent.push_back(CapturedSend{std::vector<uint8_t>(
            static_cast<const uint8_t*>(buf->GetBuffer()),
            static_cast<const uint8_t*>(buf->GetBuffer()) + buf->GetSize())});
        return true;
      },
      {[&](double sec, std::function<void()> cb) {
         return timer.Arm(sec, std::move(cb));
       },
       [&](uint64_t id) { timer.Cancel(id); }},
      ref_gemm);

  bool fired = false;
  mb::MeasurementResult result;
  s.Start([&](const mb::MeasurementResult& r) {
    fired = true;
    result = r;
  });

  ASSERT_EQ(s.state(), mb::MeasurementState::FLOPS_PENDING);
  ASSERT_EQ(sent.size(), 1u);

  // Build a *wrong* device-side C (all zeros).
  mb::ProbeMessageCodec::FlopsRequestView v;
  ASSERT_TRUE(mb::ProbeMessageCodec::ParseFlopsRequest(
      sent[0].bytes.data(), sent[0].bytes.size(), v));
  std::vector<float> wrong_c(static_cast<size_t>(v.m) * v.n, 999.0f);
  auto rsp = mb::ProbeMessageCodec::SerializeFlopsResponse(
      v.probe_id, wrong_c.data(), v.m, v.n);
  ASSERT_TRUE(s.OnProbeFlopsResponse(rsp->GetBuffer(), rsp->GetSize()));

  EXPECT_TRUE(fired);
  // ok mirrors `verified`; mismatch -> ok=false, state FAILED.
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.terminal_state, mb::MeasurementState::FAILED);
  EXPECT_FALSE(result.measured_profile.measured_flops_verified);
  // FLOPS number is still recorded (so the operator can see it was alive).
  EXPECT_GT(result.measured_profile.measured_flops, 0ULL);
}

TEST(DeviceMeasurementSession, LatencyTimeoutFails) {
  std::vector<CapturedSend> sent;
  FakeTimer timer;

  mb::DeviceMeasurementSession::Config cfg;
  cfg.enable_latency = true;
  cfg.probe_timeout_sec = 0.001;

  mb::DeviceMeasurementSession s(
      /*device_id=*/3, cfg,
      [&](SerializationBufferPtr buf) {
        sent.push_back(CapturedSend{std::vector<uint8_t>(
            static_cast<const uint8_t*>(buf->GetBuffer()),
            static_cast<const uint8_t*>(buf->GetBuffer()) + buf->GetSize())});
        return true;
      },
      {[&](double sec, std::function<void()> cb) {
         return timer.Arm(sec, std::move(cb));
       },
       [&](uint64_t id) { timer.Cancel(id); }},
      /*reference_gemm=*/nullptr);

  bool fired = false;
  mb::MeasurementResult result;
  s.Start([&](const mb::MeasurementResult& r) {
    fired = true;
    result = r;
  });

  ASSERT_EQ(s.state(), mb::MeasurementState::LAT_PENDING);
  ASSERT_EQ(sent.size(), 1u);

  // Trigger the timer instead of delivering the response.
  timer.FireAll();

  EXPECT_TRUE(fired);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.terminal_state, mb::MeasurementState::FAILED);
  EXPECT_EQ(s.state(), mb::MeasurementState::FAILED);
}

TEST(DeviceMeasurementSession, StaleProbeIdIgnored) {
  std::vector<CapturedSend> sent;
  FakeTimer timer;

  mb::DeviceMeasurementSession::Config cfg;
  cfg.enable_latency = true;
  cfg.probe_timeout_sec = 10.0;

  mb::DeviceMeasurementSession s(
      /*device_id=*/9, cfg,
      [&](SerializationBufferPtr buf) {
        sent.push_back(CapturedSend{std::vector<uint8_t>(
            static_cast<const uint8_t*>(buf->GetBuffer()),
            static_cast<const uint8_t*>(buf->GetBuffer()) + buf->GetSize())});
        return true;
      },
      {[&](double sec, std::function<void()> cb) {
         return timer.Arm(sec, std::move(cb));
       },
       [&](uint64_t id) { timer.Cancel(id); }},
      /*reference_gemm=*/nullptr);

  bool fired = false;
  mb::MeasurementResult result;
  s.Start([&](const mb::MeasurementResult& r) {
    fired = true;
    result = r;
  });

  ASSERT_EQ(s.state(), mb::MeasurementState::LAT_PENDING);

  // Forge a response with a different probe_id; session should ignore it.
  std::string payload(64, 'X');
  auto stale = mb::ProbeMessageCodec::SerializeLatencyResponse(
      /*probe_id=*/999, payload.data(), payload.size());
  EXPECT_FALSE(s.OnProbeLatencyResponse(stale->GetBuffer(), stale->GetSize()));
  EXPECT_FALSE(fired);
  EXPECT_EQ(s.state(), mb::MeasurementState::LAT_PENDING);
}
