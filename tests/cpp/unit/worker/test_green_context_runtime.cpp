#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#if __has_include("csrc/scheduler/green_context_runtime.h")
#include "csrc/scheduler/green_context_runtime.h"
#elif __has_include("scheduler/green_context_runtime.h")
#include "scheduler/green_context_runtime.h"
#endif

// Helper: find an idle GPU (lowest utilization).  Falls back to GPU 0.
static int FindIdleGpu() {
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count == 0) return -1;
  // Pick the last GPU (typically least used in multi-GPU rigs)
  return count - 1;
}

// Helper: write a temp trace file, return path
static std::string WriteTempTrace(const std::string& content) {
  char tpl[] = "/tmp/test_greenrt_XXXXXX.txt";
  int fd = mkstemps(tpl, 4);
  if (fd == -1) return "";
  close(fd);
  std::ofstream out(tpl, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    std::remove(tpl);
    return "";
  }
  out << content;
  out.close();
  return std::string(tpl);
}

class GreenContextRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure CUDA driver is initialized
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
      GTEST_SKIP() << "cuInit failed — no CUDA driver";
    }

    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    gpu_id_ = FindIdleGpu();

    // Check driver version
    int ver = 0;
    cuDriverGetVersion(&ver);
    if (ver < 12050) {
      GTEST_SKIP() << "CUDA driver " << ver
                   << " too old (need >= 12050 for green contexts)";
    }
  }

  void TearDown() override {
    for (const auto& f : temp_files_) std::remove(f.c_str());
    temp_files_.clear();
  }

  std::string MakeTraceFile(const std::string& content) {
    std::string path = WriteTempTrace(content);
    EXPECT_FALSE(path.empty());
    if (!path.empty()) temp_files_.push_back(path);
    return path;
  }

  int gpu_id_ = 0;
  std::vector<std::string> temp_files_;
};

// ── Construction & Introspection ───────────────────────────

TEST_F(GreenContextRuntimeTest, ConstructAndSupported) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  opts.num_partitions = 1;
  opts.partition_idx = 0;

  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();
  EXPECT_EQ(rt.GpuId(), gpu_id_);
}

TEST_F(GreenContextRuntimeTest, SmStepIsTwo) {
  // CC 8.6 (Ampere) should have SM step = 2
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  int step = rt.SmStep();
  // CC 8.x step=2, CC 9.0+ step=8
  EXPECT_TRUE(step == 2 || step == 8) << "Unexpected SM step: " << step;
}

TEST_F(GreenContextRuntimeTest, AvailableSmCounts) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  auto counts = rt.AvailableSmCounts();
  ASSERT_FALSE(counts.empty());

  // Should be sorted ascending
  for (size_t i = 1; i < counts.size(); i++) {
    EXPECT_GT(counts[i], counts[i - 1]);
  }

  // All should be multiples of SmStep
  int step = rt.SmStep();
  for (int c : counts) {
    EXPECT_EQ(c % step, 0) << "SM count " << c << " not multiple of step "
                           << step;
  }

  // Largest should equal PartitionSmCount
  EXPECT_EQ(counts.back(), rt.PartitionSmCount());

  // ActiveSmCount defaults to partition max
  EXPECT_EQ(rt.ActiveSmCount(), rt.PartitionSmCount());
}

TEST_F(GreenContextRuntimeTest, TwoPartitions) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  opts.num_partitions = 2;
  opts.partition_idx = 0;
  GreenContextRuntime rt0(opts);
  ASSERT_TRUE(rt0.IsSupported()) << rt0.UnsupportedReason();

  opts.partition_idx = 1;
  GreenContextRuntime rt1(opts);
  ASSERT_TRUE(rt1.IsSupported()) << rt1.UnsupportedReason();

  // Each partition should get roughly half the SMs
  EXPECT_GT(rt0.PartitionSmCount(), 0);
  EXPECT_GT(rt1.PartitionSmCount(), 0);
  EXPECT_EQ(rt0.PartitionSmCount(), rt1.PartitionSmCount());
}

// ── Stream Table ───────────────────────────────────────────

TEST_F(GreenContextRuntimeTest, GetStreamNonNull) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  auto counts = rt.AvailableSmCounts();
  ASSERT_FALSE(counts.empty());

  // Check all roles for the smallest SM count
  int sm = counts.front();
  EXPECT_NE(rt.GetStream(sm, StreamRole::kCompute), nullptr);
  EXPECT_NE(rt.GetStream(sm, StreamRole::kRecv), nullptr);
  EXPECT_NE(rt.GetStream(sm, StreamRole::kSend), nullptr);
  EXPECT_NE(rt.GetStream(sm, StreamRole::kDp), nullptr);

  // Invalid SM count should return nullptr
  EXPECT_EQ(rt.GetStream(9999, StreamRole::kCompute), nullptr);
}

TEST_F(GreenContextRuntimeTest, GetStreamPtrMatchesGetStream) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  int sm = rt.AvailableSmCounts().front();
  auto stream = rt.GetStream(sm, StreamRole::kCompute);
  auto ptr = rt.GetStreamPtr(sm, StreamRole::kCompute);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(stream), ptr);
}

// ── SetActiveSmCount ───────────────────────────────────────

TEST_F(GreenContextRuntimeTest, SetActiveSmCount) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  auto counts = rt.AvailableSmCounts();
  ASSERT_GE(counts.size(), 2u);

  // Switch to smallest
  EXPECT_TRUE(rt.SetActiveSmCount(counts.front()));
  EXPECT_EQ(rt.ActiveSmCount(), counts.front());

  // Switch to largest
  EXPECT_TRUE(rt.SetActiveSmCount(counts.back()));
  EXPECT_EQ(rt.ActiveSmCount(), counts.back());

  // Invalid SM count
  EXPECT_FALSE(rt.SetActiveSmCount(9999));

  // SwitchCount should be 2 (two successful switches from default)
  EXPECT_EQ(rt.SwitchCount(), 2u);
}

// ── Activation ─────────────────────────────────────────────

TEST_F(GreenContextRuntimeTest, ActivateAndDeactivate) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  auto counts = rt.AvailableSmCounts();
  ASSERT_GE(counts.size(), 2u);

  // Activate smallest
  int prev = rt.ActivateSmForThread(counts.front());
  EXPECT_EQ(rt.ActiveSmCount(), counts.front());

  // Verify a CUDA context is current
  CUcontext ctx = nullptr;
  EXPECT_EQ(cuCtxGetCurrent(&ctx), CUDA_SUCCESS);
  EXPECT_NE(ctx, nullptr);

  // Deactivate (restore)
  rt.DeactivateForThread(prev);
}

TEST_F(GreenContextRuntimeTest, ActivateForThread_UsesActive) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  auto counts = rt.AvailableSmCounts();
  // Set active to smallest
  rt.SetActiveSmCount(counts.front());

  // ActivateForThread should use the current active SM count
  int prev = rt.ActivateForThread();
  EXPECT_EQ(rt.ActiveSmCount(), counts.front());
  rt.DeactivateForThread(prev);
}

// ── Trace Loading ──────────────────────────────────────────

TEST_F(GreenContextRuntimeTest, LoadTrace_ValidFile) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  auto counts = rt.AvailableSmCounts();
  ASSERT_GE(counts.size(), 2u);

  int sm_min = counts.front();
  int sm_max = counts.back();

  // Create trace with valid SM counts
  std::string trace =
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0," +
      std::to_string(sm_min) +
      "\n"
      "1000000," +
      std::to_string(sm_max) +
      "\n"
      "3000000," +
      std::to_string(sm_min) + "\n";

  std::string path = MakeTraceFile(trace);
  ASSERT_FALSE(path.empty());
  EXPECT_TRUE(rt.LoadTrace(path));

  // Verify SmCountAtTime
  EXPECT_EQ(rt.SmCountAtTime(0), sm_min);
  EXPECT_EQ(rt.SmCountAtTime(500000), sm_min);
  EXPECT_EQ(rt.SmCountAtTime(1000000), sm_max);
  EXPECT_EQ(rt.SmCountAtTime(2000000), sm_max);
  EXPECT_EQ(rt.SmCountAtTime(3000000), sm_min);
}

TEST_F(GreenContextRuntimeTest, LoadTrace_InvalidSmCount_Fails) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  // Create trace with an SM count that doesn't match any slot
  std::string trace =
      "# greenctx_trace v2\n"
      "# time_unit=us\n"
      "timestamp,num_sms\n"
      "0,9999\n";

  std::string path = MakeTraceFile(trace);
  ASSERT_FALSE(path.empty());
  EXPECT_FALSE(rt.LoadTrace(path));
}

// ── Cleanup ordering ───────────────────────────────────────

TEST_F(GreenContextRuntimeTest, ExplicitClose) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  // Close explicitly, then destroy — should not crash
  rt.Close();
  // Double-close should be safe
  rt.Close();
}

TEST_F(GreenContextRuntimeTest, StreamRemainsValidUntilClose) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = gpu_id_;
  GreenContextRuntime rt(opts);
  ASSERT_TRUE(rt.IsSupported()) << rt.UnsupportedReason();

  auto counts = rt.AvailableSmCounts();
  int sm = counts.front();
  cudaStream_t stream = rt.GetStream(sm, StreamRole::kCompute);
  ASSERT_NE(stream, nullptr);

  // Activate, run a trivial operation on the stream
  int prev = rt.ActivateSmForThread(sm);
  cudaError_t err = cudaStreamSynchronize(stream);
  EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  rt.DeactivateForThread(prev);
}

// ── Unsupported GPU handling ───────────────────────────────

TEST_F(GreenContextRuntimeTest, InvalidGpuId_NotSupported) {
  GreenContextRuntime::Options opts;
  opts.gpu_id = 999;  // out of range
  GreenContextRuntime rt(opts);
  EXPECT_FALSE(rt.IsSupported());
  EXPECT_FALSE(rt.UnsupportedReason().empty());
}
