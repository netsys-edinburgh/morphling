#include <benchmark/benchmark.h>

#include <memory>
#include <mutex>
#include <string>

#include "bench_cuda_utils.h"
#include "core/logger.h"
#include "scheduler/gpu_worker.h"
#include "scheduler/sm_notify_source.h"

namespace {

// A fixed SM target: SnapDown clamps a large value to the partition maximum, so
// the chunked loop runs without ever switching contexts. This isolates the
// chunking + adaptive-sizing overhead from switch overhead.
class ConstSource : public SmTargetSource {
 public:
  explicit ConstSource(int v) : v_(v) {}
  int Read() override { return v_; }

 private:
  int v_;
};

XtGemmWorker::DynConfig ChunkedCfg() {
  XtGemmWorker::DynConfig c;
  c.enabled = true;
  c.chunk_target_us = 200;
  c.min_chunk_cols = 128;
  c.min_gemm_chunk_threshold = 0;
  return c;
}

void RunGemmLoop(benchmark::State& state, std::shared_ptr<XtGemmWorker> w,
                 int dim) {
  PinnedBuffer h_A(static_cast<size_t>(dim) * dim);
  PinnedBuffer h_B(static_cast<size_t>(dim) * dim);
  PinnedBuffer h_C(static_cast<size_t>(dim) * dim);
  auto args = MakeNNGemmArgs(dim, dim, dim, h_A.ptr, h_B.ptr, h_C.ptr);

  w->AddTask("warmup", [&]() { w->RunChunkedGemm(args); });
  w->WaitTaskDone("warmup");

  int iter = 0;
  for (auto _ : state) {
    std::string tid = "g_" + std::to_string(iter++);
    w->AddTask(tid, [&]() { w->RunChunkedGemm(args); });
    w->WaitTaskDone(tid);
  }
  state.counters["GFLOPS"] = benchmark::Counter(
      2.0 * dim * dim * dim, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  w->Stop();
}

}  // namespace

// Single-shot plain cuBLAS at partition max — the default proxy path.
static void BM_ChunkedGemm_SingleShot(benchmark::State& state) {
  static std::once_flag flag;
  std::call_once(flag, []() { InitLogger(); });
  RunGemmLoop(state, std::make_shared<XtGemmWorker>(0, 1, 0, 512_MB),
              static_cast<int>(state.range(0)));
}
BENCHMARK(BM_ChunkedGemm_SingleShot)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// Adaptive chunked path with a constant SM target (no switch): measures the
// chunking overhead relative to single-shot.
static void BM_ChunkedGemm_ChunkedNoSwitch(benchmark::State& state) {
  static std::once_flag flag;
  std::call_once(flag, []() { InitLogger(); });
  auto w = std::make_shared<XtGemmWorker>(
      0, 1, 0, 512_MB, ChunkedCfg(), std::make_unique<ConstSource>(1000000));
  RunGemmLoop(state, w, static_cast<int>(state.range(0)));
}
BENCHMARK(BM_ChunkedGemm_ChunkedNoSwitch)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);
