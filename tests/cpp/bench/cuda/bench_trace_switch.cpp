// bench_trace_switch.cpp — Trace-driven green context switch benchmark.
//
// Replays scheduling decisions from a trace CSV and measures whether
// green context switches complete within the per-entry budget
// (decode_start_est_ns − sched_ns).  Reports violation count/rate
// and switch latency percentiles.
//
// Trace CSV columns used:
//   time_slot_sched_ns        — scheduling decision timestamp  (ns)
//   time_decode_start_est_ns  — predicted decode start         (ns)
//   sm_count                  — target SM partition count
//
// Environment:
//   TRACE_CSV=<path>  (default: data/example_trace.csv)
//
// Benchmarks:
//   TraceSwitchBench/ReplayViolation
//       Replays trace faithfully; tracks violations vs budget.
//   TraceSwitchBench/ForcedSwitch
//       Forces a real cuCtxSetCurrent every entry (alternates
//       with a different-SM "away" context) so the measured
//       latency always reflects an actual context switch.

#include <benchmark/benchmark.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "bench_cuda_utils.h"

// ===================================================================
// Trace CSV parser
// ===================================================================

struct TraceEntry {
  int64_t sched_ns;
  int64_t decode_start_est_ns;
  int sm_count;

  int64_t budget_ns() const { return decode_start_est_ns - sched_ns; }
};

/// Parse a scheduling-trace CSV.  Locates the three required
/// columns by header name so column order does not matter.
static std::vector<TraceEntry> ParseTraceCsv(const std::string& path) {
  std::vector<TraceEntry> out;
  std::ifstream f(path);
  if (!f.is_open()) return out;

  std::string line;
  if (!std::getline(f, line)) return out;

  // ---- header -------------------------------------------------
  std::istringstream hdr(line);
  std::string col;
  int idx = 0, ci_sched = -1, ci_decode = -1, ci_sm = -1;
  while (std::getline(hdr, col, ',')) {
    if (col == "time_slot_sched_ns")
      ci_sched = idx;
    else if (col == "time_decode_start_est_ns")
      ci_decode = idx;
    else if (col == "sm_count")
      ci_sm = idx;
    ++idx;
  }
  if (ci_sched < 0 || ci_decode < 0 || ci_sm < 0) return out;

  const int need = std::max({ci_sched, ci_decode, ci_sm}) + 1;

  // ---- data rows ----------------------------------------------
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    std::vector<std::string> flds;
    std::string tok;
    while (std::getline(row, tok, ',')) flds.push_back(tok);
    if (static_cast<int>(flds.size()) < need) continue;
    out.push_back({std::stoll(flds[ci_sched]), std::stoll(flds[ci_decode]),
                   std::stoi(flds[ci_sm])});
  }
  return out;
}

static std::string GetTracePath() {
  const char* env = std::getenv("TRACE_CSV");
  return (env && env[0]) ? std::string(env) : "data/example_trace.csv";
}

// ===================================================================
// Green-context bookkeeping
// ===================================================================

struct GreenCtxInfo {
  int sm_count = 0;
  unsigned int num_groups = 0;
  CUdevResourceDesc desc = nullptr;
  CUgreenCtx green_ctx = nullptr;
  CUcontext cuda_ctx = nullptr;
};

// ===================================================================
// Fixture
// ===================================================================

class TraceSwitchBench : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();

    trace_ = ParseTraceCsv(GetTracePath());
    if (trace_.empty()) {
      state.SkipWithMessage("Trace CSV empty or not found");
      return;
    }

    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));
    total_sm_ = GetSmCount();

    // -- split device SMs into finest groups --------------------
    CUdevResource dev_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &dev_sm, CU_DEV_RESOURCE_TYPE_SM));

    unsigned int nb = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb, &dev_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));
    if (nb == 0) {
      state.SkipWithMessage("SM split not supported");
      return;
    }
    groups_.resize(nb);
    CUdevResource rem = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        groups_.data(), &nb, &dev_sm, &rem,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));
    nb_groups_ = nb;
    group_sm_ = groups_[0].sm.smCount;

    // -- collect unique group-counts we need --------------------
    //  • one per unique trace sm_count (rounded down)
    //  • one extra at 1-group for forced-switch alternate ctx
    std::set<unsigned int> needed_ng;
    for (auto& e : trace_) {
      unsigned int ng = static_cast<unsigned int>(e.sm_count) / group_sm_;
      if (ng == 0) ng = 1;
      if (ng > nb_groups_) ng = nb_groups_;
      needed_ng.insert(ng);
    }
    needed_ng.insert(1);  // alt ctx for ForcedSwitch

    // -- allocate non-overlapping group ranges (large first) ----
    std::vector<unsigned int> sorted_ng(needed_ng.rbegin(), needed_ng.rend());
    unsigned int offset = 0;
    for (unsigned int ng : sorted_ng) {
      if (offset + ng > nb_groups_) {
        ng = nb_groups_ - offset;
        if (ng == 0) continue;
      }
      int sm = static_cast<int>(ng * group_sm_);

      GreenCtxInfo info;
      info.sm_count = sm;
      info.num_groups = ng;
      CHECK_CU_RESULT(
          cuDevResourceGenerateDesc(&info.desc, &groups_[offset], ng));
      CHECK_CU_RESULT(cuGreenCtxCreate(&info.green_ctx, info.desc, dev_,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&info.cuda_ctx, info.green_ctx));

      pool_[sm] = info;
      offset += ng;
    }
    if (pool_.empty()) {
      state.SkipWithMessage("Failed to create any green contexts");
      return;
    }

    // -- map each trace entry to the best available ctx ---------
    ctx_for_entry_.reserve(trace_.size());
    sm_for_entry_.reserve(trace_.size());
    for (auto& e : trace_) {
      unsigned int ng = static_cast<unsigned int>(e.sm_count) / group_sm_;
      if (ng == 0) ng = 1;
      if (ng > nb_groups_) ng = nb_groups_;
      int target = static_cast<int>(ng * group_sm_);

      // exact match first, then closest
      if (!pool_.count(target)) {
        int best = pool_.begin()->first;
        for (auto& [sm, _] : pool_) {
          if (std::abs(sm - target) < std::abs(best - target)) best = sm;
        }
        target = best;
      }
      ctx_for_entry_.push_back(pool_[target].cuda_ctx);
      sm_for_entry_.push_back(pool_[target].sm_count);
    }

    // -- find alternate ctx for forced switching ----------------
    int primary_sm = sm_for_entry_[0];
    alt_ctx_ = nullptr;
    for (auto& [sm, info] : pool_) {
      if (sm != primary_sm) {
        alt_ctx_ = info.cuda_ctx;
        alt_sm_ = sm;
        break;
      }
    }

    // -- warmup: touch every context once -----------------------
    for (auto& [sm, info] : pool_)
      CHECK_CU_RESULT(cuCtxSetCurrent(info.cuda_ctx));
  }

  void TearDown(benchmark::State&) override {
    for (auto& [sm, info] : pool_) {
      if (info.cuda_ctx) cuCtxDestroy(info.cuda_ctx);
      if (info.green_ctx) cuGreenCtxDestroy(info.green_ctx);
    }
    pool_.clear();
    ctx_for_entry_.clear();
    sm_for_entry_.clear();
    // Restore primary context so CUDA runtime cleanup
    // does not SIGSEGV.
    cudaSetDevice(0);
  }

 protected:
  CUdevice dev_ = 0;
  int total_sm_ = 0;
  unsigned int nb_groups_ = 0;
  unsigned int group_sm_ = 0;
  std::vector<CUdevResource> groups_;

  std::vector<TraceEntry> trace_;
  std::map<int, GreenCtxInfo> pool_;  // actual_sm → info
  std::vector<CUcontext> ctx_for_entry_;
  std::vector<int> sm_for_entry_;

  CUcontext alt_ctx_ = nullptr;  // for ForcedSwitch
  int alt_sm_ = 0;
};

// ===================================================================
// Benchmark 1 — Trace replay with violation tracking
// ===================================================================
//
// Replays every trace entry's SM count as a cuCtxSetCurrent call.
// If the switch latency exceeds the per-entry budget
// (decode_start_est − sched) the entry counts as a violation.
//
// When consecutive entries share the same sm_count the driver call
// is essentially a no-op, which is the realistic behavior.

BENCHMARK_DEFINE_F(TraceSwitchBench, ReplayViolation)
(benchmark::State& state) {
  using Clock = std::chrono::high_resolution_clock;

  const size_t n = trace_.size();
  std::vector<int64_t> latencies;
  latencies.reserve(n * 128);

  int64_t total_violations = 0;
  int64_t total_actual_switches = 0;
  int64_t total_entries = 0;

  for (auto _ : state) {
    CUcontext cur = nullptr;
    cuCtxGetCurrent(&cur);

    for (size_t i = 0; i < n; ++i) {
      CUcontext tgt = ctx_for_entry_[i];
      int64_t budget = trace_[i].budget_ns();

      auto t0 = Clock::now();
      CHECK_CU_RESULT(cuCtxSetCurrent(tgt));
      auto t1 = Clock::now();

      int64_t lat =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

      latencies.push_back(lat);
      if (tgt != cur) ++total_actual_switches;
      if (lat > budget) ++total_violations;
      cur = tgt;
    }
    total_entries += static_cast<int64_t>(n);
  }

  // -- percentiles ----------------------------------------------
  std::sort(latencies.begin(), latencies.end());
  const size_t cnt = latencies.size();
  auto pctl = [&](double p) -> double {
    if (cnt == 0) return 0.0;
    size_t idx = static_cast<size_t>(p * static_cast<double>(cnt - 1));
    return static_cast<double>(latencies[idx]);
  };

  // -- budget range ---------------------------------------------
  int64_t bmin = INT64_MAX, bmax = 0;
  for (auto& e : trace_) {
    int64_t b = e.budget_ns();
    bmin = std::min(bmin, b);
    bmax = std::max(bmax, b);
  }

  // -- counters -------------------------------------------------
  state.counters["trace_entries"] = static_cast<double>(n);
  state.counters["violations"] = static_cast<double>(total_violations);
  state.counters["violation_rate"] =
      total_entries > 0 ? static_cast<double>(total_violations) /
                              static_cast<double>(total_entries)
                        : 0.0;
  state.counters["actual_ctx_switches"] =
      static_cast<double>(total_actual_switches);
  state.counters["switch_p50_ns"] = pctl(0.50);
  state.counters["switch_p99_ns"] = pctl(0.99);
  state.counters["switch_max_ns"] =
      cnt > 0 ? static_cast<double>(latencies.back()) : 0.0;
  state.counters["switch_min_ns"] =
      cnt > 0 ? static_cast<double>(latencies.front()) : 0.0;
  state.counters["budget_min_us"] = static_cast<double>(bmin) / 1e3;
  state.counters["budget_max_us"] = static_cast<double>(bmax) / 1e3;
  state.counters["gpu_total_SMs"] = total_sm_;
  state.counters["unique_sm_cfgs"] = static_cast<double>(pool_.size());
}

BENCHMARK_REGISTER_F(TraceSwitchBench, ReplayViolation)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100);

// ===================================================================
// Benchmark 2 — Forced context switch (worst-case latency)
// ===================================================================
//
// Before each measured cuCtxSetCurrent we unconditionally set an
// "away" green context with a different SM count, guaranteeing
// every measured call is a *real* context switch rather than a
// same-context no-op.  Still compares against the trace budget.

BENCHMARK_DEFINE_F(TraceSwitchBench, ForcedSwitch)
(benchmark::State& state) {
  if (!alt_ctx_) {
    state.SkipWithMessage(
        "Only 1 SM config available; "
        "cannot force switches");
    return;
  }

  using Clock = std::chrono::high_resolution_clock;

  const size_t n = trace_.size();
  std::vector<int64_t> latencies;
  latencies.reserve(n * 128);

  int64_t total_violations = 0;
  int64_t total_entries = 0;

  for (auto _ : state) {
    for (size_t i = 0; i < n; ++i) {
      CUcontext tgt = ctx_for_entry_[i];
      int64_t budget = trace_[i].budget_ns();

      // Force a real switch: move to alt ctx (unmeasured)
      cuCtxSetCurrent(alt_ctx_);

      // Measured switch to trace target
      auto t0 = Clock::now();
      CHECK_CU_RESULT(cuCtxSetCurrent(tgt));
      auto t1 = Clock::now();

      int64_t lat =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

      latencies.push_back(lat);
      if (lat > budget) ++total_violations;
    }
    total_entries += static_cast<int64_t>(n);
  }

  // -- percentiles ----------------------------------------------
  std::sort(latencies.begin(), latencies.end());
  const size_t cnt = latencies.size();
  auto pctl = [&](double p) -> double {
    if (cnt == 0) return 0.0;
    size_t idx = static_cast<size_t>(p * static_cast<double>(cnt - 1));
    return static_cast<double>(latencies[idx]);
  };

  int64_t bmin = INT64_MAX;
  for (auto& e : trace_) bmin = std::min(bmin, e.budget_ns());

  state.counters["trace_entries"] = static_cast<double>(n);
  state.counters["violations"] = static_cast<double>(total_violations);
  state.counters["violation_rate"] =
      total_entries > 0 ? static_cast<double>(total_violations) /
                              static_cast<double>(total_entries)
                        : 0.0;
  state.counters["forced_switch_p50_ns"] = pctl(0.50);
  state.counters["forced_switch_p99_ns"] = pctl(0.99);
  state.counters["forced_switch_max_ns"] =
      cnt > 0 ? static_cast<double>(latencies.back()) : 0.0;
  state.counters["forced_switch_min_ns"] =
      cnt > 0 ? static_cast<double>(latencies.front()) : 0.0;
  state.counters["budget_min_us"] = static_cast<double>(bmin) / 1e3;
  state.counters["alt_sm_count"] = alt_sm_;
}

BENCHMARK_REGISTER_F(TraceSwitchBench, ForcedSwitch)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100);
