#include <benchmark/benchmark.h>

#include "scheduler/sliding_window_tracker.h"

namespace {

constexpr int64_t kWarmupSamples = 64;

static void BM_RecordDuration(benchmark::State& state) {
  SlidingWindowDurationTracker<> tracker;
  int64_t fake_duration = 42;
  for (auto _ : state) {
    tracker.RecordDuration(fake_duration);
    benchmark::DoNotOptimize(fake_duration);
    ++fake_duration;
  }
  state.counters["records_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
  state.counters["ns_per_record"] = benchmark::Counter(0.0);
  state.counters["ns_per_average"] = benchmark::Counter(0.0);
}

static void BM_GetAverageDuration(benchmark::State& state) {
  SlidingWindowDurationTracker<> tracker;
  for (int i = 0; i < kWarmupSamples; ++i) {
    tracker.RecordDuration(i + 1);
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(tracker.GetAverageDurationUs());
  }
  state.counters["records_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
  state.counters["ns_per_record"] = benchmark::Counter(0.0);
  state.counters["ns_per_average"] = benchmark::Counter(0.0);
}

static void BM_NowCall(benchmark::State& state) {
  for (auto _ : state) {
    auto t = SlidingWindowDurationTracker<>::Now();
    benchmark::DoNotOptimize(t);
  }
  state.counters["records_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
  state.counters["ns_per_record"] = benchmark::Counter(0.0);
  state.counters["ns_per_average"] = benchmark::Counter(0.0);
}

static void BM_ElapsedUsCall(benchmark::State& state) {
  for (auto _ : state) {
    auto start = SlidingWindowDurationTracker<>::Now();
    benchmark::DoNotOptimize(SlidingWindowDurationTracker<>::ElapsedUs(start));
  }
  state.counters["records_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
  state.counters["ns_per_record"] = benchmark::Counter(0.0);
  state.counters["ns_per_average"] = benchmark::Counter(0.0);
}

template <int WindowSize>
static void BM_VaryingWindowSize(benchmark::State& state) {
  SlidingWindowDurationTracker<WindowSize> tracker;
  for (int i = 0; i < WindowSize; ++i) {
    tracker.RecordDuration(i + 1);
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(tracker.GetAverageDurationUs());
  }
  state.counters["records_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
  state.counters["ns_per_record"] = benchmark::Counter(0.0);
  state.counters["ns_per_average"] = benchmark::Counter(0.0);
}

}  // namespace

BENCHMARK(BM_RecordDuration);
BENCHMARK(BM_GetAverageDuration);
BENCHMARK(BM_NowCall);
BENCHMARK(BM_ElapsedUsCall);
BENCHMARK_TEMPLATE(BM_VaryingWindowSize, 16);
BENCHMARK_TEMPLATE(BM_VaryingWindowSize, 32);
BENCHMARK_TEMPLATE(BM_VaryingWindowSize, 64);
BENCHMARK_TEMPLATE(BM_VaryingWindowSize, 128);
BENCHMARK_TEMPLATE(BM_VaryingWindowSize, 256);
