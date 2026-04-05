#include <benchmark/benchmark.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

namespace {

class AffinitySnapshot {
 public:
  AffinitySnapshot() {
    CPU_ZERO(&saved_);
    has_snapshot_ = (sched_getaffinity(0, sizeof(cpu_set_t), &saved_) == 0);
  }

  ~AffinitySnapshot() {
    if (has_snapshot_) {
      sched_setaffinity(0, sizeof(cpu_set_t), &saved_);
    }
  }

 private:
  cpu_set_t saved_{};
  bool has_snapshot_ = false;
};

int GetOnlineCoreCount() {
  int n = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
  return std::max(1, n);
}

cpu_set_t BuildCpuSet(int core_count) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int core = 0; core < core_count; ++core) {
    CPU_SET(core, &cpuset);
  }
  return cpuset;
}

void BM_SetAffinity_SingleCore(benchmark::State& state) {
  AffinitySnapshot snapshot;
  cpu_set_t cpuset = BuildCpuSet(1);
  for (auto _ : state) {
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
  }
  state.counters["switches_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

void BM_SetAffinity_VaryingCores(benchmark::State& state) {
  AffinitySnapshot snapshot;
  const int requested = static_cast<int>(state.range(0));
  const int target = std::max(1, std::min(requested, GetOnlineCoreCount()));
  cpu_set_t cpuset = BuildCpuSet(target);
  for (auto _ : state) {
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
  }
  state.counters["switches_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

void BM_SetAffinity_SameSet(benchmark::State& state) {
  AffinitySnapshot snapshot;
  cpu_set_t cpuset = BuildCpuSet(1);
  for (auto _ : state) {
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
  }
  state.counters["switches_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

void BM_AffinitySlotLookup(benchmark::State& state) {
  AffinitySnapshot snapshot;
  const int target = std::max(
      1, std::min(static_cast<int>(state.range(0)), GetOnlineCoreCount()));
  std::vector<cpu_set_t> slots;
  slots.reserve(target);
  for (int core = 1; core <= target; ++core) {
    slots.push_back(BuildCpuSet(core));
  }

  std::size_t idx = 0;
  for (auto _ : state) {
    const cpu_set_t& slot = slots[idx];
    sched_setaffinity(0, sizeof(cpu_set_t), &slot);
    idx = (idx + 1) % slots.size();
  }
  state.counters["switches_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

void BM_BuildCpuSet(benchmark::State& state) {
  const int target = std::max(
      1, std::min(static_cast<int>(state.range(0)), GetOnlineCoreCount()));
  for (auto _ : state) {
    benchmark::DoNotOptimize(BuildCpuSet(target));
  }
  state.counters["switches_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_SetAffinity_SingleCore);
BENCHMARK(BM_SetAffinity_VaryingCores)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_SetAffinity_SameSet);
BENCHMARK(BM_AffinitySlotLookup)->Arg(1)->Arg(2)->Arg(4)->Arg(8);
BENCHMARK(BM_BuildCpuSet)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

}  // namespace
