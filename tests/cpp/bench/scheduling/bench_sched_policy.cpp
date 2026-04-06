#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

constexpr double kMB = 1024.0 * 1024.0;
constexpr double kTF = 1024.0 * 1024.0 * 1024.0 * 1024.0;
constexpr int kFairnessDispatches = 10000;

struct MockPartition {
  int64_t row_key;
  int64_t col_key;
  int64_t row_bytes;
  int64_t col_bytes;
  int64_t rows;
  int64_t cols;
  int64_t h_dim;
};

std::vector<MockPartition> BuildMockPartitions(int count) {
  std::vector<MockPartition> parts;
  parts.reserve(count);

  for (int i = 0; i < count; ++i) {
    const int64_t rows = 64 + (i % 7) * 64;
    const int64_t cols = 64 + (i % 5) * 128;
    const int64_t h_dim = 256 + (i % 4) * 128;
    const int64_t row_bytes =
        rows * h_dim * static_cast<int64_t>(sizeof(float));
    const int64_t col_bytes =
        cols * h_dim * static_cast<int64_t>(sizeof(float));

    parts.push_back(MockPartition{i % 97, (i * 7) % 89, row_bytes, col_bytes,
                                  rows, cols, h_dim});
  }

  return parts;
}

double ComputeFairnessStddev(const std::vector<int>& worker_counts) {
  if (worker_counts.empty()) {
    return 0.0;
  }

  const double n = static_cast<double>(worker_counts.size());
  const double sum =
      std::accumulate(worker_counts.begin(), worker_counts.end(), 0.0);
  const double mean = sum / n;

  double variance = 0.0;
  for (int count : worker_counts) {
    const double diff = static_cast<double>(count) - mean;
    variance += diff * diff;
  }
  variance /= n;

  return std::sqrt(variance);
}

class RoundRobinPolicyModel {
 public:
  explicit RoundRobinPolicyModel(int workers)
      : workers_(std::max(1, workers)), next_worker_(0) {}

  int Schedule(const MockPartition&) {
    const int selected = next_worker_;
    next_worker_ = (next_worker_ + 1) % workers_;
    return selected;
  }

 private:
  int workers_;
  int next_worker_;
};

class GreedyPolicyModel {
 public:
  explicit GreedyPolicyModel(int workers)
      : workers_(std::max(1, workers)),
        worker_time_(workers_, 0.0),
        cached_keys_(workers_) {}

  int Schedule(const MockPartition& part) {
    double best_time = std::numeric_limits<double>::max();
    int best_worker = 0;

    for (int i = 0; i < workers_; ++i) {
      const bool row_cached =
          cached_keys_[i].find(part.row_key) != cached_keys_[i].end();
      const bool col_cached =
          cached_keys_[i].find(part.col_key) != cached_keys_[i].end();

      const double ul_time =
          static_cast<double>(part.rows * part.cols * sizeof(float)) / kMB;
      const double dl_time =
          static_cast<double>((row_cached ? 0 : part.row_bytes) +
                              (col_cached ? 0 : part.col_bytes)) /
          kMB;
      const double flops = 2.0 * static_cast<double>(part.rows) *
                           static_cast<double>(part.cols) *
                           static_cast<double>(part.h_dim) / kTF;

      const double score =
          std::max(std::max(ul_time, dl_time), flops) + worker_time_[i];
      if (score < best_time) {
        best_time = score;
        best_worker = i;
      }
    }

    worker_time_[best_worker] = best_time;
    cached_keys_[best_worker].insert(part.row_key);
    cached_keys_[best_worker].insert(part.col_key);
    return best_worker;
  }

 private:
  int workers_;
  std::vector<double> worker_time_;
  std::vector<std::unordered_set<int64_t>> cached_keys_;
};

class LoadBalancedPolicyModel {
 public:
  explicit LoadBalancedPolicyModel(int workers)
      : workers_(std::max(1, workers)), current_load_(workers_, 0) {}

  int Schedule(const MockPartition&) {
    int best_worker = 0;
    int min_load = current_load_[0];
    for (int i = 1; i < workers_; ++i) {
      if (current_load_[i] < min_load) {
        min_load = current_load_[i];
        best_worker = i;
      }
    }
    ++current_load_[best_worker];
    return best_worker;
  }

 private:
  int workers_;
  std::vector<int> current_load_;
};

template <typename Policy>
void RunDispatchBenchmark(benchmark::State& state) {
  const int workers = static_cast<int>(state.range(0));
  const auto partitions = BuildMockPartitions(1024);
  Policy policy(workers);
  size_t idx = 0;

  for (auto _ : state) {
    (void)_;
    const int assigned = policy.Schedule(partitions[idx]);
    benchmark::DoNotOptimize(static_cast<int64_t>(assigned));
    idx = (idx + 1) % partitions.size();
  }

  state.SetItemsProcessed(state.iterations());
  state.counters["dispatches_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

template <typename Policy>
void RunFairnessBenchmark(benchmark::State& state) {
  const int workers = static_cast<int>(state.range(0));
  const auto partitions = BuildMockPartitions(1024);
  double stddev_sum = 0.0;

  for (auto _ : state) {
    (void)_;
    Policy policy(workers);
    std::vector<int> assignment_counts(workers, 0);

    for (int i = 0; i < kFairnessDispatches; ++i) {
      const int assigned = policy.Schedule(partitions[i % partitions.size()]);
      ++assignment_counts[assigned];
    }

    stddev_sum += ComputeFairnessStddev(assignment_counts);
    benchmark::ClobberMemory();
  }

  const double avg_stddev =
      stddev_sum / static_cast<double>(state.iterations());

  state.SetItemsProcessed(state.iterations() * kFairnessDispatches);
  state.counters["dispatches_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations() * kFairnessDispatches),
      benchmark::Counter::kIsRate);
  state.counters["fairness_stddev"] =
      benchmark::Counter(avg_stddev, benchmark::Counter::kAvgThreads);
}

static void BM_RoundRobin_Dispatch(benchmark::State& state) {
  RunDispatchBenchmark<RoundRobinPolicyModel>(state);
}

static void BM_Greedy_Dispatch(benchmark::State& state) {
  RunDispatchBenchmark<GreedyPolicyModel>(state);
}

static void BM_LoadBalanced_Dispatch(benchmark::State& state) {
  RunDispatchBenchmark<LoadBalancedPolicyModel>(state);
}

static void BM_PolicyFairness_RoundRobin(benchmark::State& state) {
  RunFairnessBenchmark<RoundRobinPolicyModel>(state);
}

static void BM_PolicyFairness_Greedy(benchmark::State& state) {
  RunFairnessBenchmark<GreedyPolicyModel>(state);
}

static void BM_PolicyFairness_LoadBalanced(benchmark::State& state) {
  RunFairnessBenchmark<LoadBalancedPolicyModel>(state);
}

BENCHMARK(BM_RoundRobin_Dispatch)->Arg(2)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(BM_Greedy_Dispatch)->Arg(2)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(BM_LoadBalanced_Dispatch)->Arg(2)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(BM_PolicyFairness_RoundRobin)->Arg(2)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(BM_PolicyFairness_Greedy)->Arg(2)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(BM_PolicyFairness_LoadBalanced)->Arg(2)->Arg(4)->Arg(8)->Arg(16);
