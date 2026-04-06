#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../bench_common_utils.h"
#include "../cuda/bench_cuda_utils.h"

constexpr int kWarmupIterations = 3;

static bool EnsureCudaReady(benchmark::State& state) {
  if (!CheckCudaAvailable()) {
    state.SkipWithMessage("No GPU");
    return false;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    state.SkipWithMessage("No GPU");
    return false;
  }
  return true;
}

static std::string GetPipelineMode() {
  const char* env = std::getenv("MORPHLING_WORKER_PIPELINE");
  if (env == nullptr || env[0] == '\0') {
    return "legacy";
  }
  return std::string(env);
}

static void WarmupWorker(const std::shared_ptr<XtGemmWorker>& worker,
                         const std::shared_ptr<GemmArgs>& args) {
  for (int i = 0; i < kWarmupIterations; ++i) {
    std::string warm_id = "warmup_" + std::to_string(i);
    worker->AddTask(warm_id, [worker, args]() { worker->RunXtGemm(args); });
    worker->WaitTaskDone(warm_id);
  }
}

static double RunSingleTaskTimed(const std::shared_ptr<XtGemmWorker>& worker,
                                 const std::shared_ptr<GemmArgs>& args,
                                 const std::string& task_id) {
  std::shared_ptr<double> elapsed_us = std::make_shared<double>(0.0);
  worker->AddTask(task_id, [worker, args, elapsed_us]() {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&start_evt, cudaEventDefault));
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&stop_evt, cudaEventDefault));

    CHECK_CUDA_ERROR(cudaEventRecord(start_evt, worker->GetStream()));
    worker->RunXtGemm(args);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, worker->GetStream()));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    *elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
  });
  worker->WaitTaskDone(task_id);
  return *elapsed_us;
}

struct BurstTiming {
  double total_us = 0.0;
  double sum_task_us = 0.0;
};

static BurstTiming RunBurstTimed(
    const std::shared_ptr<XtGemmWorker>& worker,
    const std::vector<std::shared_ptr<GemmArgs>>& args_vec, int burst_size,
    const std::string& prefix) {
  struct MarkerState {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    double total_us = 0.0;

    ~MarkerState() {
      if (start_evt) {
        cudaEventDestroy(start_evt);
      }
      if (stop_evt) {
        cudaEventDestroy(stop_evt);
      }
    }
  };

  std::shared_ptr<MarkerState> marker = std::make_shared<MarkerState>();
  std::shared_ptr<double> sum_task_us = std::make_shared<double>(0.0);

  const std::string begin_id = prefix + "_begin";
  worker->AddTask(begin_id, [worker, marker]() {
    CHECK_CUDA_ERROR(
        cudaEventCreateWithFlags(&marker->start_evt, cudaEventDefault));
    CHECK_CUDA_ERROR(
        cudaEventCreateWithFlags(&marker->stop_evt, cudaEventDefault));
    CHECK_CUDA_ERROR(cudaEventRecord(marker->start_evt, worker->GetStream()));
  });

  std::vector<std::string> task_ids;
  task_ids.reserve(static_cast<size_t>(burst_size));
  for (int i = 0; i < burst_size; ++i) {
    const std::string task_id = prefix + "_task_" + std::to_string(i);
    task_ids.push_back(task_id);
    const auto& args = args_vec[static_cast<size_t>(i)];
    worker->AddTask(task_id, [worker, args, sum_task_us]() {
      cudaEvent_t start_evt = nullptr;
      cudaEvent_t stop_evt = nullptr;
      CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&start_evt, cudaEventDefault));
      CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&stop_evt, cudaEventDefault));

      CHECK_CUDA_ERROR(cudaEventRecord(start_evt, worker->GetStream()));
      worker->RunXtGemm(args);
      CHECK_CUDA_ERROR(cudaEventRecord(stop_evt, worker->GetStream()));
      CHECK_CUDA_ERROR(cudaEventSynchronize(stop_evt));

      float elapsed_ms = 0.0f;
      CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
      *sum_task_us += static_cast<double>(elapsed_ms) * 1000.0;

      cudaEventDestroy(start_evt);
      cudaEventDestroy(stop_evt);
    });
  }

  const std::string end_id = prefix + "_end";
  worker->AddTask(end_id, [worker, marker]() {
    CHECK_CUDA_ERROR(cudaEventRecord(marker->stop_evt, worker->GetStream()));
    CHECK_CUDA_ERROR(cudaEventSynchronize(marker->stop_evt));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(
        cudaEventElapsedTime(&elapsed_ms, marker->start_evt, marker->stop_evt));
    marker->total_us = static_cast<double>(elapsed_ms) * 1000.0;
  });

  worker->WaitTaskDone(end_id);
  return {marker->total_us, *sum_task_us};
}

static void BM_Gemm_SingleTask(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  if (!EnsureCudaReady(state)) {
    return;
  }

  const std::string mode = GetPipelineMode();
  const int dim = static_cast<int>(state.range(0));
  const size_t elems = static_cast<size_t>(dim) * dim;

  PinnedBuffer h_a(elems);
  PinnedBuffer h_b(elems);
  PinnedBuffer h_c(elems);

  auto worker = std::make_shared<XtGemmWorker>(0, 1, 0, 512_MB);
  auto args = MakeNNGemmArgs(dim, dim, dim, h_a.ptr, h_b.ptr, h_c.ptr);

  WarmupWorker(worker, args);

  double total_us_acc = 0.0;
  int iter = 0;
  for (auto _ : state) {
    const std::string task_id = "single_" + std::to_string(iter++);
    const double task_us = RunSingleTaskTimed(worker, args, task_id);
    total_us_acc += task_us;
    state.SetIterationTime(task_us / 1e6);
  }

  const double flops = 2.0 * dim * dim * dim;
  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["total_us"] = total_us_acc / iters;
  state.SetLabel("mode=" + mode);

  worker->Stop();
}

BENCHMARK(BM_Gemm_SingleTask)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void BM_Gemm_Burst(benchmark::State& state) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { InitLogger(); });

  if (!EnsureCudaReady(state)) {
    return;
  }

  const std::string mode = GetPipelineMode();
  const int dim = static_cast<int>(state.range(0));
  const int burst_size = static_cast<int>(state.range(1));
  const size_t elems = static_cast<size_t>(dim) * dim;

  std::vector<PinnedBuffer> h_a;
  std::vector<PinnedBuffer> h_b;
  std::vector<PinnedBuffer> h_c;
  h_a.reserve(static_cast<size_t>(burst_size));
  h_b.reserve(static_cast<size_t>(burst_size));
  h_c.reserve(static_cast<size_t>(burst_size));

  std::vector<std::shared_ptr<GemmArgs>> args_vec;
  args_vec.reserve(static_cast<size_t>(burst_size));
  for (int i = 0; i < burst_size; ++i) {
    h_a.emplace_back(elems);
    h_b.emplace_back(elems);
    h_c.emplace_back(elems);
    args_vec.push_back(MakeNNGemmArgs(dim, dim, dim, h_a.back().ptr,
                                      h_b.back().ptr, h_c.back().ptr));
  }

  auto worker = std::make_shared<XtGemmWorker>(0, 1, 0, 512_MB);

  WarmupWorker(worker, args_vec[0]);

  double total_us_acc = 0.0;
  double sum_task_us_acc = 0.0;
  int iter = 0;
  for (auto _ : state) {
    const std::string prefix = "burst_" + std::to_string(iter++);
    const BurstTiming timing =
        RunBurstTimed(worker, args_vec, burst_size, prefix);
    total_us_acc += timing.total_us;
    sum_task_us_acc += timing.sum_task_us;
    state.SetIterationTime(timing.total_us / 1e6);
  }

  const double iters =
      static_cast<double>(std::max<int64_t>(1, state.iterations()));
  const double total_flops =
      static_cast<double>(burst_size) * 2.0 * dim * dim * dim;
  const double avg_total_us = total_us_acc / iters;
  const double avg_sum_task_us = sum_task_us_acc / iters;
  const double per_task_us = avg_total_us / static_cast<double>(burst_size);
  const double efficiency_pct =
      (avg_total_us > 0.0) ? (100.0 * avg_sum_task_us / avg_total_us) : 0.0;

  state.counters["GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["total_us"] = avg_total_us;
  state.counters["per_task_us"] = per_task_us;
  state.counters["pipeline_efficiency_pct"] = efficiency_pct;
  state.SetLabel("mode=" + mode);

  worker->Stop();
}

BENCHMARK(BM_Gemm_Burst)
    ->Args({128, 1})
    ->Args({128, 4})
    ->Args({128, 16})
    ->Args({512, 1})
    ->Args({512, 4})
    ->Args({512, 16})
    ->Args({1024, 1})
    ->Args({1024, 4})
    ->Args({1024, 16})
    ->Args({2048, 1})
    ->Args({2048, 4})
    ->Args({2048, 16})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
