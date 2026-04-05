#include <benchmark/benchmark.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "bench_common_utils.h"
#include "scheduler/worker_base.h"

namespace {

std::string MakeTaskId(const char* prefix, uint64_t id) {
  return std::string(prefix) + "_" + std::to_string(id);
}

static void BM_AddTask_SingleProducer(benchmark::State& state) {
  WorkerBase worker;
  worker.Start();

  uint64_t task_id = 0;
  for (auto _ : state) {
    (void)_;
    worker.AddTask(MakeTaskId("sp", task_id++), [] {});
  }

  worker.WaitTaskDone();
  worker.Stop();

  state.counters["tasks_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_AddTask_SingleProducer);

std::mutex g_multi_mu;
std::unique_ptr<WorkerBase> g_multi_worker;
std::atomic<uint64_t> g_multi_task_id{0};
std::atomic<int> g_multi_active_threads{0};
std::atomic<bool> g_multi_ready{false};

static void BM_AddTask_MultiProducer(benchmark::State& state) {
  g_multi_active_threads.fetch_add(1, std::memory_order_acq_rel);

  if (state.thread_index() == 0) {
    auto worker = std::make_unique<WorkerBase>();
    worker->Start();

    {
      std::lock_guard<std::mutex> lk(g_multi_mu);
      g_multi_worker = std::move(worker);
      g_multi_task_id.store(0, std::memory_order_relaxed);
    }
    g_multi_ready.store(true, std::memory_order_release);
  } else {
    while (!g_multi_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  WorkerBase* worker = nullptr;
  while (worker == nullptr) {
    {
      std::lock_guard<std::mutex> lk(g_multi_mu);
      worker = g_multi_worker.get();
    }
    if (worker == nullptr) {
      std::this_thread::yield();
    }
  }

  for (auto _ : state) {
    (void)_;
    const uint64_t id = g_multi_task_id.fetch_add(1, std::memory_order_relaxed);
    worker->AddTask(MakeTaskId("mp", id), [] {});
  }

  g_multi_active_threads.fetch_sub(1, std::memory_order_acq_rel);

  if (state.thread_index() == 0) {
    while (g_multi_active_threads.load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }

    WorkerBase* local_worker = nullptr;
    {
      std::lock_guard<std::mutex> lk(g_multi_mu);
      local_worker = g_multi_worker.get();
    }

    if (local_worker != nullptr) {
      local_worker->WaitTaskDone();
      local_worker->Stop();
    }

    {
      std::lock_guard<std::mutex> lk(g_multi_mu);
      g_multi_worker.reset();
    }
    g_multi_ready.store(false, std::memory_order_release);
  }

  state.counters["tasks_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_AddTask_MultiProducer)
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(static_cast<int>(GetHardwareConcurrency()));

static void BM_TaskRoundtrip_Latency(benchmark::State& state) {
  WorkerBase worker;
  worker.Start();

  uint64_t task_id = 0;
  for (auto _ : state) {
    (void)_;
    auto handle = worker.AddTask(MakeTaskId("rt", task_id++), [] {});
    handle->Wait();
  }

  worker.Stop();

  state.counters["tasks_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_TaskRoundtrip_Latency);

static void BM_TaskRoundtrip_UnderLoad(benchmark::State& state) {
  WorkerBase worker;
  worker.Start();

  const int queue_depth = static_cast<int>(state.range(0));
  uint64_t task_id = 0;

  for (auto _ : state) {
    (void)_;

    std::mutex gate_mu;
    std::condition_variable gate_cv;
    bool gate_open = false;

    worker.AddTask(MakeTaskId("gate", task_id++), [&] {
      std::unique_lock<std::mutex> lk(gate_mu);
      gate_cv.wait(lk, [&] { return gate_open; });
    });

    for (int i = 0; i < queue_depth; ++i) {
      worker.AddTask(MakeTaskId("bg", task_id++), [] {});
    }

    auto handle = worker.AddTask(MakeTaskId("ul", task_id++), [] {});

    {
      std::lock_guard<std::mutex> lk(gate_mu);
      gate_open = true;
    }
    gate_cv.notify_one();

    handle->Wait();
  }

  worker.WaitTaskDone();
  worker.Stop();

  state.counters["tasks_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_TaskRoundtrip_UnderLoad)->Arg(0)->Arg(10)->Arg(100)->Arg(1000);

static void BM_WaitTaskDone_Latency(benchmark::State& state) {
  WorkerBase worker;
  worker.Start();

  uint64_t task_id = 0;
  for (auto _ : state) {
    (void)_;
    const auto id = MakeTaskId("wait", task_id++);
    worker.AddTask(id, [] {});
    worker.WaitTaskDone(id);
  }

  worker.Stop();

  state.counters["tasks_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_WaitTaskDone_Latency);

static void BM_TaskCount_AtomicRead(benchmark::State& state) {
  WorkerBase worker;

  for (auto _ : state) {
    (void)_;
    benchmark::DoNotOptimize(worker.GetTaskCount());
  }

  state.counters["tasks_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_TaskCount_AtomicRead);

}  // namespace
