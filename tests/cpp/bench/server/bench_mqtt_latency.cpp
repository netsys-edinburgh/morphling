#include <benchmark/benchmark.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace {

constexpr size_t k1KB = 1ULL << 10;
constexpr size_t k16KB = 16ULL << 10;
constexpr size_t k256KB = 256ULL << 10;
constexpr size_t k1MB = 1ULL << 20;

struct PubTask {
  std::string topic;
  std::string payload;
  int qos = 0;
  bool retain = false;
};

class PubTaskQueue {
 public:
  void Enqueue(PubTask task) {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (quit_) {
        return;
      }
      queue_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  bool Dequeue(PubTask& out) {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [this] { return quit_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    out = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      quit_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::deque<PubTask> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool quit_ = false;
};

void AddPayloadSizes(benchmark::internal::Benchmark* bench) {
  bench->Arg(static_cast<int64_t>(k1KB))
      ->Arg(static_cast<int64_t>(k16KB))
      ->Arg(static_cast<int64_t>(k256KB))
      ->Arg(static_cast<int64_t>(k1MB));
}

void SetThroughputCounters(benchmark::State& state, size_t bytes_per_message) {
  state.counters["messages_per_second"] =
      benchmark::Counter(1.0, benchmark::Counter::kIsRate);
  state.counters["bytes_per_second"] = benchmark::Counter(
      static_cast<double>(bytes_per_message), benchmark::Counter::kIsRate);
}

void DrainQueue(PubTaskQueue& queue) {
  PubTask task;
  while (queue.Dequeue(task)) {
    benchmark::DoNotOptimize(task.payload.data());
  }
}

static void BM_MqttPublish_Enqueue(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));
  const std::string topic = "bench/topic/enqueue";
  const std::string payload(payload_size, 'x');

  PubTaskQueue queue;
  std::thread drain([&queue] { DrainQueue(queue); });

  for (auto _ : state) {
    (void)_;
    queue.Enqueue(PubTask{topic, payload, 0, false});
  }

  queue.Stop();
  drain.join();

  SetThroughputCounters(state, payload_size);
}
BENCHMARK(BM_MqttPublish_Enqueue)->Apply(AddPayloadSizes);

static void BM_MqttSerializePublish(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));
  const std::string topic = "bench/topic/serialize";
  const std::string raw_payload(payload_size, 's');

  PubTaskQueue queue;
  std::thread drain([&queue] { DrainQueue(queue); });

  for (auto _ : state) {
    (void)_;
    std::string serialized;
    serialized.reserve(raw_payload.size() + 8);
    serialized.append("{\"p\":\"");
    serialized.append(raw_payload);
    serialized.append("\"}");

    queue.Enqueue(PubTask{topic, std::move(serialized), 0, false});
  }

  queue.Stop();
  drain.join();

  SetThroughputCounters(state, payload_size + 8);
}
BENCHMARK(BM_MqttSerializePublish)->Apply(AddPayloadSizes);

std::mutex g_contention_mu;
std::unique_ptr<PubTaskQueue> g_contention_queue;
std::unique_ptr<std::thread> g_contention_drain;
std::atomic<int> g_contention_active_threads{0};
std::atomic<bool> g_contention_ready{false};

static void BM_PubTaskQueue_Contention(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));
  const std::string topic = "bench/topic/contention";
  const std::string payload(payload_size, 'c');

  g_contention_active_threads.fetch_add(1, std::memory_order_acq_rel);

  if (state.thread_index() == 0) {
    {
      std::lock_guard<std::mutex> lk(g_contention_mu);
      g_contention_queue = std::make_unique<PubTaskQueue>();
      PubTaskQueue* queue_ptr = g_contention_queue.get();
      g_contention_drain = std::make_unique<std::thread>(
          [queue_ptr] { DrainQueue(*queue_ptr); });
    }
    g_contention_ready.store(true, std::memory_order_release);
  } else {
    while (!g_contention_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  PubTaskQueue* queue_ptr = nullptr;
  while (queue_ptr == nullptr) {
    {
      std::lock_guard<std::mutex> lk(g_contention_mu);
      queue_ptr = g_contention_queue.get();
    }
    if (queue_ptr == nullptr) {
      std::this_thread::yield();
    }
  }

  for (auto _ : state) {
    (void)_;
    queue_ptr->Enqueue(PubTask{topic, payload, 0, false});
  }

  g_contention_active_threads.fetch_sub(1, std::memory_order_acq_rel);

  if (state.thread_index() == 0) {
    while (g_contention_active_threads.load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }

    std::thread* drain_ptr = nullptr;
    {
      std::lock_guard<std::mutex> lk(g_contention_mu);
      queue_ptr = g_contention_queue.get();
      drain_ptr = g_contention_drain.get();
    }

    if (queue_ptr != nullptr) {
      queue_ptr->Stop();
    }
    if (drain_ptr != nullptr && drain_ptr->joinable()) {
      drain_ptr->join();
    }

    {
      std::lock_guard<std::mutex> lk(g_contention_mu);
      g_contention_drain.reset();
      g_contention_queue.reset();
    }
    g_contention_ready.store(false, std::memory_order_release);
  }

  SetThroughputCounters(state, payload_size);
}

BENCHMARK(BM_PubTaskQueue_Contention)
    ->Arg(static_cast<int64_t>(k1KB))
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);

static void BM_MqttPublish_ThreadDrain(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));
  const std::string topic = "bench/topic/drain";
  const std::string payload(payload_size, 'd');
  constexpr size_t kMaxBacklog = 128;

  PubTaskQueue queue;
  std::atomic<bool> producer_running{true};
  std::atomic<size_t> produced{0};
  std::atomic<size_t> consumed{0};

  std::thread producer([&] {
    while (producer_running.load(std::memory_order_acquire)) {
      const size_t backlog = produced.load(std::memory_order_relaxed) -
                             consumed.load(std::memory_order_relaxed);
      if (backlog >= kMaxBacklog) {
        std::this_thread::yield();
        continue;
      }

      queue.Enqueue(PubTask{topic, payload, 0, false});
      produced.fetch_add(1, std::memory_order_relaxed);
    }
  });

  for (auto _ : state) {
    (void)_;
    PubTask task;
    if (!queue.Dequeue(task)) {
      state.SkipWithMessage("Publish queue stopped unexpectedly");
      break;
    }
    consumed.fetch_add(1, std::memory_order_relaxed);
    benchmark::DoNotOptimize(task.payload.data());
  }

  producer_running.store(false, std::memory_order_release);
  queue.Stop();
  producer.join();

  SetThroughputCounters(state, payload_size);
}
BENCHMARK(BM_MqttPublish_ThreadDrain)->Apply(AddPayloadSizes);

static void BM_MqttRoundtrip_Localhost(benchmark::State& state) {
  state.SkipWithMessage(
      "No mosquitto broker - roundtrip test requires live broker");
}
BENCHMARK(BM_MqttRoundtrip_Localhost);

}  // namespace
