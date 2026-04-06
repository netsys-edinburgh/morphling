#include <benchmark/benchmark.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "backend/scatter_gather_buffer.h"

namespace {

constexpr size_t k1KB = 1ULL << 10;
constexpr size_t k4KB = 4ULL << 10;
constexpr size_t k16KB = 16ULL << 10;
constexpr size_t k64KB = 64ULL << 10;
constexpr size_t k256KB = 256ULL << 10;
constexpr size_t k1MB = 1ULL << 20;
constexpr size_t k16MB = 16ULL << 20;

constexpr size_t kMultiSegmentBytes = 4ULL << 10;

void AddPayloadSizes(benchmark::internal::Benchmark* bench) {
  bench->Arg(static_cast<int64_t>(k1KB))
      ->Arg(static_cast<int64_t>(k4KB))
      ->Arg(static_cast<int64_t>(k16KB))
      ->Arg(static_cast<int64_t>(k64KB))
      ->Arg(static_cast<int64_t>(k256KB))
      ->Arg(static_cast<int64_t>(k1MB))
      ->Arg(static_cast<int64_t>(k16MB));
}

void AddSegmentCounts(benchmark::internal::Benchmark* bench) {
  bench->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(64);
}

void RecordThroughputCounters(benchmark::State& state,
                              size_t bytes_per_iteration,
                              size_t segments_per_iteration) {
  state.counters["bytes_per_second"] = benchmark::Counter(
      static_cast<double>(bytes_per_iteration), benchmark::Counter::kIsRate);
  state.counters["segments_per_second"] = benchmark::Counter(
      static_cast<double>(segments_per_iteration), benchmark::Counter::kIsRate);
}

void NoopCleanup(const void*, size_t, void*) {}

void SharedVectorCleanup(const void*, size_t, void* extra) {
  delete static_cast<std::shared_ptr<std::vector<char>>*>(extra);
}

int SendDataZeroCopyCompat(struct bufferevent* bev, const void* data,
                           size_t data_len,
                           void (*cleanup_cb)(const void*, size_t, void*),
                           void* cleanup_arg) {
  struct evbuffer* outbuf = bufferevent_get_output(bev);
  if (outbuf == nullptr) {
    return -1;
  }
  return evbuffer_add_reference(outbuf, data, data_len, cleanup_cb,
                                cleanup_arg);
}

struct LocalBevPair {
  event_base* base = nullptr;
  bufferevent* send_bev = nullptr;
  bufferevent* recv_bev = nullptr;

  LocalBevPair() {
    int fds[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
      return;
    }

    base = event_base_new();
    if (base == nullptr) {
      ::close(fds[0]);
      ::close(fds[1]);
      return;
    }

    send_bev = bufferevent_socket_new(base, fds[0], BEV_OPT_CLOSE_ON_FREE);
    recv_bev = bufferevent_socket_new(base, fds[1], BEV_OPT_CLOSE_ON_FREE);
    if (send_bev == nullptr || recv_bev == nullptr) {
      if (send_bev != nullptr) {
        bufferevent_free(send_bev);
        send_bev = nullptr;
      }
      if (recv_bev != nullptr) {
        bufferevent_free(recv_bev);
        recv_bev = nullptr;
      }
      event_base_free(base);
      base = nullptr;
      return;
    }
  }

  ~LocalBevPair() {
    if (send_bev != nullptr) {
      bufferevent_free(send_bev);
      send_bev = nullptr;
    }
    if (recv_bev != nullptr) {
      bufferevent_free(recv_bev);
      recv_bev = nullptr;
    }
    if (base != nullptr) {
      event_base_free(base);
      base = nullptr;
    }
  }

  bool valid() const {
    return base != nullptr && send_bev != nullptr && recv_bev != nullptr;
  }
};

static void BM_EvbufferAddReference(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  std::vector<char> data(size, 'x');

  for (auto _ : state) {
    (void)_;
    evbuffer* buf = evbuffer_new();
    if (buf == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }

    const int rc = evbuffer_add_reference(buf, data.data(), data.size(),
                                          NoopCleanup, nullptr);
    if (rc != 0) {
      evbuffer_free(buf);
      state.SkipWithMessage("evbuffer_add_reference failed");
      return;
    }

    benchmark::DoNotOptimize(evbuffer_get_length(buf));
    evbuffer_free(buf);
  }

  RecordThroughputCounters(state, size, 1);
}
BENCHMARK(BM_EvbufferAddReference)->Apply(AddPayloadSizes);

static void BM_EvbufferAdd_Copy(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  std::vector<char> data(size, 'y');

  for (auto _ : state) {
    (void)_;
    evbuffer* buf = evbuffer_new();
    if (buf == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }

    const int rc = evbuffer_add(buf, data.data(), data.size());
    if (rc != 0) {
      evbuffer_free(buf);
      state.SkipWithMessage("evbuffer_add failed");
      return;
    }

    benchmark::DoNotOptimize(evbuffer_get_length(buf));
    evbuffer_free(buf);
  }

  RecordThroughputCounters(state, size, 1);
}
BENCHMARK(BM_EvbufferAdd_Copy)->Apply(AddPayloadSizes);

static void BM_ScatterGather_SingleSegment(benchmark::State& state) {
  const size_t segment_size = static_cast<size_t>(state.range(0));
  std::vector<char> segment(segment_size, 's');

  for (auto _ : state) {
    (void)_;
    ScatterGatherBuffer sg;
    sg.AddReferenceSegment(segment.data(), segment.size());

    evbuffer* buf = evbuffer_new();
    if (buf == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }

    for (const auto& part : sg.GetSegments()) {
      const int rc = evbuffer_add_reference(buf, part.data, part.size,
                                            NoopCleanup, nullptr);
      if (rc != 0) {
        evbuffer_free(buf);
        state.SkipWithMessage(
            "evbuffer_add_reference failed for single segment");
        return;
      }
    }

    benchmark::DoNotOptimize(evbuffer_get_length(buf));
    evbuffer_free(buf);
  }

  RecordThroughputCounters(state, segment_size, 1);
}
BENCHMARK(BM_ScatterGather_SingleSegment)->Apply(AddPayloadSizes);

static void BM_ScatterGather_MultiSegment(benchmark::State& state) {
  const size_t segment_count = static_cast<size_t>(state.range(0));
  std::vector<std::vector<char>> segments(
      segment_count, std::vector<char>(kMultiSegmentBytes, 'm'));
  const size_t total_bytes = segment_count * kMultiSegmentBytes;

  for (auto _ : state) {
    (void)_;
    ScatterGatherBuffer sg;
    for (const auto& seg : segments) {
      sg.AddReferenceSegment(seg.data(), seg.size());
    }

    evbuffer* buf = evbuffer_new();
    if (buf == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }

    for (const auto& part : sg.GetSegments()) {
      const int rc = evbuffer_add_reference(buf, part.data, part.size,
                                            NoopCleanup, nullptr);
      if (rc != 0) {
        evbuffer_free(buf);
        state.SkipWithMessage(
            "evbuffer_add_reference failed for multi-segment");
        return;
      }
    }

    benchmark::DoNotOptimize(evbuffer_get_length(buf));
    evbuffer_free(buf);
  }

  RecordThroughputCounters(state, total_bytes, segment_count);
}
BENCHMARK(BM_ScatterGather_MultiSegment)->Apply(AddSegmentCounts);

static void BM_ZeroCopy_CleanupCallback(benchmark::State& state) {
  constexpr size_t kPayloadSize = k1KB;
  std::vector<char> template_data(kPayloadSize, 'c');

  for (auto _ : state) {
    (void)_;
    auto lifetime = std::make_shared<std::vector<char>>(template_data);

    evbuffer* buf = evbuffer_new();
    if (buf == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }

    auto* raw_ref = new std::shared_ptr<std::vector<char>>(lifetime);
    const int rc = evbuffer_add_reference(
        buf, lifetime->data(), lifetime->size(), SharedVectorCleanup, raw_ref);
    if (rc != 0) {
      delete raw_ref;
      evbuffer_free(buf);
      state.SkipWithMessage("evbuffer_add_reference failed for cleanup path");
      return;
    }

    evbuffer_free(buf);
    benchmark::DoNotOptimize(lifetime.use_count());
  }

  RecordThroughputCounters(state, kPayloadSize, 1);
}
BENCHMARK(BM_ZeroCopy_CleanupCallback);

static void BM_SendDataZeroCopy_FullPath(benchmark::State& state) {
  const size_t payload_size = static_cast<size_t>(state.range(0));
  std::vector<char> template_payload(payload_size, 'f');

  LocalBevPair io_pair;
  if (!io_pair.valid()) {
    state.SkipWithMessage("socketpair/bufferevent setup failed");
    return;
  }

  for (auto _ : state) {
    (void)_;
    auto lifetime = std::make_shared<std::vector<char>>(template_payload);
    auto* ref = new std::shared_ptr<std::vector<char>>(lifetime);

    const int rc =
        SendDataZeroCopyCompat(io_pair.send_bev, lifetime->data(),
                               lifetime->size(), SharedVectorCleanup, ref);
    if (rc != 0) {
      delete ref;
      state.SkipWithMessage("SendDataZeroCopyCompat failed");
      return;
    }

    evbuffer* outbuf = bufferevent_get_output(io_pair.send_bev);
    const size_t out_len = evbuffer_get_length(outbuf);
    benchmark::DoNotOptimize(static_cast<int64_t>(out_len));
    evbuffer_drain(outbuf, out_len);
    benchmark::DoNotOptimize(lifetime.use_count());
  }

  RecordThroughputCounters(state, payload_size, 1);
}
BENCHMARK(BM_SendDataZeroCopy_FullPath)->Apply(AddPayloadSizes);

}  // namespace
