#include <arpa/inet.h>
#include <benchmark/benchmark.h>
#include <event2/buffer.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../../external/protobuf/src/google/protobuf/port_def.inc"
#include "../../../../external/protobuf/src/google/protobuf/port_undef.inc"
#include "../../../../external/protobuf/src/google/protobuf/wrappers.pb.h"
#include "backend/scatter_gather_buffer.h"

namespace {

struct MessageFormat {
  static constexpr size_t PAYLOAD_SIZE_OFFSET = 0;
  static constexpr size_t PAYLOAD_SIZE_LENGTH = 4;
  static constexpr size_t PROTO_SIZE_OFFSET = 4;
  static constexpr size_t PROTO_SIZE_LENGTH = 4;
  static constexpr size_t TENSOR_SIZE_OFFSET = 8;
  static constexpr size_t TENSOR_SIZE_LENGTH = 8;
  static constexpr size_t HEADER_SIZE = 16;
};

struct WireHeader {
  uint32_t payload_size = 0;
  uint32_t proto_size = 0;
  uint64_t tensor_size = 0;
};

struct PipelineCase {
  int dim = 0;
  int partitions = 0;
  std::vector<int> partition_rows;
  std::vector<size_t> partition_bytes;
  std::vector<float> tensor;
  std::string proto_bytes;
  std::vector<uint8_t> wire;
  uint32_t proto_size = 0;
  uint64_t tensor_size = 0;

  PipelineCase(int matrix_dim, int partition_count)
      : dim(matrix_dim), partitions(partition_count) {
    if (dim <= 0 || partitions <= 0) {
      throw std::runtime_error("invalid PipelineCase dimensions");
    }

    partition_rows.resize(static_cast<size_t>(partitions), dim / partitions);
    const int remainder = dim % partitions;
    for (int i = 0; i < remainder; ++i) {
      partition_rows[static_cast<size_t>(i)] += 1;
    }

    partition_bytes.reserve(static_cast<size_t>(partitions));
    for (int rows : partition_rows) {
      partition_bytes.push_back(static_cast<size_t>(rows) *
                                static_cast<size_t>(dim) * sizeof(float));
    }

    tensor.resize(static_cast<size_t>(dim) * static_cast<size_t>(dim));
    for (size_t i = 0; i < tensor.size(); ++i) {
      tensor[i] = static_cast<float>(i % 97U) * 0.125f;
    }

    proto_bytes = BuildProtoBytes(partition_bytes);
    proto_size = static_cast<uint32_t>(proto_bytes.size());
    tensor_size = static_cast<uint64_t>(tensor.size() *
                                        static_cast<size_t>(sizeof(float)));

    const uint32_t payload_size = static_cast<uint32_t>(
        MessageFormat::PROTO_SIZE_LENGTH + MessageFormat::TENSOR_SIZE_LENGTH +
        proto_size + tensor_size);

    wire.resize(MessageFormat::HEADER_SIZE + static_cast<size_t>(proto_size) +
                static_cast<size_t>(tensor_size));
    WriteHeader(wire.data(), payload_size, proto_size, tensor_size);
    std::memcpy(wire.data() + MessageFormat::HEADER_SIZE, proto_bytes.data(),
                static_cast<size_t>(proto_size));
    std::memcpy(wire.data() + MessageFormat::HEADER_SIZE +
                    static_cast<size_t>(proto_size),
                tensor.data(), static_cast<size_t>(tensor_size));
  }

  static std::string BuildProtoBytes(const std::vector<size_t>& sizes) {
    std::string packed;
    packed.resize(sizeof(uint64_t) * sizes.size());

    for (size_t i = 0; i < sizes.size(); ++i) {
      const uint64_t value = static_cast<uint64_t>(sizes[i]);
      std::memcpy(packed.data() + i * sizeof(uint64_t), &value,
                  sizeof(uint64_t));
    }

    google::protobuf::BytesValue msg;
    msg.set_value(packed);
    return msg.SerializeAsString();
  }

  static void WriteHeader(uint8_t* dst, uint32_t payload_size,
                          uint32_t proto_size, uint64_t tensor_size) {
    uint32_t payload_n = htonl(payload_size);
    std::memcpy(dst + MessageFormat::PAYLOAD_SIZE_OFFSET, &payload_n,
                MessageFormat::PAYLOAD_SIZE_LENGTH);
    std::memcpy(dst + MessageFormat::PROTO_SIZE_OFFSET, &proto_size,
                MessageFormat::PROTO_SIZE_LENGTH);
    std::memcpy(dst + MessageFormat::TENSOR_SIZE_OFFSET, &tensor_size,
                MessageFormat::TENSOR_SIZE_LENGTH);
  }
};

struct DeserializedPayload {
  WireHeader header;
  std::vector<size_t> partition_bytes;
  std::vector<const uint8_t*> tensor_ptrs;
};

struct ZeroCopyPacket {
  ScatterGatherBuffer sg;
  std::vector<uint8_t> header_proto;
  size_t total_bytes = 0;
};

struct LocalSocketPair {
  int send_fd = -1;
  int recv_fd = -1;

  LocalSocketPair() {
    int fds[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0) {
      send_fd = fds[0];
      recv_fd = fds[1];
    }
  }

  ~LocalSocketPair() {
    if (send_fd >= 0) {
      ::close(send_fd);
      send_fd = -1;
    }
    if (recv_fd >= 0) {
      ::close(recv_fd);
      recv_fd = -1;
    }
  }

  bool valid() const { return send_fd >= 0 && recv_fd >= 0; }
};

void AddMatrixDims(benchmark::internal::Benchmark* bench) {
  bench->Arg(128)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);
}

void AddPartitionCounts(benchmark::internal::Benchmark* bench) {
  bench->Arg(1)->Arg(2)->Arg(4)->Arg(8);
}

void RecordRateCounters(benchmark::State& state, double bytes_per_request) {
  state.counters["requests_per_second"] =
      benchmark::Counter(1.0, benchmark::Counter::kIsRate);
  state.counters["bytes_per_second"] =
      benchmark::Counter(bytes_per_request, benchmark::Counter::kIsRate);
  state.counters["stage_breakdown_ns"] = benchmark::Counter(0.0);
}

WireHeader ParseWireHeader(const uint8_t* data, size_t size) {
  if (data == nullptr || size < MessageFormat::HEADER_SIZE) {
    throw std::runtime_error("header parse failed");
  }

  WireHeader header;
  std::memcpy(&header.payload_size, data + MessageFormat::PAYLOAD_SIZE_OFFSET,
              MessageFormat::PAYLOAD_SIZE_LENGTH);
  std::memcpy(&header.proto_size, data + MessageFormat::PROTO_SIZE_OFFSET,
              MessageFormat::PROTO_SIZE_LENGTH);
  std::memcpy(&header.tensor_size, data + MessageFormat::TENSOR_SIZE_OFFSET,
              MessageFormat::TENSOR_SIZE_LENGTH);
  header.payload_size = ntohl(header.payload_size);
  return header;
}

std::vector<size_t> ParsePartitionSizes(const uint8_t* proto_ptr,
                                        uint32_t proto_size) {
  google::protobuf::BytesValue msg;
  if (!msg.ParseFromArray(proto_ptr, static_cast<int>(proto_size))) {
    throw std::runtime_error("protobuf parse failed");
  }

  const std::string& value = msg.value();
  if (value.size() % sizeof(uint64_t) != 0) {
    throw std::runtime_error("partition size payload malformed");
  }

  const size_t count = value.size() / sizeof(uint64_t);
  std::vector<size_t> sizes(count, 0);
  for (size_t i = 0; i < count; ++i) {
    uint64_t bytes = 0;
    std::memcpy(&bytes, value.data() + i * sizeof(uint64_t), sizeof(uint64_t));
    sizes[i] = static_cast<size_t>(bytes);
  }

  return sizes;
}

DeserializedPayload DeserializePipelinePayload(const uint8_t* wire,
                                               size_t wire_size,
                                               size_t expected_partitions) {
  DeserializedPayload out;
  out.header = ParseWireHeader(wire, wire_size);

  const size_t required = MessageFormat::HEADER_SIZE +
                          static_cast<size_t>(out.header.proto_size) +
                          static_cast<size_t>(out.header.tensor_size);
  if (required > wire_size) {
    throw std::runtime_error("wire size validation failed");
  }

  const uint8_t* proto_ptr = wire + MessageFormat::HEADER_SIZE;
  const uint8_t* tensor_ptr = proto_ptr + out.header.proto_size;

  out.partition_bytes = ParsePartitionSizes(proto_ptr, out.header.proto_size);
  if (expected_partitions != 0 &&
      out.partition_bytes.size() != expected_partitions) {
    throw std::runtime_error("unexpected partition count");
  }

  size_t offset = 0;
  out.tensor_ptrs.reserve(out.partition_bytes.size());
  for (size_t bytes : out.partition_bytes) {
    if (offset + bytes > static_cast<size_t>(out.header.tensor_size)) {
      throw std::runtime_error("tensor partition exceeds payload");
    }
    out.tensor_ptrs.push_back(tensor_ptr + offset);
    offset += bytes;
  }

  return out;
}

void BenchIndexPutImpl(float* dst, const float* src, int dst_row_offset,
                       int src_row_offset, int num_rows, int num_cols,
                       int dst_stride_cols, int src_stride_cols) {
  const size_t row_bytes = static_cast<size_t>(num_cols) * sizeof(float);
  for (int row = 0; row < num_rows; ++row) {
    const float* src_row =
        src + static_cast<size_t>(src_row_offset + row) * src_stride_cols;
    float* dst_row =
        dst + static_cast<size_t>(dst_row_offset + row) * dst_stride_cols;
    std::memcpy(dst_row, src_row, row_bytes);
  }
}

void IndexPutFromPartitions(const DeserializedPayload& payload,
                            const std::vector<int>& partition_rows, int dim,
                            float* dst) {
  int row_offset = 0;
  for (size_t i = 0; i < partition_rows.size(); ++i) {
    const float* src = reinterpret_cast<const float*>(payload.tensor_ptrs[i]);
    const int rows = partition_rows[i];
    BenchIndexPutImpl(dst, src, row_offset, 0, rows, dim, dim, dim);
    row_offset += rows;
  }
}

std::vector<std::pair<const void*, size_t>> BuildTensorRefs(
    const float* tensor, const std::vector<size_t>& partition_bytes) {
  std::vector<std::pair<const void*, size_t>> refs;
  refs.reserve(partition_bytes.size());

  const uint8_t* base = reinterpret_cast<const uint8_t*>(tensor);
  size_t offset = 0;
  for (size_t bytes : partition_bytes) {
    refs.push_back({base + offset, bytes});
    offset += bytes;
  }
  return refs;
}

ZeroCopyPacket SerializeZeroCopyLike(
    const std::string& proto_bytes,
    const std::vector<std::pair<const void*, size_t>>& tensor_refs) {
  ZeroCopyPacket packet;

  const uint32_t proto_size = static_cast<uint32_t>(proto_bytes.size());
  uint64_t tensor_size = 0;
  for (const auto& ref : tensor_refs) {
    tensor_size += static_cast<uint64_t>(ref.second);
  }

  const uint32_t payload_size = static_cast<uint32_t>(
      MessageFormat::PROTO_SIZE_LENGTH + MessageFormat::TENSOR_SIZE_LENGTH +
      proto_size + tensor_size);

  packet.header_proto.resize(MessageFormat::HEADER_SIZE +
                             static_cast<size_t>(proto_size));
  PipelineCase::WriteHeader(packet.header_proto.data(), payload_size,
                            proto_size, tensor_size);
  std::memcpy(packet.header_proto.data() + MessageFormat::HEADER_SIZE,
              proto_bytes.data(), static_cast<size_t>(proto_size));

  packet.sg.AddReferenceSegment(packet.header_proto.data(),
                                packet.header_proto.size());
  for (const auto& ref : tensor_refs) {
    if (ref.first != nullptr && ref.second > 0) {
      packet.sg.AddReferenceSegment(ref.first, ref.second);
    }
  }

  packet.total_bytes =
      packet.header_proto.size() + static_cast<size_t>(tensor_size);
  return packet;
}

void NoopCleanup(const void*, size_t, void*) {}

void AddScatterGatherToEvbuffer(const ScatterGatherBuffer& sg, evbuffer* out) {
  for (const auto& seg : sg.GetSegments()) {
    if (evbuffer_add_reference(out, seg.data, seg.size, NoopCleanup, nullptr) !=
        0) {
      throw std::runtime_error("evbuffer_add_reference failed");
    }
  }
}

size_t TransferOverSocketpair(const ScatterGatherBuffer& sg,
                              const LocalSocketPair& sockets,
                              std::vector<uint8_t>& recv_storage) {
  const size_t total_bytes = sg.GetTotalSize();
  recv_storage.resize(total_bytes);

  size_t recv_offset = 0;
  for (const auto& seg : sg.GetSegments()) {
    const uint8_t* send_ptr = static_cast<const uint8_t*>(seg.data);
    size_t remaining = seg.size;

    while (remaining > 0) {
      const ssize_t sent =
          ::send(sockets.send_fd, send_ptr, remaining, MSG_DONTWAIT);
      if (sent > 0) {
        send_ptr += sent;
        remaining -= static_cast<size_t>(sent);
      } else if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      } else {
        throw std::runtime_error("socket send failed");
      }

      while (recv_offset < total_bytes) {
        const ssize_t recv_now =
            ::recv(sockets.recv_fd, recv_storage.data() + recv_offset,
                   total_bytes - recv_offset, MSG_DONTWAIT);
        if (recv_now > 0) {
          recv_offset += static_cast<size_t>(recv_now);
          continue;
        }
        if (recv_now < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
          break;
        }
        if (recv_now == 0) {
          throw std::runtime_error("socket closed unexpectedly");
        }
        throw std::runtime_error("socket recv failed");
      }
    }
  }

  while (recv_offset < total_bytes) {
    const ssize_t recv_now =
        ::recv(sockets.recv_fd, recv_storage.data() + recv_offset,
               total_bytes - recv_offset, 0);
    if (recv_now <= 0) {
      throw std::runtime_error("socket recv tail failed");
    }
    recv_offset += static_cast<size_t>(recv_now);
  }

  return recv_offset;
}

static void BM_Pipeline_HeaderParse(benchmark::State& state) {
  PipelineCase test_case(static_cast<int>(state.range(0)), 2);

  for (auto _ : state) {
    (void)_;
    const WireHeader header =
        ParseWireHeader(test_case.wire.data(), test_case.wire.size());
    benchmark::DoNotOptimize(static_cast<int64_t>(header.payload_size));
    benchmark::DoNotOptimize(static_cast<int64_t>(header.proto_size));
    benchmark::DoNotOptimize(static_cast<int64_t>(header.tensor_size));
  }

  RecordRateCounters(state, static_cast<double>(MessageFormat::HEADER_SIZE));
}
BENCHMARK(BM_Pipeline_HeaderParse)->Apply(AddMatrixDims);

static void BM_Pipeline_Deserialize(benchmark::State& state) {
  PipelineCase test_case(static_cast<int>(state.range(0)), 2);
  const double bytes_per_request = static_cast<double>(test_case.wire.size());

  for (auto _ : state) {
    (void)_;
    const DeserializedPayload payload =
        DeserializePipelinePayload(test_case.wire.data(), test_case.wire.size(),
                                   static_cast<size_t>(test_case.partitions));
    uintptr_t ptr_mix = 0;
    for (const uint8_t* ptr : payload.tensor_ptrs) {
      ptr_mix ^= reinterpret_cast<uintptr_t>(ptr);
    }
    benchmark::DoNotOptimize(ptr_mix);
    benchmark::DoNotOptimize(payload.partition_bytes.size());
  }

  RecordRateCounters(state, bytes_per_request);
}
BENCHMARK(BM_Pipeline_Deserialize)->Apply(AddMatrixDims);

static void BM_Pipeline_IndexPut(benchmark::State& state) {
  const int dim = static_cast<int>(state.range(0));
  PipelineCase test_case(dim, 2);
  std::vector<float> dst(static_cast<size_t>(dim) * static_cast<size_t>(dim),
                         0.0f);
  const double bytes_per_request = static_cast<double>(test_case.wire.size()) +
                                   static_cast<double>(test_case.tensor_size);

  for (auto _ : state) {
    (void)_;
    const DeserializedPayload payload =
        DeserializePipelinePayload(test_case.wire.data(), test_case.wire.size(),
                                   static_cast<size_t>(test_case.partitions));
    IndexPutFromPartitions(payload, test_case.partition_rows, dim, dst.data());
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  RecordRateCounters(state, bytes_per_request);
}
BENCHMARK(BM_Pipeline_IndexPut)->Apply(AddMatrixDims);

static void BM_Pipeline_SerializeZeroCopy(benchmark::State& state) {
  PipelineCase test_case(static_cast<int>(state.range(0)), 2);
  const auto tensor_refs =
      BuildTensorRefs(test_case.tensor.data(), test_case.partition_bytes);
  const double bytes_per_request =
      static_cast<double>(MessageFormat::HEADER_SIZE + test_case.proto_size) +
      static_cast<double>(test_case.tensor_size);

  for (auto _ : state) {
    (void)_;
    ZeroCopyPacket packet =
        SerializeZeroCopyLike(test_case.proto_bytes, tensor_refs);
    benchmark::DoNotOptimize(packet.total_bytes);
    benchmark::DoNotOptimize(packet.sg.GetSegments().size());
  }

  RecordRateCounters(state, bytes_per_request);
}
BENCHMARK(BM_Pipeline_SerializeZeroCopy)->Apply(AddMatrixDims);

static void BM_Pipeline_ScatterGatherSend(benchmark::State& state) {
  PipelineCase test_case(static_cast<int>(state.range(0)), 2);
  const auto tensor_refs =
      BuildTensorRefs(test_case.tensor.data(), test_case.partition_bytes);
  const double bytes_per_request =
      static_cast<double>(MessageFormat::HEADER_SIZE + test_case.proto_size) +
      static_cast<double>(test_case.tensor_size);

  for (auto _ : state) {
    (void)_;
    ZeroCopyPacket packet =
        SerializeZeroCopyLike(test_case.proto_bytes, tensor_refs);

    evbuffer* out = evbuffer_new();
    if (out == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }

    try {
      AddScatterGatherToEvbuffer(packet.sg, out);
    } catch (const std::exception&) {
      evbuffer_free(out);
      state.SkipWithMessage("evbuffer_add_reference failed");
      return;
    }

    benchmark::DoNotOptimize(evbuffer_get_length(out));
    evbuffer_free(out);
  }

  RecordRateCounters(state, bytes_per_request);
}
BENCHMARK(BM_Pipeline_ScatterGatherSend)->Apply(AddMatrixDims);

static void BM_Pipeline_FullRoundtrip(benchmark::State& state) {
  const int dim = static_cast<int>(state.range(0));
  PipelineCase request_case(dim, 2);
  std::vector<float> output(static_cast<size_t>(dim) * static_cast<size_t>(dim),
                            0.0f);
  const auto output_refs =
      BuildTensorRefs(output.data(), request_case.partition_bytes);
  LocalSocketPair sockets;
  if (!sockets.valid()) {
    state.SkipWithMessage("socketpair failed");
    return;
  }

  std::vector<uint8_t> recv_storage;
  const double bytes_per_request =
      static_cast<double>(request_case.wire.size()) +
      static_cast<double>(MessageFormat::HEADER_SIZE + request_case.proto_size +
                          request_case.tensor_size);

  for (auto _ : state) {
    (void)_;
    const DeserializedPayload request = DeserializePipelinePayload(
        request_case.wire.data(), request_case.wire.size(),
        static_cast<size_t>(request_case.partitions));
    IndexPutFromPartitions(request, request_case.partition_rows, dim,
                           output.data());

    ZeroCopyPacket response =
        SerializeZeroCopyLike(request_case.proto_bytes, output_refs);

    evbuffer* out = evbuffer_new();
    if (out == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }
    try {
      AddScatterGatherToEvbuffer(response.sg, out);
    } catch (const std::exception&) {
      evbuffer_free(out);
      state.SkipWithMessage("evbuffer_add_reference failed");
      return;
    }
    benchmark::DoNotOptimize(evbuffer_get_length(out));
    evbuffer_free(out);

    const size_t transferred =
        TransferOverSocketpair(response.sg, sockets, recv_storage);
    benchmark::DoNotOptimize(static_cast<int64_t>(transferred));

    const WireHeader response_header =
        ParseWireHeader(recv_storage.data(), recv_storage.size());
    benchmark::DoNotOptimize(
        static_cast<int64_t>(response_header.payload_size));
    benchmark::DoNotOptimize(static_cast<int64_t>(response_header.proto_size));
    benchmark::DoNotOptimize(static_cast<int64_t>(response_header.tensor_size));
    benchmark::ClobberMemory();
  }

  RecordRateCounters(state, bytes_per_request);
}
BENCHMARK(BM_Pipeline_FullRoundtrip)->Apply(AddMatrixDims);

static void BM_Pipeline_VaryingPartitions(benchmark::State& state) {
  constexpr int kDim = 1024;
  const int partitions = static_cast<int>(state.range(0));

  PipelineCase request_case(kDim, partitions);
  std::vector<float> output(
      static_cast<size_t>(kDim) * static_cast<size_t>(kDim), 0.0f);
  const auto output_refs =
      BuildTensorRefs(output.data(), request_case.partition_bytes);
  LocalSocketPair sockets;
  if (!sockets.valid()) {
    state.SkipWithMessage("socketpair failed");
    return;
  }

  std::vector<uint8_t> recv_storage;
  const double bytes_per_request =
      static_cast<double>(request_case.wire.size()) +
      static_cast<double>(MessageFormat::HEADER_SIZE + request_case.proto_size +
                          request_case.tensor_size);

  for (auto _ : state) {
    (void)_;
    const DeserializedPayload request = DeserializePipelinePayload(
        request_case.wire.data(), request_case.wire.size(),
        static_cast<size_t>(request_case.partitions));
    IndexPutFromPartitions(request, request_case.partition_rows, kDim,
                           output.data());

    ZeroCopyPacket response =
        SerializeZeroCopyLike(request_case.proto_bytes, output_refs);

    evbuffer* out = evbuffer_new();
    if (out == nullptr) {
      state.SkipWithMessage("evbuffer_new failed");
      return;
    }
    try {
      AddScatterGatherToEvbuffer(response.sg, out);
    } catch (const std::exception&) {
      evbuffer_free(out);
      state.SkipWithMessage("evbuffer_add_reference failed");
      return;
    }
    benchmark::DoNotOptimize(evbuffer_get_length(out));
    evbuffer_free(out);

    const size_t transferred =
        TransferOverSocketpair(response.sg, sockets, recv_storage);
    benchmark::DoNotOptimize(static_cast<int64_t>(transferred));
  }

  RecordRateCounters(state, bytes_per_request);
}
BENCHMARK(BM_Pipeline_VaryingPartitions)->Apply(AddPartitionCounts);

}  // namespace
