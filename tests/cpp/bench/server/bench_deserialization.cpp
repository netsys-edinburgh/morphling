#include <arpa/inet.h>
#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../../external/protobuf/src/google/protobuf/wrappers.pb.h"

namespace {

constexpr size_t kHeaderSize = 16;

struct WireHeader {
  uint32_t payload_size = 0;
  uint32_t proto_size = 0;
  uint64_t tensor_size = 0;
};

struct SerializedCase {
  std::vector<uint8_t> wire;
  std::vector<size_t> matrix_sizes;
  const uint8_t* proto_ptr = nullptr;
  const uint8_t* tensor_ptr = nullptr;
  uint32_t proto_size = 0;
  uint64_t tensor_size = 0;

  explicit SerializedCase(int dim) {
    const size_t tensor_bytes =
        static_cast<size_t>(dim) * static_cast<size_t>(dim) * sizeof(float);
    const uint64_t first_size = static_cast<uint64_t>(tensor_bytes / 2);
    const uint64_t second_size =
        static_cast<uint64_t>(tensor_bytes - static_cast<size_t>(first_size));
    matrix_sizes = {static_cast<size_t>(first_size),
                    static_cast<size_t>(second_size)};

    std::string packed_sizes;
    packed_sizes.resize(sizeof(uint64_t) * 2);
    std::memcpy(packed_sizes.data(), &first_size, sizeof(uint64_t));
    std::memcpy(packed_sizes.data() + sizeof(uint64_t), &second_size,
                sizeof(uint64_t));

    google::protobuf::BytesValue proto_msg;
    proto_msg.set_value(packed_sizes);
    const std::string proto_bytes = proto_msg.SerializeAsString();

    proto_size = static_cast<uint32_t>(proto_bytes.size());
    tensor_size = static_cast<uint64_t>(tensor_bytes);
    const uint32_t payload_size = static_cast<uint32_t>(
        sizeof(uint32_t) + sizeof(uint64_t) + proto_size + tensor_size);

    wire.resize(kHeaderSize + static_cast<size_t>(proto_size) +
                static_cast<size_t>(tensor_size));

    uint32_t payload_size_network = htonl(payload_size);
    std::memcpy(wire.data(), &payload_size_network, sizeof(uint32_t));
    std::memcpy(wire.data() + 4, &proto_size, sizeof(uint32_t));
    std::memcpy(wire.data() + 8, &tensor_size, sizeof(uint64_t));

    proto_ptr = wire.data() + kHeaderSize;
    std::memcpy(wire.data() + kHeaderSize, proto_bytes.data(),
                proto_bytes.size());

    tensor_ptr = proto_ptr + proto_size;
    std::memset(const_cast<uint8_t*>(tensor_ptr), 0xA5,
                static_cast<size_t>(first_size));
    std::memset(const_cast<uint8_t*>(tensor_ptr + first_size), 0x5A,
                static_cast<size_t>(second_size));
  }
};

WireHeader ParseWireHeader(const uint8_t* data, size_t size) {
  if (data == nullptr || size < kHeaderSize) {
    throw std::runtime_error("wire header parse failed");
  }

  WireHeader header;
  std::memcpy(&header.payload_size, data, sizeof(uint32_t));
  std::memcpy(&header.proto_size, data + 4, sizeof(uint32_t));
  std::memcpy(&header.tensor_size, data + 8, sizeof(uint64_t));
  header.payload_size = ntohl(header.payload_size);
  return header;
}

std::array<size_t, 2> ParseTwoTensorSizes(const uint8_t* proto_ptr,
                                          uint32_t proto_size) {
  google::protobuf::BytesValue msg;
  if (!msg.ParseFromArray(proto_ptr, static_cast<int>(proto_size))) {
    throw std::runtime_error("protobuf ParseFromArray failed");
  }

  const std::string& value = msg.value();
  if (value.size() < sizeof(uint64_t) * 2) {
    throw std::runtime_error("protobuf payload too small for tensor sizes");
  }

  std::array<size_t, 2> sizes{};
  uint64_t first = 0;
  uint64_t second = 0;
  std::memcpy(&first, value.data(), sizeof(uint64_t));
  std::memcpy(&second, value.data() + sizeof(uint64_t), sizeof(uint64_t));
  sizes[0] = static_cast<size_t>(first);
  sizes[1] = static_cast<size_t>(second);
  return sizes;
}

size_t DeserializeLikeHotPath(const SerializedCase& test_case) {
  const WireHeader header =
      ParseWireHeader(test_case.wire.data(), test_case.wire.size());
  const size_t required = kHeaderSize + static_cast<size_t>(header.proto_size) +
                          static_cast<size_t>(header.tensor_size);
  if (required > test_case.wire.size()) {
    throw std::runtime_error("wire size validation failed");
  }

  const uint8_t* proto_ptr = test_case.wire.data() + kHeaderSize;
  const uint8_t* tensor_ptr = proto_ptr + header.proto_size;
  const auto sizes = ParseTwoTensorSizes(proto_ptr, header.proto_size);

  uintptr_t ptr_mix = 0;
  size_t offset = 0;
  for (size_t bytes : sizes) {
    const uint8_t* current = tensor_ptr + offset;
    ptr_mix ^= reinterpret_cast<uintptr_t>(current);
    offset += bytes;
  }

  benchmark::DoNotOptimize(ptr_mix);
  return offset;
}

void RecordRateCounters(benchmark::State& state, double bytes_per_message) {
  state.counters["bytes_per_second"] =
      benchmark::Counter(bytes_per_message, benchmark::Counter::kIsRate);
  state.counters["messages_per_second"] =
      benchmark::Counter(1.0, benchmark::Counter::kIsRate);
}

static void BM_DeserializeProto_Small(benchmark::State& state) {
  SerializedCase test_case(128);
  const double bytes_per_message = static_cast<double>(test_case.wire.size());

  for (auto _ : state) {
    const size_t parsed = DeserializeLikeHotPath(test_case);
    benchmark::DoNotOptimize(&parsed);
  }

  RecordRateCounters(state, bytes_per_message);
}
BENCHMARK(BM_DeserializeProto_Small);

static void BM_DeserializeProto_Medium(benchmark::State& state) {
  SerializedCase test_case(1024);
  const double bytes_per_message = static_cast<double>(test_case.wire.size());

  for (auto _ : state) {
    const size_t parsed = DeserializeLikeHotPath(test_case);
    benchmark::DoNotOptimize(&parsed);
  }

  RecordRateCounters(state, bytes_per_message);
}
BENCHMARK(BM_DeserializeProto_Medium);

static void BM_DeserializeProto_Large(benchmark::State& state) {
  SerializedCase test_case(4096);
  const double bytes_per_message = static_cast<double>(test_case.wire.size());

  for (auto _ : state) {
    const size_t parsed = DeserializeLikeHotPath(test_case);
    benchmark::DoNotOptimize(&parsed);
  }

  RecordRateCounters(state, bytes_per_message);
}
BENCHMARK(BM_DeserializeProto_Large)->MinTime(0.1);

static void BM_ParseFromArray_Isolated(benchmark::State& state) {
  SerializedCase test_case(1024);
  const double bytes_per_message = static_cast<double>(test_case.proto_size);

  for (auto _ : state) {
    google::protobuf::BytesValue msg;
    const bool parsed = msg.ParseFromArray(
        test_case.proto_ptr, static_cast<int>(test_case.proto_size));
    const auto value_size = msg.value().size();
    benchmark::DoNotOptimize(&parsed);
    benchmark::DoNotOptimize(&value_size);
  }

  RecordRateCounters(state, bytes_per_message);
  state.counters["proto_parse_ns"] = benchmark::Counter(0.0);
}
BENCHMARK(BM_ParseFromArray_Isolated);

static void BM_HeaderParse_Only(benchmark::State& state) {
  SerializedCase test_case(1024);
  const double bytes_per_message = static_cast<double>(kHeaderSize);

  for (auto _ : state) {
    const WireHeader header =
        ParseWireHeader(test_case.wire.data(), test_case.wire.size());
    benchmark::DoNotOptimize(&header.payload_size);
    benchmark::DoNotOptimize(&header.proto_size);
    benchmark::DoNotOptimize(&header.tensor_size);
  }

  RecordRateCounters(state, bytes_per_message);
}
BENCHMARK(BM_HeaderParse_Only);

static void BM_TensorPointerExtraction(benchmark::State& state) {
  SerializedCase test_case(1024);
  const std::array<size_t, 2> sizes =
      ParseTwoTensorSizes(test_case.proto_ptr, test_case.proto_size);
  const double bytes_per_message = static_cast<double>(test_case.tensor_size);

  for (auto _ : state) {
    const uint8_t* base = test_case.tensor_ptr;
    uintptr_t ptr_mix = 0;
    size_t offset = 0;

    for (size_t bytes : sizes) {
      const uint8_t* current = base + offset;
      ptr_mix ^= reinterpret_cast<uintptr_t>(current);
      offset += bytes;
      benchmark::DoNotOptimize(current);
    }

    benchmark::DoNotOptimize(ptr_mix);
    benchmark::DoNotOptimize(offset);
  }

  RecordRateCounters(state, bytes_per_message);
  state.counters["tensor_extract_ns"] = benchmark::Counter(0.0);
}
BENCHMARK(BM_TensorPointerExtraction);

}  // namespace
