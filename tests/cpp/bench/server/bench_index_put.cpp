#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

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

void RecordBytesPerSecond(benchmark::State& state, size_t total_bytes) {
  state.counters["bytes_per_second"] = benchmark::Counter(
      static_cast<double>(total_bytes), benchmark::Counter::kIsRate);
}

static void BM_IndexPut_SingleBlock(benchmark::State& state) {
  constexpr int kDim = 1024;
  constexpr int kBlockRows = 512;
  constexpr int kBlockCols = 1024;

  std::vector<float> src(static_cast<size_t>(kBlockRows) * kBlockCols, 1.0f);
  std::vector<float> dst(static_cast<size_t>(kDim) * kDim, 0.0f);

  const size_t total_bytes =
      static_cast<size_t>(kBlockRows) * kBlockCols * sizeof(float);

  for (auto _ : state) {
    (void)_;
    BenchIndexPutImpl(dst.data(), src.data(), 0, 0, kBlockRows, kBlockCols,
                      kDim, kDim);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  RecordBytesPerSecond(state, total_bytes);
}
BENCHMARK(BM_IndexPut_SingleBlock);

static void BM_IndexPut_MultiBlock(benchmark::State& state) {
  constexpr int kDim = 1024;

  const int num_blocks = static_cast<int>(state.range(0));
  const int block_rows = kDim / num_blocks;

  std::vector<float> src(static_cast<size_t>(block_rows) * kDim, 1.0f);
  std::vector<float> dst(static_cast<size_t>(kDim) * kDim, 0.0f);

  const size_t total_bytes = static_cast<size_t>(kDim) * kDim * sizeof(float);

  for (auto _ : state) {
    (void)_;
    for (int block = 0; block < num_blocks; ++block) {
      const int row_offset = block * block_rows;
      BenchIndexPutImpl(dst.data(), src.data(), row_offset, 0, block_rows, kDim,
                        kDim, kDim);
    }
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  RecordBytesPerSecond(state, total_bytes);
}
BENCHMARK(BM_IndexPut_MultiBlock)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

static void BM_IndexPut_VaryingDims(benchmark::State& state) {
  const int dim = static_cast<int>(state.range(0));
  const int block_rows = dim / 2;

  std::vector<float> src(static_cast<size_t>(block_rows) * dim, 1.0f);
  std::vector<float> dst(static_cast<size_t>(dim) * dim, 0.0f);

  const size_t total_bytes =
      static_cast<size_t>(block_rows) * dim * sizeof(float);

  for (auto _ : state) {
    (void)_;
    BenchIndexPutImpl(dst.data(), src.data(), 0, 0, block_rows, dim, dim, dim);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  RecordBytesPerSecond(state, total_bytes);
}
BENCHMARK(BM_IndexPut_VaryingDims)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Arg(8192);

static void BM_RawMemcpy_Baseline(benchmark::State& state) {
  const int dim = static_cast<int>(state.range(0));
  const int block_rows = dim / 2;

  std::vector<float> src(static_cast<size_t>(block_rows) * dim, 1.0f);
  std::vector<float> dst(static_cast<size_t>(dim) * dim, 0.0f);

  const size_t total_bytes =
      static_cast<size_t>(block_rows) * dim * sizeof(float);

  for (auto _ : state) {
    (void)_;
    std::memcpy(dst.data(), src.data(), total_bytes);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  RecordBytesPerSecond(state, total_bytes);
}
BENCHMARK(BM_RawMemcpy_Baseline)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Arg(8192);

}  // namespace
