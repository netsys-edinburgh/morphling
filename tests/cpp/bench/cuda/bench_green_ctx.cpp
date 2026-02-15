#include <benchmark/benchmark.h>
#include <cuda.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/types_and_defs.h"
#include "scheduler/gpu_worker.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

// ============================================================================
// Shared utilities
// ============================================================================

static void EnsureLoggerInit() {
  static std::once_flag flag;
  std::call_once(flag, []() { InitLogger(); });
}

static void EnsureDriverInit() {
  static std::once_flag flag;
  std::call_once(flag, []() { cuInit(0); });
}

static int GetSmCount() {
  EnsureDriverInit();
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  int sm_count = 0;
  cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                       dev);
  return sm_count;
}

// Pinned host memory RAII wrapper
struct PinnedBuffer {
  float* ptr = nullptr;
  size_t bytes = 0;

  PinnedBuffer() = default;

  explicit PinnedBuffer(size_t elems) : bytes(elems * sizeof(float)) {
    if (elems > 0) {
      cudaHostAlloc(reinterpret_cast<void**>(&ptr), bytes,
                    cudaHostAllocDefault);
      for (size_t i = 0; i < elems; i++) {
        ptr[i] = static_cast<float>(i % 1000) * 0.001f;
      }
    }
  }
  ~PinnedBuffer() {
    if (ptr) cudaFreeHost(ptr);
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
    o.ptr = nullptr;
    o.bytes = 0;
  }
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr) cudaFreeHost(ptr);
      ptr = o.ptr;
      bytes = o.bytes;
      o.ptr = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
};

static std::shared_ptr<GemmArgs> MakeNNGemmArgs(int m, int n, int k, float* a,
                                                float* b, float* c) {
  auto args = std::make_shared<GemmArgs>();
  args->group_size = 1;
  args->transa[0] = 'N';
  args->transb[0] = 'N';
  args->m[0] = m;
  args->n[0] = n;
  args->k[0] = k;
  args->alpha[0] = 1.0f;
  args->a[0] = a;
  args->lda[0] = m;
  args->b[0] = b;
  args->ldb[0] = k;
  args->beta[0] = 0.0f;
  args->c[0] = c;
  args->ldc[0] = m;
  return args;
}

// Run H2D + cublasSgemm + D2H on pre-allocated device buffers.
// Does NOT synchronize — caller must sync the stream.
static void RunGemmDirect(cublasHandle_t handle, cudaStream_t stream,
                          float* d_A, float* d_B, float* d_C,
                          const GemmArgs* args) {
  // H2D: copy A
  CUDA_MEMCPY_ASYNC_LOOP(args->transa[0], d_A, args->a[0], args->lda[0],
                         args->m[0], args->k[0], cudaMemcpyHostToDevice,
                         stream);
  // H2D: copy B
  CUDA_MEMCPY_ASYNC_LOOP(args->transb[0], d_B, args->b[0], args->ldb[0],
                         args->k[0], args->n[0], cudaMemcpyHostToDevice,
                         stream);

  cublasOperation_t transa = CUDA_TRANS_OP(args->transa[0]);
  cublasOperation_t transb = CUDA_TRANS_OP(args->transb[0]);

  CHECK_CUBLAS_ERROR(cublasSgemm_v2(handle, transa, transb, args->m[0],
                                    args->n[0], args->k[0], args->alpha, d_A,
                                    args->lda[0], d_B, args->ldb[0], args->beta,
                                    d_C, args->ldc[0]));

  // D2H: copy C
  CUDA_MEMCPY_ASYNC_LOOP('N', args->c[0], d_C, args->ldc[0], args->m[0],
                         args->n[0], cudaMemcpyDeviceToHost, stream);
}

// Helper: run one full green context lifecycle (split -> create -> destroy)
static void WarmupGreenCtxLifecycle(CUdevice dev, unsigned int min_sm) {
  CUdevResource sm = {};
  CHECK_CU_RESULT(
      cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM));
  unsigned int n = 0;
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      nullptr, &n, &sm, nullptr,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
  std::vector<CUdevResource> splits(n);
  CUdevResource rem = {};
  CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
      splits.data(), &n, &sm, &rem,
      CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
  CUdevResourceDesc desc = nullptr;
  CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc, &splits[0], 1));
  CUgreenCtx gc = nullptr;
  CHECK_CU_RESULT(
      cuGreenCtxCreate(&gc, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CUcontext ctx = nullptr;
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
  cuCtxDestroy(ctx);
  cuGreenCtxDestroy(gc);
}

// ============================================================================
// Group 1: Green Context Create/Destroy Overhead
// ============================================================================

// --- Full lifecycle fixture ---
class GreenCtxLifecycle : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    sm_count_ = GetSmCount();
    num_partitions_ = state.range(0);
    min_sm_ = static_cast<unsigned int>(sm_count_ / num_partitions_);
    if (min_sm_ == 0) {
      state.SkipWithMessage("Not enough SMs for requested partitions");
      return;
    }
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));
    WarmupGreenCtxLifecycle(dev_, min_sm_);
  }

 protected:
  CUdevice dev_ = 0;
  int sm_count_ = 0;
  int num_partitions_ = 1;
  unsigned int min_sm_ = 0;
};

BENCHMARK_DEFINE_F(GreenCtxLifecycle, FullLifecycle)
(benchmark::State& state) {
  for (auto _ : state) {
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));

    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));

    std::vector<CUdevResource> splits(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));

    CUdevResourceDesc desc = nullptr;
    CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc, &splits[0], 1));

    CUgreenCtx green_ctx = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&green_ctx, desc, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext cuda_ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&cuda_ctx, green_ctx));

    cuCtxDestroy(cuda_ctx);
    cuGreenCtxDestroy(green_ctx);
  }

  state.SetLabel(std::to_string(num_partitions_) + " partitions");
  state.counters["SMs_total"] = sm_count_;
  state.counters["SMs_per_partition"] = sm_count_ / num_partitions_;
}

BENCHMARK_REGISTER_F(GreenCtxLifecycle, FullLifecycle)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond);

// --- Split only fixture ---
class GreenCtxSplit : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    int sm_count = GetSmCount();
    int num_partitions = state.range(0);
    min_sm_ = static_cast<unsigned int>(sm_count / num_partitions);
    if (min_sm_ == 0) {
      state.SkipWithMessage("Not enough SMs for requested partitions");
      return;
    }
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Warmup: one split cycle
    CUdevResource sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int n = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &n, &sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));
    std::vector<CUdevResource> splits(n);
    CUdevResource rem = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits.data(), &n, &sm, &rem,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));
  }

 protected:
  CUdevice dev_ = 0;
  unsigned int min_sm_ = 0;
};

BENCHMARK_DEFINE_F(GreenCtxSplit, SplitOnly)(benchmark::State& state) {
  for (auto _ : state) {
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));

    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));

    std::vector<CUdevResource> splits(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_));
  }
}

BENCHMARK_REGISTER_F(GreenCtxSplit, SplitOnly)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond);

// --- Create only fixture (pre-computed split/desc, warmup create+destroy) ---
class GreenCtxCreate : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    int sm_count = GetSmCount();
    int num_partitions = state.range(0);
    unsigned int min_sm =
        static_cast<unsigned int>(sm_count / num_partitions);
    if (min_sm == 0) {
      state.SkipWithMessage("Not enough SMs for requested partitions");
      return;
    }
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Pre-compute split and descriptor
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    splits_.resize(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits_.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    desc_ = nullptr;
    CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc_, &splits_[0], 1));

    // Warmup: one create/destroy cycle
    CUgreenCtx gc = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
    cuCtxDestroy(ctx);
    cuGreenCtxDestroy(gc);
  }

 protected:
  CUdevice dev_ = 0;
  std::vector<CUdevResource> splits_;
  CUdevResourceDesc desc_ = nullptr;
};

BENCHMARK_DEFINE_F(GreenCtxCreate, CreateOnly)(benchmark::State& state) {
  for (auto _ : state) {
    CUgreenCtx green_ctx = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&green_ctx, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext cuda_ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&cuda_ctx, green_ctx));

    cuCtxDestroy(cuda_ctx);
    cuGreenCtxDestroy(green_ctx);
  }
}

BENCHMARK_REGISTER_F(GreenCtxCreate, CreateOnly)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond);

// --- Destroy only fixture (pre-computed desc, warmup, PauseTiming for create)
class GreenCtxDestroy : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    EnsureDriverInit();
    int sm_count = GetSmCount();
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Pre-compute split and descriptor (1 partition = all SMs)
    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev_, &device_sm, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int min_sm = static_cast<unsigned int>(sm_count);
    unsigned int nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    splits_.resize(nb_groups);
    CUdevResource remaining = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        splits_.data(), &nb_groups, &device_sm, &remaining,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm));
    desc_ = nullptr;
    CHECK_CU_RESULT(cuDevResourceGenerateDesc(&desc_, &splits_[0], 1));

    // Warmup: one create/destroy cycle
    CUgreenCtx gc = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
    cuCtxDestroy(ctx);
    cuGreenCtxDestroy(gc);
  }

 protected:
  CUdevice dev_ = 0;
  std::vector<CUdevResource> splits_;
  CUdevResourceDesc desc_ = nullptr;
};

BENCHMARK_DEFINE_F(GreenCtxDestroy, DestroyOnly)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    CUgreenCtx green_ctx = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&green_ctx, desc_, dev_,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext cuda_ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&cuda_ctx, green_ctx));
    state.ResumeTiming();

    cuCtxDestroy(cuda_ctx);
    cuGreenCtxDestroy(green_ctx);
  }
}

BENCHMARK_REGISTER_F(GreenCtxDestroy, DestroyOnly)
    ->Unit(benchmark::kMicrosecond);

// --- Regular context baseline fixture ---
class RegularCtx : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureDriverInit();
    CHECK_CU_RESULT(cuDeviceGet(&dev_, 0));

    // Warmup: one create/destroy cycle
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxCreate(&ctx, 0, dev_));
    cuCtxDestroy(ctx);
  }

 protected:
  CUdevice dev_ = 0;
};

BENCHMARK_DEFINE_F(RegularCtx, Baseline)(benchmark::State& state) {
  for (auto _ : state) {
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxCreate(&ctx, 0, dev_));
    cuCtxDestroy(ctx);
  }
}

BENCHMARK_REGISTER_F(RegularCtx, Baseline)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Group 2: SM Count vs GEMM Performance
// ============================================================================

// --- Single-partition GEMM perf fixture ---
class GreenCtxGemmPerf : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    num_partitions_ = state.range(0);
    dim_ = state.range(1);
    size_t elems = static_cast<size_t>(dim_) * dim_;

    h_A_ = PinnedBuffer(elems);
    h_B_ = PinnedBuffer(elems);
    h_C_ = PinnedBuffer(elems);

    worker_ = std::make_shared<XtGemmWorker>(
        0, num_partitions_, 0, 512_MB);
    args_ = MakeNNGemmArgs(dim_, dim_, dim_,
                           h_A_.ptr, h_B_.ptr, h_C_.ptr);

    // Warmup GEMM
    worker_->AddTask("warmup", [this]() { worker_->RunXtGemm(args_); });
    worker_->WaitTaskDone("warmup");
  }

  void TearDown(benchmark::State& state) override {
    worker_->Stop();
    worker_.reset();
    args_.reset();
    h_A_ = PinnedBuffer();
    h_B_ = PinnedBuffer();
    h_C_ = PinnedBuffer();
  }

 protected:
  int num_partitions_ = 1;
  int dim_ = 0;
  PinnedBuffer h_A_, h_B_, h_C_;
  std::shared_ptr<XtGemmWorker> worker_;
  std::shared_ptr<GemmArgs> args_;
};

BENCHMARK_DEFINE_F(GreenCtxGemmPerf, GemmPerf)
(benchmark::State& state) {
  int iter = 0;
  for (auto _ : state) {
    std::string tid = "iter_" + std::to_string(iter++);
    worker_->AddTask(tid, [this]() { worker_->RunXtGemm(args_); });
    worker_->WaitTaskDone(tid);
  }

  double flops = 2.0 * dim_ * dim_ * dim_;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["SMs_assigned"] = GetSmCount() / num_partitions_;
}

BENCHMARK_REGISTER_F(GreenCtxGemmPerf, GemmPerf)
    ->Args({1, 512})
    ->Args({1, 1024})
    ->Args({1, 2048})
    ->Args({1, 4096})
    ->Args({2, 512})
    ->Args({2, 1024})
    ->Args({2, 2048})
    ->Args({2, 4096})
    ->Args({4, 512})
    ->Args({4, 1024})
    ->Args({4, 2048})
    ->Args({4, 4096})
    ->Unit(benchmark::kMicrosecond);

// --- Aggregate scaling fixture ---
class GreenCtxGemmScaling : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    num_partitions_ = state.range(0);
    const size_t elems = static_cast<size_t>(kDim) * kDim;

    data_.clear();
    data_.reserve(num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
      auto d = std::make_unique<WorkerData>();
      d->h_A = PinnedBuffer(elems);
      d->h_B = PinnedBuffer(elems);
      d->h_C = PinnedBuffer(elems);
      d->worker = std::make_shared<XtGemmWorker>(
          0, num_partitions_, i, 512_MB);
      d->args = MakeNNGemmArgs(kDim, kDim, kDim,
                               d->h_A.ptr, d->h_B.ptr, d->h_C.ptr);
      data_.push_back(std::move(d));
    }

    // Warmup all workers
    for (int i = 0; i < num_partitions_; i++) {
      data_[i]->worker->AddTask(
          "warmup_" + std::to_string(i),
          [&d = data_[i]]() { d->worker->RunXtGemm(d->args); });
    }
    for (auto& d : data_) {
      d->worker->WaitTaskDone();
    }
  }

  void TearDown(benchmark::State& state) override {
    for (auto& d : data_) {
      d->worker->Stop();
    }
    data_.clear();
  }

 protected:
  static constexpr int kDim = 1024;

  struct WorkerData {
    PinnedBuffer h_A{0};
    PinnedBuffer h_B{0};
    PinnedBuffer h_C{0};
    std::shared_ptr<XtGemmWorker> worker;
    std::shared_ptr<GemmArgs> args;
  };

  int num_partitions_ = 1;
  std::vector<std::unique_ptr<WorkerData>> data_;
};

BENCHMARK_DEFINE_F(GreenCtxGemmScaling, GemmScaling)
(benchmark::State& state) {
  int iter = 0;
  for (auto _ : state) {
    for (int i = 0; i < num_partitions_; i++) {
      std::string tid =
          "iter" + std::to_string(iter) + "_w" + std::to_string(i);
      data_[i]->worker->AddTask(
          tid, [&d = data_[i]]() { d->worker->RunXtGemm(d->args); });
    }
    for (auto& d : data_) {
      d->worker->WaitTaskDone();
    }
    iter++;
  }

  double total_flops =
      static_cast<double>(num_partitions_) * 2.0 * kDim * kDim * kDim;
  state.counters["Aggregate_GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["Partitions"] = num_partitions_;
}

BENCHMARK_REGISTER_F(GreenCtxGemmScaling, GemmScaling)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Group 3: Non-Uniform SM Split Benchmarks
// ============================================================================

// Helper: split device SMs into the smallest possible groups (minCount=2).
// All experiments in Group 3 use this to build non-uniform allocations.
struct NonUniformSplit {
  CUdevice dev = 0;
  std::vector<CUdevResource> all_groups;
  CUdevResource remainder = {};
  unsigned int nb_groups = 0;
  unsigned int group_sm_count = 0;  // actual SMs per group (may be >2)

  // Returns false if the GPU doesn't support fine-grained splitting.
  bool Init() {
    EnsureDriverInit();
    CHECK_CU_RESULT(cuDeviceGet(&dev, 0));

    CUdevResource device_sm = {};
    CHECK_CU_RESULT(
        cuDeviceGetDevResource(dev, &device_sm, CU_DEV_RESOURCE_TYPE_SM));

    // Query how many groups we get with minCount=2
    nb_groups = 0;
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        nullptr, &nb_groups, &device_sm, nullptr,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));
    if (nb_groups == 0) return false;

    all_groups.resize(nb_groups);
    remainder = {};
    CHECK_CU_RESULT(cuDevSmResourceSplitByCount(
        all_groups.data(), &nb_groups, &device_sm, &remainder,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 2));

    group_sm_count = all_groups[0].sm.smCount;
    return true;
  }
};

// --- Experiment 1: MultiGreenCtxCoexist ---
// Create multiple green contexts simultaneously with different SM counts
// (geometric: 1, 2, 4, 8 base groups) and run concurrent GEMMs.

struct GreenCtxSlot {
  int num_groups = 0;
  int sm_count = 0;
  CUdevResourceDesc desc = nullptr;
  CUgreenCtx green_ctx = nullptr;
  CUcontext cuda_ctx = nullptr;
  CUstream cu_stream = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  PinnedBuffer h_A{0};
  PinnedBuffer h_B{0};
  PinnedBuffer h_C{0};
  std::shared_ptr<GemmArgs> args;
};

class MultiGreenCtxCoexist : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    dim_ = state.range(0);
    size_t elems = static_cast<size_t>(dim_) * dim_;
    size_t bytes = elems * sizeof(float);

    if (!split_.Init()) {
      state.SkipWithMessage("NonUniformSplit failed");
      return;
    }

    // Geometric allocation: slot i gets 2^i groups => need 1+2+4+8=15 groups
    static constexpr int kSlotGroups[] = {1, 2, 4, 8};
    static constexpr int kNumSlots = 4;
    static constexpr int kTotalGroupsNeeded = 15;

    if (split_.nb_groups < kTotalGroupsNeeded) {
      state.SkipWithMessage(
          "Not enough groups (" + std::to_string(split_.nb_groups) +
          ") for 15 required");
      return;
    }

    slots_.resize(kNumSlots);
    unsigned int offset = 0;
    for (int i = 0; i < kNumSlots; i++) {
      auto& s = slots_[i];
      s.num_groups = kSlotGroups[i];
      s.sm_count = s.num_groups * static_cast<int>(split_.group_sm_count);

      CHECK_CU_RESULT(cuDevResourceGenerateDesc(
          &s.desc, &split_.all_groups[offset], s.num_groups));
      offset += s.num_groups;

      CHECK_CU_RESULT(cuGreenCtxCreate(&s.green_ctx, s.desc, split_.dev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&s.cuda_ctx, s.green_ctx));

      // Create stream and cuBLAS within this green context
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      CHECK_CU_RESULT(cuGreenCtxStreamCreate(&s.cu_stream, s.green_ctx,
                                             CU_STREAM_NON_BLOCKING, 0));
      s.stream = s.cu_stream;

      CHECK_CUBLAS_ERROR(cublasCreate(&s.cublas_handle));
      CHECK_CUBLAS_ERROR(cublasSetStream(s.cublas_handle, s.stream));

      // Allocate device memory in this context
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_A), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_B), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_C), bytes));

      // Pinned host buffers per slot
      s.h_A = PinnedBuffer(elems);
      s.h_B = PinnedBuffer(elems);
      s.h_C = PinnedBuffer(elems);

      s.args = MakeNNGemmArgs(dim_, dim_, dim_,
                              s.h_A.ptr, s.h_B.ptr, s.h_C.ptr);
    }

    // Warmup: run one GEMM per slot
    for (auto& s : slots_) {
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  void TearDown(benchmark::State&) override {
    for (auto& s : slots_) {
      if (s.cublas_handle) cublasDestroy(s.cublas_handle);
      if (s.stream) {
        CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
        cudaStreamDestroy(s.stream);
      }
      if (s.d_A) cudaFree(s.d_A);
      if (s.d_B) cudaFree(s.d_B);
      if (s.d_C) cudaFree(s.d_C);
      if (s.cuda_ctx) cuCtxDestroy(s.cuda_ctx);
      if (s.green_ctx) cuGreenCtxDestroy(s.green_ctx);
    }
    slots_.clear();
  }

 protected:
  int dim_ = 0;
  NonUniformSplit split_;
  std::vector<GreenCtxSlot> slots_;
};

BENCHMARK_DEFINE_F(MultiGreenCtxCoexist, ConcurrentGemm)
(benchmark::State& state) {
  for (auto _ : state) {
    // Launch GEMM on all slots
    for (auto& s : slots_) {
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
    }
    // Sync all
    for (auto& s : slots_) {
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  double total_flops = 0;
  for (size_t i = 0; i < slots_.size(); i++) {
    double flops = 2.0 * dim_ * dim_ * dim_;
    total_flops += flops;
    state.counters["SMs_slot" + std::to_string(i)] = slots_[i].sm_count;
  }
  state.counters["Aggregate_GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["num_contexts"] = static_cast<double>(slots_.size());
}

BENCHMARK_REGISTER_F(MultiGreenCtxCoexist, ConcurrentGemm)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// Measure cuCtxSetCurrent overhead when round-robin switching between
// non-uniform green contexts (1, 2, 4, 8 groups of SMs).
BENCHMARK_DEFINE_F(MultiGreenCtxCoexist, CtxSwitchOverhead)
(benchmark::State& state) {
  int idx = 0;
  int n = static_cast<int>(slots_.size());
  for (auto _ : state) {
    CHECK_CU_RESULT(cuCtxSetCurrent(slots_[idx].cuda_ctx));
    idx = (idx + 1) % n;
  }
  state.counters["num_contexts"] = static_cast<double>(n);
  for (int i = 0; i < n; i++) {
    state.counters["SMs_slot" + std::to_string(i)] = slots_[i].sm_count;
  }
}

BENCHMARK_REGISTER_F(MultiGreenCtxCoexist, CtxSwitchOverhead)
    ->Arg(512)
    ->Unit(benchmark::kNanosecond);

// --- Experiment 2: GreenCtxImmutability ---
// Verify green context SM resources are immutable after creation.

// Sub-bench A: DestroyRecreate — alternates between small (1 group) and
// large (4 groups) descriptors. Measures destroy+recreate cycle cost.
class GreenCtxImmutability : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    if (!split_.Init()) {
      state.SkipWithMessage("NonUniformSplit failed");
      return;
    }
    if (split_.nb_groups < 4) {
      state.SkipWithMessage(
          "Not enough groups (" + std::to_string(split_.nb_groups) +
          ") for immutability test");
      return;
    }

    // Pre-build small (1 group) and large (4 groups) descriptors
    CHECK_CU_RESULT(
        cuDevResourceGenerateDesc(&small_desc_, &split_.all_groups[0], 1));
    CHECK_CU_RESULT(
        cuDevResourceGenerateDesc(&large_desc_, &split_.all_groups[0], 4));

    // Warmup: one create/destroy cycle with each
    for (auto desc : {small_desc_, large_desc_}) {
      CUgreenCtx gc = nullptr;
      CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc, split_.dev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CUcontext ctx = nullptr;
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));
      cuCtxDestroy(ctx);
      cuGreenCtxDestroy(gc);
    }
  }

 protected:
  NonUniformSplit split_;
  CUdevResourceDesc small_desc_ = nullptr;
  CUdevResourceDesc large_desc_ = nullptr;
};

BENCHMARK_DEFINE_F(GreenCtxImmutability, DestroyRecreate)
(benchmark::State& state) {
  bool use_small = true;
  for (auto _ : state) {
    CUdevResourceDesc desc = use_small ? small_desc_ : large_desc_;
    use_small = !use_small;

    CUgreenCtx gc = nullptr;
    CHECK_CU_RESULT(cuGreenCtxCreate(&gc, desc, split_.dev,
                                     CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext ctx = nullptr;
    CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));

    cuCtxDestroy(ctx);
    cuGreenCtxDestroy(gc);
  }

  state.counters["small_SMs"] =
      1 * static_cast<double>(split_.group_sm_count);
  state.counters["large_SMs"] =
      4 * static_cast<double>(split_.group_sm_count);
}

BENCHMARK_REGISTER_F(GreenCtxImmutability, DestroyRecreate)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(GreenCtxImmutability, VerifyBoundSMs)
(benchmark::State& state) {
  // Create a green ctx with 4 groups and verify SM count is stable
  CUgreenCtx gc = nullptr;
  CHECK_CU_RESULT(cuGreenCtxCreate(&gc, large_desc_, split_.dev,
                                   CU_GREEN_CTX_DEFAULT_STREAM));
  CUcontext ctx = nullptr;
  CHECK_CU_RESULT(cuCtxFromGreenCtx(&ctx, gc));

  unsigned int expected_sm =
      4 * static_cast<unsigned int>(split_.group_sm_count);
  int queries = 0;

  for (auto _ : state) {
    CUdevResource queried = {};
    CHECK_CU_RESULT(
        cuGreenCtxGetDevResource(gc, &queried, CU_DEV_RESOURCE_TYPE_SM));
    if (queried.sm.smCount != expected_sm) {
      state.SkipWithMessage(
          "SM count mismatch: expected " + std::to_string(expected_sm) +
          " got " + std::to_string(queried.sm.smCount));
      break;
    }
    queries++;
  }

  cuCtxDestroy(ctx);
  cuGreenCtxDestroy(gc);

  state.counters["expected_SMs"] = static_cast<double>(expected_sm);
  state.counters["queries_ok"] = queries;
}

BENCHMARK_REGISTER_F(GreenCtxImmutability, VerifyBoundSMs)
    ->Unit(benchmark::kNanosecond);

// --- Experiment 3: RoundRobinGreenCtxStreams ---
// Pre-create green contexts with geometric SM counts, round-robin GEMM tasks.

class RoundRobinGreenCtxStreams : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    EnsureLoggerInit();
    dim_ = state.range(0);
    total_tasks_ = state.range(1);
    size_t elems = static_cast<size_t>(dim_) * dim_;
    size_t bytes = elems * sizeof(float);

    if (!split_.Init()) {
      state.SkipWithMessage("NonUniformSplit failed");
      return;
    }

    // Geometric allocation: slot i gets 2^i groups.
    // Max slots where 2^0 + ... + 2^(N-1) = 2^N - 1 <= nb_groups
    num_slots_ = 0;
    unsigned int sum = 0;
    while (sum + (1u << num_slots_) <= split_.nb_groups) {
      sum += (1u << num_slots_);
      num_slots_++;
    }
    if (num_slots_ == 0) {
      state.SkipWithMessage("Not enough groups for any slot");
      return;
    }

    slots_.resize(num_slots_);
    unsigned int offset = 0;
    for (int i = 0; i < num_slots_; i++) {
      auto& s = slots_[i];
      s.num_groups = (1 << i);
      s.sm_count = s.num_groups * static_cast<int>(split_.group_sm_count);

      CHECK_CU_RESULT(cuDevResourceGenerateDesc(
          &s.desc, &split_.all_groups[offset], s.num_groups));
      offset += s.num_groups;

      CHECK_CU_RESULT(cuGreenCtxCreate(&s.green_ctx, s.desc, split_.dev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CU_RESULT(cuCtxFromGreenCtx(&s.cuda_ctx, s.green_ctx));

      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      CHECK_CU_RESULT(cuGreenCtxStreamCreate(&s.cu_stream, s.green_ctx,
                                             CU_STREAM_NON_BLOCKING, 0));
      s.stream = s.cu_stream;

      CHECK_CUBLAS_ERROR(cublasCreate(&s.cublas_handle));
      CHECK_CUBLAS_ERROR(cublasSetStream(s.cublas_handle, s.stream));

      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_A), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_B), bytes));
      CHECK_CUDA_ERROR(
          cudaMalloc(reinterpret_cast<void**>(&s.d_C), bytes));

      s.h_A = PinnedBuffer(elems);
      s.h_B = PinnedBuffer(elems);
      s.h_C = PinnedBuffer(elems);

      s.args = MakeNNGemmArgs(dim_, dim_, dim_,
                              s.h_A.ptr, s.h_B.ptr, s.h_C.ptr);
    }

    // Warmup
    for (auto& s : slots_) {
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  void TearDown(benchmark::State&) override {
    for (auto& s : slots_) {
      if (s.cublas_handle) cublasDestroy(s.cublas_handle);
      if (s.stream) {
        CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
        cudaStreamDestroy(s.stream);
      }
      if (s.d_A) cudaFree(s.d_A);
      if (s.d_B) cudaFree(s.d_B);
      if (s.d_C) cudaFree(s.d_C);
      if (s.cuda_ctx) cuCtxDestroy(s.cuda_ctx);
      if (s.green_ctx) cuGreenCtxDestroy(s.green_ctx);
    }
    slots_.clear();
  }

 protected:
  int dim_ = 0;
  int total_tasks_ = 0;
  int num_slots_ = 0;
  NonUniformSplit split_;
  std::vector<GreenCtxSlot> slots_;
};

BENCHMARK_DEFINE_F(RoundRobinGreenCtxStreams, RoundRobin)
(benchmark::State& state) {
  for (auto _ : state) {
    for (int t = 0; t < total_tasks_; t++) {
      int slot_idx = t % num_slots_;
      auto& s = slots_[slot_idx];

      // Sync this slot's stream before reusing it
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
      CHECK_CU_RESULT(cuCtxSetCurrent(s.cuda_ctx));
      RunGemmDirect(s.cublas_handle, s.stream, s.d_A, s.d_B, s.d_C,
                    s.args.get());
    }
    // Sync all at end of iteration
    for (auto& s : slots_) {
      CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    }
  }

  double total_flops = static_cast<double>(total_tasks_) * 2.0 *
                       dim_ * dim_ * dim_;
  state.counters["Aggregate_GFLOPS"] = benchmark::Counter(
      total_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.counters["tasks_per_iter"] = total_tasks_;
  state.counters["num_slots"] = num_slots_;
  for (int i = 0; i < num_slots_; i++) {
    state.counters["SMs_slot" + std::to_string(i)] = slots_[i].sm_count;
  }
}

BENCHMARK_REGISTER_F(RoundRobinGreenCtxStreams, RoundRobin)
    ->Args({512, 8})
    ->Args({512, 16})
    ->Args({1024, 8})
    ->Args({1024, 16})
    ->Args({2048, 8})
    ->Args({2048, 16})
    ->Unit(benchmark::kMicrosecond);
