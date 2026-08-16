#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "core/logger.h"
#include "scheduler/gpu_worker.h"
#include "scheduler/sm_notify_source.h"

static std::shared_ptr<GemmArgs> MakeGemmArgs(char transa, char transb, int m,
                                              int n, int k, float alpha,
                                              const float* a, int lda,
                                              const float* b, int ldb,
                                              float beta, float* c, int ldc) {
  auto args = std::make_shared<GemmArgs>();
  args->transa = transa;
  args->transb = transb;
  args->m = m;
  args->n = n;
  args->k = k;
  args->alpha = alpha;
  args->a = a;
  args->lda = lda;
  args->b = b;
  args->ldb = ldb;
  args->beta = beta;
  args->c = c;
  args->ldc = ldc;
  return args;
}

static torch::Tensor ReferenceGemm(char transa, char transb, int m, int n,
                                   int k, float alpha, const float* a_data,
                                   int lda, const float* b_data, int ldb,
                                   float beta, const float* c_data, int ldc) {
  torch::Tensor opA;
  if (transa == 'N' || transa == 'n') {
    opA = torch::from_blob(const_cast<float*>(a_data), {k, lda}, torch::kFloat)
              .slice(1, 0, m)
              .t()
              .contiguous();
  } else {
    opA = torch::from_blob(const_cast<float*>(a_data), {m, lda}, torch::kFloat)
              .slice(1, 0, k)
              .contiguous();
  }
  torch::Tensor opB;
  if (transb == 'N' || transb == 'n') {
    opB = torch::from_blob(const_cast<float*>(b_data), {n, ldb}, torch::kFloat)
              .slice(1, 0, k)
              .t()
              .contiguous();
  } else {
    opB = torch::from_blob(const_cast<float*>(b_data), {k, ldb}, torch::kFloat)
              .slice(1, 0, n)
              .contiguous();
  }
  auto C_orig =
      torch::from_blob(const_cast<float*>(c_data), {n, ldc}, torch::kFloat)
          .slice(1, 0, m)
          .t()
          .contiguous();
  return alpha * torch::mm(opA, opB) + beta * C_orig;
}

// Scripted SM-target source: returns each value in `seq` on successive Read()
// calls, holding the final value once exhausted.
class FakeSource : public SmTargetSource {
 public:
  explicit FakeSource(std::vector<int> seq) : seq_(std::move(seq)) {}
  int Read() override {
    int v = seq_.empty() ? kNoChange : seq_[idx_];
    if (idx_ + 1 < seq_.size()) ++idx_;
    return v;
  }

 private:
  std::vector<int> seq_;
  size_t idx_ = 0;
};

class ChunkedGemmTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { InitLogger(); }

  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) GTEST_SKIP() << "No CUDA devices available";
  }

  struct Buffers {
    float *h_A, *h_B, *h_C;
    torch::Tensor ref;
    int lda, ldb, ldc, m, n, k;
    char ta, tb;
    float alpha, beta;
  };

  Buffers MakeProblem(int m, int n, int k, char ta = 'N', char tb = 'N',
                      float alpha = 1.0f, float beta = 0.0f) {
    Buffers b;
    b.m = m;
    b.n = n;
    b.k = k;
    b.ta = ta;
    b.tb = tb;
    b.alpha = alpha;
    b.beta = beta;
    b.lda = (ta == 'N' || ta == 'n') ? m : k;
    b.ldb = (tb == 'N' || tb == 'n') ? k : n;
    b.ldc = m;
    size_t a_elems =
        (ta == 'N' || ta == 'n') ? (size_t)b.lda * k : (size_t)b.lda * m;
    size_t b_elems =
        (tb == 'N' || tb == 'n') ? (size_t)b.ldb * n : (size_t)b.ldb * k;
    size_t c_elems = (size_t)b.ldc * n;
    cudaHostAlloc(reinterpret_cast<void**>(&b.h_A), a_elems * sizeof(float),
                  cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&b.h_B), b_elems * sizeof(float),
                  cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&b.h_C), c_elems * sizeof(float),
                  cudaHostAllocDefault);
    auto tA = torch::rand({(long)a_elems});
    auto tB = torch::rand({(long)b_elems});
    auto tC = torch::rand({(long)c_elems});
    memcpy(b.h_A, tA.data_ptr<float>(), a_elems * sizeof(float));
    memcpy(b.h_B, tB.data_ptr<float>(), b_elems * sizeof(float));
    memcpy(b.h_C, tC.data_ptr<float>(), c_elems * sizeof(float));
    b.ref = ReferenceGemm(ta, tb, m, n, k, alpha, b.h_A, b.lda, b.h_B, b.ldb,
                          beta, b.h_C, b.ldc);
    return b;
  }

  void Verify(const Buffers& b) {
    auto result = torch::from_blob(b.h_C, {b.n, b.ldc}, torch::kFloat)
                      .slice(1, 0, b.m)
                      .t()
                      .contiguous();
    EXPECT_TRUE(torch::allclose(result, b.ref, 1e-3, 1e-3))
        << "max diff=" << (result - b.ref).abs().max().item<float>();
  }

  void Free(Buffers& b) {
    cudaFreeHost(b.h_A);
    cudaFreeHost(b.h_B);
    cudaFreeHost(b.h_C);
  }

  void RunOnWorker(XtGemmWorker& w, const Buffers& b) {
    auto args = MakeGemmArgs(b.ta, b.tb, b.m, b.n, b.k, b.alpha, b.h_A, b.lda,
                             b.h_B, b.ldb, b.beta, b.h_C, b.ldc);
    w.AddTask("gemm", [&w, args]() { w.RunChunkedGemm(args); });
    w.WaitTaskDone("gemm");
  }
};

TEST_F(ChunkedGemmTest, SingleShot_MatchesReference) {
  auto w = std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
  auto b = MakeProblem(256, 512, 128);
  RunOnWorker(*w, b);
  Verify(b);
  Free(b);
  w->Stop();
}

TEST_F(ChunkedGemmTest, SingleShot_AlphaBetaTranspose) {
  auto w = std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
  auto b = MakeProblem(137, 251, 73, 'T', 'N', 2.5f, 0.7f);
  RunOnWorker(*w, b);
  Verify(b);
  Free(b);
  w->Stop();
}

TEST_F(ChunkedGemmTest, ChunkedNoSwitch_MatchesReference) {
  auto w = std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
  w->AddTask("warmup", [] {});
  w->WaitTaskDone("warmup");
  auto avail = w->GetAvailableSmCounts();
  ASSERT_FALSE(avail.empty());

  XtGemmWorker::DynConfig cfg;
  cfg.enabled = true;
  cfg.chunk_target_us = 1;  // tiny target => min-sized chunks => many chunks
  cfg.min_chunk_cols = 32;
  cfg.min_gemm_chunk_threshold = 0;
  w->SetDynConfig(cfg);
  w->SetSmTargetSource(std::make_unique<FakeSource>(
      std::vector<int>{avail.back()}));  // stay at max => no switch

  auto b = MakeProblem(256, 512, 128);
  RunOnWorker(*w, b);
  Verify(b);
  EXPECT_EQ(w->GetActiveSmCount(), avail.back());
  Free(b);
  w->Stop();
}

TEST_F(ChunkedGemmTest, ChunkedWithSwitch_ResumeMatchesReference) {
  auto w = std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
  w->AddTask("warmup", [] {});
  w->WaitTaskDone("warmup");
  auto avail = w->GetAvailableSmCounts();
  ASSERT_GE(avail.size(), 2u) << "need >=2 SM slots to exercise a switch";
  const int big = avail.back();
  const int small = avail.front();

  XtGemmWorker::DynConfig cfg;
  cfg.enabled = true;
  cfg.chunk_target_us = 1;
  cfg.min_chunk_cols = 32;
  cfg.min_gemm_chunk_threshold = 0;
  w->SetDynConfig(cfg);
  // First read (initial ctx) => big (no-op, worker starts at max);
  // subsequent reads => small => switch mid-GEMM.
  w->SetSmTargetSource(
      std::make_unique<FakeSource>(std::vector<int>{big, small}));

  auto b = MakeProblem(256, 512, 128);
  RunOnWorker(*w, b);
  Verify(b);
  EXPECT_EQ(w->GetActiveSmCount(), small)
      << "expected a mid-GEMM switch to the smaller SM partition";
  Free(b);
  w->Stop();
}

TEST_F(ChunkedGemmTest, ChunkedWithSwitch_TransposeB) {
  auto w = std::make_shared<XtGemmWorker>(0, 1, 0, 256_MB);
  w->AddTask("warmup", [] {});
  w->WaitTaskDone("warmup");
  auto avail = w->GetAvailableSmCounts();
  ASSERT_GE(avail.size(), 2u);

  XtGemmWorker::DynConfig cfg;
  cfg.enabled = true;
  cfg.chunk_target_us = 1;
  cfg.min_chunk_cols = 16;
  cfg.min_gemm_chunk_threshold = 0;
  w->SetDynConfig(cfg);
  w->SetSmTargetSource(std::make_unique<FakeSource>(
      std::vector<int>{avail.back(), avail.front()}));

  auto b = MakeProblem(128, 256, 64, 'N', 'T');
  RunOnWorker(*w, b);
  Verify(b);
  Free(b);
  w->Stop();
}
