// Issue #49: regression lock for the multi-file consistency check in
// CheckpointHandle::ReadCheckpoint.
//
// The pre-fix code stat()ed only archer_param_0 then compared its size against
// `pin_mem_size`, which is the sum across ALL tensors regardless of file_id.
// A genuine two-file checkpoint therefore tripped a false-positive
// "Pin memory size != file size" LOG_WARN every time it loaded.
//
// The fix rebuilds the expected size per-file_id from tensor_index_, stats
// each param file once, and compares against the on-disk total. This test
// asserts the new behavior on a synthetic two-file checkpoint built directly
// against ArcherTensorIndex (so we don't need a multi-file writer to exercise
// the multi-file READER path).
//
// Log capture uses base::Logger::setOutput. OutputFunc is a raw function
// pointer (typedef void (*)(const char*, int)), so capture happens via a
// file-scope buffer rather than a capturing lambda. TearDown swaps in a
// no-op sink because muduo does not expose its default sink as a public
// symbol; restoring the default would dangle pointers in any later test
// running in the same binary.

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "checkpoint/archer_tensor_handle.h"
#include "checkpoint/archer_tensor_index.h"
#include "checkpoint/checkpoint_handle.h"
#include "core/logger.h"
#include "core/types_and_defs.h"
#include "muduo_base/logging.h"

namespace fs = std::filesystem;

namespace {

// ---- log capture --------------------------------------------------------

std::string g_capture_buf;  // appended to by CaptureLog while a test runs

void CaptureLog(const char* msg, int len) {
  g_capture_buf.append(msg, static_cast<size_t>(len));
}

void DiscardLog(const char* /*msg*/, int /*len*/) {}

class CheckpointMultiFileTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // CachingAllocator::Allocate path needs MORPHLING_PIN_SIZE before the
    // CheckpointHandle ctor runs InitCachingAllocator(PIN_SHM). 64 MB is
    // enough for the few-KB synthetic tensors we read back.
    setenv("MORPHLING_PIN_SIZE", "67108864", 0);
    InitLogger();
  }

  void SetUp() override {
    g_capture_buf.clear();
    base::Logger::setOutput(&CaptureLog);

    // Tests share a process-wide tmp dir slot, but each test gets its own
    // subdirectory to keep archer_index / archer_param_* writes isolated.
    char tmpl[] = "/tmp/morphling_ck_mf_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    ASSERT_NE(dir, nullptr) << "mkdtemp failed";
    tmp_dir_ = fs::path(dir);
  }

  void TearDown() override {
    base::Logger::setOutput(&DiscardLog);
    // Intentionally do NOT remove tmp_dir_: leaked handles still hold open
    // file descriptors into archer_param_* via the busy-looping AIO
    // background thread. fs::remove_all() works (Linux unlinks open files)
    // but emits noisy warnings; safer to let /tmp cleanup reap the dir.
  }

  // Build param_meta_map.json with three tensors. Sizes are in bytes.
  // The file_offset values must agree with the byte layout we write into
  // archer_param_<file_id> so pread sees the right region.
  void WriteParamMetaMap(
      const std::vector<std::tuple<std::string, uint32_t, size_t, size_t>>&
          entries) {
    std::ofstream ofs(tmp_dir_ / PARAM_META_FILE);
    ASSERT_TRUE(ofs.good());
    ofs << "{";
    for (size_t i = 0; i < entries.size(); ++i) {
      const auto& [name, id, size, file_offset] = entries[i];
      if (i > 0) ofs << ",";
      ofs << "\"" << name << "\":{\"id\":" << id << ",\"size\":" << size
          << ",\"file_offset\":" << file_offset << "}";
    }
    ofs << "}";
  }

  // Write `bytes` random non-zero bytes to archer_param_<file_id>. The
  // reader trips LOG_FATAL on all-zero blocks, so we must never write a
  // zero-filled region in the range any tensor will pread.
  void WriteParamFile(uint32_t file_id, size_t bytes) {
    auto p = tmp_dir_ /
             (std::string(ARCHER_PARAM_NAME) + "_" + std::to_string(file_id));
    std::ofstream ofs(p, std::ios::binary);
    ASSERT_TRUE(ofs.good());
    std::vector<char> buf(bytes);
    // Avoid std::rand zero-bit runs: write a simple non-zero pattern.
    for (size_t i = 0; i < bytes; ++i) {
      buf[i] = static_cast<char>((i % 251) + 1);  // never 0
    }
    ofs.write(buf.data(), static_cast<std::streamsize>(bytes));
  }

  // Hand-build archer_index with one TensorStorageMeta per tensor.
  // The reader only consults file_id, offset, size; shape/options are
  // serialized for round-trip but not consulted by the consistency check.
  void WriteArcherIndex(const std::vector<TensorStorageMeta>& metas) {
    ArcherTensorIndex index;
    for (const auto& m : metas) {
      index[m.id] = m;
    }
    auto idx_path = (tmp_dir_ / std::string(ARCHER_IHDEX_NAME)).string();
    index.Serialize(idx_path.c_str());
  }

  static TensorStorageMeta MakeMeta(uint32_t id, uint32_t file_id,
                                    int64_t offset, size_t size,
                                    int64_t first_dim) {
    TensorStorageMeta m;
    m.id = id;
    m.file_id = file_id;
    m.offset = offset;
    m.size = size;
    m.shape = {first_dim,
               static_cast<int64_t>(size / first_dim / sizeof(float))};
    m.options = torch::TensorOptions().dtype(torch::kFloat32);
    return m;
  }

  fs::path tmp_dir_;
  std::vector<CheckpointHandle*> leaked_handles_;
};

// ---- legacy single-file case --------------------------------------------

TEST_F(CheckpointMultiFileTest, LegacySingleFile_NoFalsePositiveWarning) {
  // Two tensors, both in file 0, contiguous offsets.
  constexpr size_t kT0Bytes = 4 * 16 * sizeof(float);  // 256 B
  constexpr size_t kT1Bytes = 8 * 8 * sizeof(float);   // 256 B

  WriteParamMetaMap({
      {"t0", 0, kT0Bytes, 0},
      {"t1", 1, kT1Bytes, kT0Bytes},
  });
  WriteArcherIndex({
      MakeMeta(/*id=*/0, /*file_id=*/0, /*offset=*/0, kT0Bytes, /*dim=*/4),
      MakeMeta(/*id=*/1, /*file_id=*/0, /*offset=*/kT0Bytes, kT1Bytes,
               /*dim=*/8),
  });
  WriteParamFile(/*file_id=*/0, kT0Bytes + kT1Bytes);

  auto* handle = new CheckpointHandle(tmp_dir_);
  ASSERT_NO_FATAL_FAILURE(handle->ReadCheckpoint());

  EXPECT_EQ(g_capture_buf.find("Pin memory size"), std::string::npos)
      << "Single-file checkpoint must not trip the consistency check.\n"
      << "Captured log: " << g_capture_buf;

  auto shm_map = handle->GetParamShmMap();
  EXPECT_EQ(shm_map.size(), 2u);
  EXPECT_NE(shm_map["t0"].ptr, nullptr);
  EXPECT_NE(shm_map["t1"].ptr, nullptr);
  leaked_handles_.push_back(handle);
}

// ---- multi-file case (the bug) ------------------------------------------

TEST_F(CheckpointMultiFileTest, MultiFile_NoFalsePositiveWarning) {
  // 3 tensors: t0 (file 0) + t1 (file 0) + t2 (file 1).
  // The pre-fix code stat()ed only archer_param_0 (size kT0+kT1) and
  // compared it against pin_mem_size (sum of ALL three) -> warning.
  constexpr size_t kT0Bytes = 4 * 16 * sizeof(float);
  constexpr size_t kT1Bytes = 8 * 8 * sizeof(float);
  constexpr size_t kT2Bytes = 4 * 32 * sizeof(float);

  WriteParamMetaMap({
      {"t0", 0, kT0Bytes, 0},
      {"t1", 1, kT1Bytes, kT0Bytes},
      {"t2", 2, kT2Bytes, 0},
  });
  WriteArcherIndex({
      MakeMeta(/*id=*/0, /*file_id=*/0, /*offset=*/0, kT0Bytes, /*dim=*/4),
      MakeMeta(/*id=*/1, /*file_id=*/0, /*offset=*/kT0Bytes, kT1Bytes,
               /*dim=*/8),
      MakeMeta(/*id=*/2, /*file_id=*/1, /*offset=*/0, kT2Bytes, /*dim=*/4),
  });
  WriteParamFile(/*file_id=*/0, kT0Bytes + kT1Bytes);
  WriteParamFile(/*file_id=*/1, kT2Bytes);

  auto* handle = new CheckpointHandle(tmp_dir_);
  ASSERT_NO_FATAL_FAILURE(handle->ReadCheckpoint());

  EXPECT_EQ(g_capture_buf.find("Pin memory size"), std::string::npos)
      << "Multi-file checkpoint tripped the pre-fix false-positive warning.\n"
      << "Captured log: " << g_capture_buf;
  EXPECT_EQ(g_capture_buf.find("< expected"), std::string::npos)
      << "Per-file size check fired unexpectedly.\n"
      << "Captured log: " << g_capture_buf;

  auto shm_map = handle->GetParamShmMap();
  EXPECT_EQ(shm_map.size(), 3u);
  EXPECT_NE(shm_map["t0"].ptr, nullptr);
  EXPECT_NE(shm_map["t1"].ptr, nullptr);
  EXPECT_NE(shm_map["t2"].ptr, nullptr);
  leaked_handles_.push_back(handle);
}

}  // namespace
