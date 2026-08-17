#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// ===========================================================================
// Cross-process shared-memory SM-target notification (per-worker).
//
// An external producer (scheduler) publishes, per proxy worker, how many SMs
// that worker may currently use. The proxy's GEMM executor reads its own
// target at each chunk boundary and switches to the matching pre-created
// green context (see gpu_worker.h). Transport is a fixed-layout POSIX shared
// memory segment; per-slot updates use a seqlock so a reader never observes a
// torn value without locking.
//
// Layout (all little-endian, natural alignment):
//   [ SmNotifyHeader (64 B) ][ SmNotifySlot (64 B) ] x num_slots
// The header is padded to 64 B so every slot starts on a 64-byte boundary,
// satisfying SmNotifySlot's alignas(64) and avoiding false sharing.
// ===========================================================================

inline constexpr uint32_t kSmNotifyMagic = 0x534D4743u;  // 'SMGC'
inline constexpr uint32_t kSmNotifyVersion = 1u;

struct SmNotifyHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t num_slots;
  uint32_t reserved[13];  // pad to 64 bytes
};
static_assert(sizeof(SmNotifyHeader) == 64, "SmNotifyHeader must be 64 bytes");

struct alignas(64) SmNotifySlot {
  uint32_t seq;           // seqlock counter: odd => write in progress
  int32_t gpu_id;         // slot identity: physical GPU index
  int32_t partition_idx;  // slot identity: green-ctx partition index
  int32_t target_sms;     // desired SM count for this worker
  uint64_t generation;    // observability: bumped on each publish
};
static_assert(sizeof(SmNotifySlot) == 64, "SmNotifySlot must be 64 bytes");

// Abstract SM-target source. Injected into the worker so the executor's
// switching logic is unit-testable without shared memory or a GPU.
class SmTargetSource {
 public:
  virtual ~SmTargetSource() = default;

  // Returns the desired SM count for this worker, or kNoChange when no target
  // is available (unattached, or a stable seqlock read could not be obtained).
  virtual int Read() = 0;

  static constexpr int kNoChange = -1;
};

// POSIX shared-memory consumer. Opens a segment read-only, validates the
// header, locates this worker's slot by (gpu_id, partition_idx), and returns
// its target_sms via a seqlock read.
class ShmSmTargetSource : public SmTargetSource {
 public:
  struct Options {
    std::string shm_name;  // e.g. "/morphling_sm_ctl"
    int gpu_id = 0;
    int partition_idx = 0;
    bool require =
        false;  // hard-fail (throw) if the segment is missing/invalid
  };

  explicit ShmSmTargetSource(const Options& opts);
  ~ShmSmTargetSource() override;

  ShmSmTargetSource(const ShmSmTargetSource&) = delete;
  ShmSmTargetSource& operator=(const ShmSmTargetSource&) = delete;

  // Seqlock read of this worker's target_sms, or kNoChange if unattached.
  int Read() override;

  // True when the segment was opened, validated, and this worker's slot found.
  bool Attached() const { return slot_ != nullptr; }

 private:
  // Opens + validates + locates the slot. Throws std::runtime_error when
  // opts_.require and any step fails; otherwise leaves the source detached.
  void Attach();
  void Detach() noexcept;

  Options opts_;
  int fd_ = -1;
  void* map_ = nullptr;
  size_t map_size_ = 0;
  const SmNotifySlot* slot_ = nullptr;  // cached pointer into the mapping
};
