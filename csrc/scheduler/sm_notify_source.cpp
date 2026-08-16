#include "scheduler/sm_notify_source.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

ShmSmTargetSource::ShmSmTargetSource(const Options& opts) : opts_(opts) {
  Attach();
}

ShmSmTargetSource::~ShmSmTargetSource() { Detach(); }

void ShmSmTargetSource::Attach() {
  auto fail = [&](const std::string& why) {
    Detach();
    if (opts_.require) {
      throw std::runtime_error("ShmSmTargetSource: " + why +
                               " (shm=" + opts_.shm_name + ")");
    }
    std::fprintf(stderr,
                 "[ShmSmTargetSource] WARN: %s (shm=%s); dynamic SM switching "
                 "disabled\n",
                 why.c_str(), opts_.shm_name.c_str());
  };

  fd_ = shm_open(opts_.shm_name.c_str(), O_RDONLY, 0);
  if (fd_ < 0) {
    fail(std::string("shm_open failed: ") + std::strerror(errno));
    return;
  }

  struct stat st{};
  if (fstat(fd_, &st) != 0) {
    fail(std::string("fstat failed: ") + std::strerror(errno));
    return;
  }
  map_size_ = static_cast<size_t>(st.st_size);
  if (map_size_ < sizeof(SmNotifyHeader)) {
    fail("segment smaller than header");
    return;
  }

  map_ = mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    fail(std::string("mmap failed: ") + std::strerror(errno));
    return;
  }

  const auto* hdr = reinterpret_cast<const SmNotifyHeader*>(map_);
  if (hdr->magic != kSmNotifyMagic) {
    fail("bad magic");
    return;
  }
  if (hdr->version != kSmNotifyVersion) {
    fail("unsupported version");
    return;
  }

  const uint32_t num_slots = hdr->num_slots;
  const size_t need = sizeof(SmNotifyHeader) +
                      static_cast<size_t>(num_slots) * sizeof(SmNotifySlot);
  if (num_slots == 0 || map_size_ < need) {
    fail("segment too small for declared num_slots");
    return;
  }

  const auto* slots = reinterpret_cast<const SmNotifySlot*>(
      reinterpret_cast<const char*>(map_) + sizeof(SmNotifyHeader));
  for (uint32_t i = 0; i < num_slots; ++i) {
    if (slots[i].gpu_id == opts_.gpu_id &&
        slots[i].partition_idx == opts_.partition_idx) {
      slot_ = &slots[i];
      return;
    }
  }
  fail("no slot for (gpu_id, partition_idx)");
}

void ShmSmTargetSource::Detach() noexcept {
  slot_ = nullptr;
  if (map_ != nullptr) {
    munmap(map_, map_size_);
    map_ = nullptr;
  }
  map_size_ = 0;
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

int ShmSmTargetSource::Read() {
  if (!slot_) return kNoChange;
  // Seqlock read: retry until the sequence number is stable and even.
  for (int attempt = 0; attempt < 64; ++attempt) {
    uint32_t s1 =
        __atomic_load_n(const_cast<uint32_t*>(&slot_->seq), __ATOMIC_ACQUIRE);
    if (s1 & 1u) continue;
    int32_t target = __atomic_load_n(const_cast<int32_t*>(&slot_->target_sms),
                                     __ATOMIC_RELAXED);
    std::atomic_thread_fence(std::memory_order_acquire);
    uint32_t s2 =
        __atomic_load_n(const_cast<uint32_t*>(&slot_->seq), __ATOMIC_ACQUIRE);
    if (s1 == s2) return target;
  }
  return kNoChange;
}
