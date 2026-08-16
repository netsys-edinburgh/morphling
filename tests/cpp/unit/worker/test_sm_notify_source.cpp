#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "scheduler/sm_notify_source.h"

namespace {

// A test-owned read/write shared-memory segment that plays the role of the
// external producer: creates the segment, lays out header + slots, and
// publishes target_sms updates under the same seqlock protocol the consumer
// expects.
class ShmProducer {
 public:
  ShmProducer(const std::string& name, uint32_t num_slots,
              uint32_t magic = kSmNotifyMagic,
              uint32_t version = kSmNotifyVersion)
      : name_(name) {
    shm_unlink(name_.c_str());  // drop any stale segment
    fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
    EXPECT_GE(fd_, 0) << "shm_open failed for " << name_;
    size_ = sizeof(SmNotifyHeader) +
            static_cast<size_t>(num_slots) * sizeof(SmNotifySlot);
    EXPECT_EQ(ftruncate(fd_, static_cast<off_t>(size_)), 0);
    map_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    EXPECT_NE(map_, MAP_FAILED);
    std::memset(map_, 0, size_);
    auto* h = header();
    h->magic = magic;
    h->version = version;
    h->num_slots = num_slots;
  }

  ~ShmProducer() {
    if (map_ && map_ != MAP_FAILED) munmap(map_, size_);
    if (fd_ >= 0) close(fd_);
    shm_unlink(name_.c_str());
  }

  SmNotifyHeader* header() { return reinterpret_cast<SmNotifyHeader*>(map_); }

  SmNotifySlot* slot(uint32_t i) {
    char* base = reinterpret_cast<char*>(map_) + sizeof(SmNotifyHeader);
    return reinterpret_cast<SmNotifySlot*>(base) + i;
  }

  void InitSlot(uint32_t i, int gpu, int part, int target) {
    SmNotifySlot* s = slot(i);
    s->seq = 0;
    s->gpu_id = gpu;
    s->partition_idx = part;
    s->target_sms = target;
    s->generation = 0;
  }

  // seqlock publish: seq even->odd, write fields, seq odd->even.
  void Publish(uint32_t i, int target) {
    SmNotifySlot* s = slot(i);
    __atomic_store_n(&s->seq, s->seq + 1, __ATOMIC_RELEASE);
    s->target_sms = target;
    __atomic_store_n(&s->generation, s->generation + 1, __ATOMIC_RELAXED);
    __atomic_store_n(&s->seq, s->seq + 1, __ATOMIC_RELEASE);
  }

 private:
  std::string name_;
  int fd_ = -1;
  size_t size_ = 0;
  void* map_ = nullptr;
};

std::string ShmName(const char* suffix) {
  return "/morphling_smtest_" + std::to_string(getpid()) + "_" + suffix;
}

}  // namespace

TEST(ShmSmTargetSource, AttachAndReadReturnsWrittenTarget) {
  const std::string name = ShmName("a");
  ShmProducer prod(name, /*num_slots=*/2);
  prod.InitSlot(0, /*gpu=*/0, /*part=*/0, /*target=*/42);
  prod.InitSlot(1, /*gpu=*/1, /*part=*/0, /*target=*/99);

  ShmSmTargetSource src({name, /*gpu=*/0, /*part=*/0, /*require=*/true});
  EXPECT_TRUE(src.Attached());
  EXPECT_EQ(src.Read(), 42);

  prod.Publish(0, 24);
  EXPECT_EQ(src.Read(), 24);
}

TEST(ShmSmTargetSource, SelectsSlotByGpuAndPartition) {
  const std::string name = ShmName("b");
  ShmProducer prod(name, 2);
  prod.InitSlot(0, 0, 0, 42);
  prod.InitSlot(1, 1, 0, 99);

  ShmSmTargetSource src({name, /*gpu=*/1, /*part=*/0, /*require=*/true});
  EXPECT_TRUE(src.Attached());
  EXPECT_EQ(src.Read(), 99);
}

TEST(ShmSmTargetSource, BadMagicBestEffortNoChange) {
  const std::string name = ShmName("c");
  ShmProducer prod(name, 1, /*magic=*/0xDEADBEEFu);
  prod.InitSlot(0, 0, 0, 42);

  ShmSmTargetSource src({name, 0, 0, /*require=*/false});
  EXPECT_FALSE(src.Attached());
  EXPECT_EQ(src.Read(), SmTargetSource::kNoChange);
}

TEST(ShmSmTargetSource, BadVersionBestEffortNoChange) {
  const std::string name = ShmName("d");
  ShmProducer prod(name, 1, kSmNotifyMagic, /*version=*/999u);
  prod.InitSlot(0, 0, 0, 42);

  ShmSmTargetSource src({name, 0, 0, /*require=*/false});
  EXPECT_FALSE(src.Attached());
  EXPECT_EQ(src.Read(), SmTargetSource::kNoChange);
}

TEST(ShmSmTargetSource, SlotNotFoundBestEffortNoChange) {
  const std::string name = ShmName("e");
  ShmProducer prod(name, 1);
  prod.InitSlot(0, 0, 0, 42);

  ShmSmTargetSource src({name, /*gpu=*/5, /*part=*/5, /*require=*/false});
  EXPECT_FALSE(src.Attached());
  EXPECT_EQ(src.Read(), SmTargetSource::kNoChange);
}

TEST(ShmSmTargetSource, SlotNotFoundRequireThrows) {
  const std::string name = ShmName("f");
  ShmProducer prod(name, 1);
  prod.InitSlot(0, 0, 0, 42);

  EXPECT_THROW(
      ShmSmTargetSource({name, /*gpu=*/5, /*part=*/5, /*require=*/true}),
      std::runtime_error);
}

TEST(ShmSmTargetSource, MissingSegmentRequireThrows) {
  EXPECT_THROW(ShmSmTargetSource({ShmName("absent"), 0, 0, /*require=*/true}),
               std::runtime_error);
}

TEST(ShmSmTargetSource, MissingSegmentBestEffortNoChange) {
  ShmSmTargetSource src({ShmName("absent2"), 0, 0, /*require=*/false});
  EXPECT_FALSE(src.Attached());
  EXPECT_EQ(src.Read(), SmTargetSource::kNoChange);
}
