#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include "memory/shm_attachment_manager.h"

namespace {

class ScopedShmSegment {
 public:
  ScopedShmSegment(std::string name, size_t size)
      : name_(std::move(name)), size_(size) {
    fd_ = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd_ == -1) {
      throw std::runtime_error(std::string("shm_open failed: ") +
                               std::strerror(errno));
    }

    int ret = ftruncate(fd_, static_cast<off_t>(size_));
    if (ret != 0) {
      close(fd_);
      fd_ = -1;
      shm_unlink(name_.c_str());
      throw std::runtime_error(std::string("ftruncate failed: ") +
                               std::strerror(errno));
    }
  }

  ~ScopedShmSegment() {
    if (fd_ >= 0) {
      close(fd_);
    }
    shm_unlink(name_.c_str());
  }

  const std::string& name() const { return name_; }
  size_t size() const { return size_; }

 private:
  std::string name_;
  size_t size_;
  int fd_ = -1;
};

std::string UniqueShmName() {
  static std::atomic<uint64_t> counter{0};
  return "/test_shm_attach_mgr_" + std::to_string(getpid()) + "_" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace

TEST(ShmAttachmentManagerTest, ReusesMappingAndRemapsAfterFinalRelease) {
  constexpr size_t kSize = 4 * 1024 * 1024;
  ScopedShmSegment segment(UniqueShmName(), kSize);
  ShmAttachmentManager manager;

  void* ptr1 = manager.GetOrAttach(segment.name(), segment.size());
  ASSERT_NE(ptr1, nullptr);

  void* ptr2 = manager.GetOrAttach(segment.name(), segment.size());
  ASSERT_EQ(ptr2, ptr1);

  manager.Release(segment.name());
  void* ptr3 = manager.GetOrAttach(segment.name(), segment.size());
  ASSERT_EQ(ptr3, ptr1);

  manager.Release(segment.name());
  manager.Release(segment.name());

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

  void* reserved =
      mmap(ptr1, kSize, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (reserved == MAP_FAILED) {
    GTEST_SKIP() << "Cannot reserve old mapping address, errno=" << errno;
  }

  void* ptr4 = manager.GetOrAttach(segment.name(), segment.size());
  ASSERT_NE(ptr4, nullptr);
  EXPECT_NE(ptr4, ptr1);
  manager.Release(segment.name());

  int ret = munmap(reserved, kSize);
  ASSERT_EQ(ret, 0);
}
