#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "utils/logger.h"
#include "utils/noncopyable.h"

// the caching allocator that supports CPU and CUDA memory
// work as an offset manager for the memory pool
class CachingAllocator : public noncopyable {
 public:
  enum class MemoryType { SHM, PIN, CUDA };

 public:
  explicit CachingAllocator(size_t bytes, MemoryType type, int device_id = -1);
  virtual ~CachingAllocator();

  virtual void* Allocate(const size_t bytes);
  virtual void Free(void* ptr);

  int GetShmId(void* ptr) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto& it = shm_id_map_.find(ptr);
    if (it == shm_id_map_.end()) {
      return -1;
    }
    return it->second;
  }

 private:
  void* AllocateAndCache(const size_t bytes);
  void FreeCached();

  void* AllocateMemory(size_t bytes);

  void* AllocCudaMemory(size_t bytes);
  void* AllocPinMemory(size_t bytes);
  void* AllocShmMemory(size_t bytes);

  void FreeMemory(void* ptr);
  void FreeCudaMemory(void* ptr);
  void FreePinMemory(void* ptr);
  void FreeShmMemory(void* ptr);

 protected:
  int device_id_;
  MemoryType type_;
  size_t max_bytes_;
  size_t allocated_bytes_;

  std::unordered_map<size_t, std::deque<void*>> available_map_;
  std::unordered_map<void*, size_t> allocation_map_;
  std::mutex mutex_;
  std::unordered_map<void*, int> shm_id_map_;
};

extern std::unique_ptr<CachingAllocator> kCachingAllocator;

static void InitCachingAllocator(CachingAllocator::MemoryType type,
                                 int device_id = -1) {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    size_t bytes = 0;
    if (type == CachingAllocator::MemoryType::CUDA) {
      LOG_FATAL_IF(device_id < 0, "Invalid device id");
      // Get environment variable MORPHLING_SHM_SIZE
      const char* size = std::getenv("MORPHLING_GPU_SIZE");
      LOG_FATAL_IF(size == nullptr, "MORPHLING_GPU_SIZE is not set");
      bytes = std::stoull(size);
    } else if (type == CachingAllocator::MemoryType::SHM) {
      // Get environment variable MORPHLING_SHM_SIZE
      const char* size = std::getenv("MORPHLING_SHM_SIZE");
      LOG_FATAL_IF(size == nullptr, "MORPHLING_SHM_SIZE is not set");
      bytes = std::stoull(size);
    } else if (type == CachingAllocator::MemoryType::PIN) {
      const char* size = std::getenv("MORPHLING_PIN_SIZE");
      LOG_FATAL_IF(size == nullptr, "MORPHLING_PIN_SIZE is not set");
      bytes = std::stoull(size);
    } else {
      LOG_FATAL("Unknown memory type");
    }
    kCachingAllocator =
        std::make_unique<CachingAllocator>(bytes, type, device_id);
  });
}