#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

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

inline void InitCachingAllocator(size_t bytes,
                                 CachingAllocator::MemoryType type,
                                 int device_id = -1) {
  // run only once, thread-safe
  if (kCachingAllocator == nullptr) {
    kCachingAllocator =
        std::make_unique<CachingAllocator>(bytes, type, device_id);
  }
}