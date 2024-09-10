#include "caching_allocator.h"

#include <cuda_runtime_api.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "common/types_and_defs.h"
#include "utils/logger.h"

std::unique_ptr<CachingAllocator> kCachingAllocator = nullptr;

void* CachingAllocator::Allocate(const size_t bytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = available_map_.find(bytes);
  if (it == available_map_.end() || it->second.empty()) {
    return AllocateAndCache(bytes);
  }
  void* ptr = it->second.back();
  it->second.pop_back();
  return ptr;
}

void CachingAllocator::Free(void* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = allocation_map_.find(ptr);
  if (it == allocation_map_.end()) {
    FreeMemory(ptr);
    return;
  }
  const size_t alloc_size = it->second;
  available_map_[alloc_size].push_back(ptr);
}

void CachingAllocator::FreeCached() {
  for (const auto& it : available_map_) {
    for (const auto& ptr : it.second) {
      FreeMemory(ptr);
      allocated_bytes_ -= it.first;
    }
  }
  available_map_.clear();
}

void* CachingAllocator::AllocateAndCache(const size_t bytes) {
  if (allocated_bytes_ + bytes > max_bytes_) {
    FreeCached();
    LOG_FATAL_IF(
        allocated_bytes_ + bytes > max_bytes_,
        "Out of memory; attempted to allocate {}GB, but only {}GB available",
        bytes / GB, (max_bytes_ - allocated_bytes_) / GB);
  }
  void* ptr = AllocateMemory(bytes);
  allocation_map_[ptr] = bytes;
  return ptr;
}

void* CachingAllocator::AllocateMemory(size_t bytes) {
  switch (type_) {
    case MemoryType::SHM:
      return AllocShmMemory(bytes);
    case MemoryType::PIN:
      return AllocPinMemory(bytes);
    case MemoryType::CUDA:
      return AllocCudaMemory(bytes);
  }
  LOG_FATAL("Unknown memory type");
  return nullptr;
}
void* CachingAllocator::AllocCudaMemory(size_t bytes) {
  void* ptr;
  cudaSetDevice(device_id_);
  cudaMalloc(&ptr, bytes);
  return ptr;
}
void* CachingAllocator::AllocPinMemory(size_t bytes) {
  void* ptr;
  cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
  return ptr;
}
void* CachingAllocator::AllocShmMemory(size_t bytes) {
  int shm_id = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0666);
  void* ptr = shmat(shm_id, nullptr, 0);
  shm_id_map_[ptr] = shm_id;
  return ptr;
}

void CachingAllocator::FreeMemory(void* ptr) {
  switch (type_) {
    case MemoryType::SHM:
      FreeShmMemory(ptr);
      break;
    case MemoryType::PIN:
      FreePinMemory(ptr);
      break;
    case MemoryType::CUDA:
      FreeCudaMemory(ptr);
      break;
  }
}
void CachingAllocator::FreeCudaMemory(void* ptr) {
  cudaSetDevice(device_id_);
  cudaFree(ptr);
}
void CachingAllocator::FreePinMemory(void* ptr) { cudaFreeHost(ptr); }
void CachingAllocator::FreeShmMemory(void* ptr) {
  shmdt(ptr);
  auto shmid = shm_id_map_[ptr];
  shmctl(shmid, IPC_RMID, nullptr);
  shm_id_map_.erase(ptr);
}

CachingAllocator::CachingAllocator(size_t bytes, MemoryType type, int device_id)
    : max_bytes_(bytes),
      allocated_bytes_(0),
      type_(type),
      device_id_(device_id) {}

CachingAllocator::~CachingAllocator() { FreeCached(); }
