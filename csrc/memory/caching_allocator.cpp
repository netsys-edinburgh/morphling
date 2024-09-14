#include "caching_allocator.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <uuid/uuid.h>

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
    case MemoryType::PIN_SHM:
      return AllocPinShmMemory(bytes);
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
  // void* ptr = aligned_alloc(4096, bytes);
  // int ret = mlock(ptr, bytes);
  // LOG_FATAL_IF(ret != 0, "mlock failed: errno {}, message {}", errno,
  //              strerror(errno));
  // cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
  void* ptr;
  cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
  return ptr;
}
void* CachingAllocator::AllocShmMemory(size_t bytes) {
  int shm_id = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0666);
  void* ptr = shmat(shm_id, nullptr, 0);
  LOG_FATAL_IF(ptr == (void*)-1, "shmat failed: errno {}, message {}", errno,
               strerror(errno));
  shm_id_map_[ptr] = {shm_id, ptr, bytes};
  return ptr;
}
void* CachingAllocator::AllocPinShmMemory(size_t bytes) {
  // generate uuid string
  ShmMeta shm_meta;
  uuid_t bin_uuid;
  uuid_generate_random(bin_uuid);
  uuid_unparse(bin_uuid, shm_meta.name + 1);
  shm_meta.name[0] = '/';  // shm_open requires the first char to be '/'

  int shm_fd = shm_open(shm_meta.name, O_CREAT | O_RDWR, 0666);
  LOG_FATAL_IF(shm_fd == -1, "shm_open failed: errno {}, message {}", errno,
               strerror(errno));
  LOG_FATAL_IF(ftruncate(shm_fd, bytes) == -1,
               "ftruncate failed: errno {}, message {}", errno,
               strerror(errno));

  // Specify the fixed address
  void* preferred_addr = AllocPinMemory(bytes);
  void* shm_addr = mmap(preferred_addr, bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED | MAP_LOCKED, shm_fd, 0);
  LOG_FATAL_IF(shm_addr == MAP_FAILED, "mmap failed: errno {}, message {}",
               errno, strerror(errno));

  shm_meta.id = shm_fd;
  shm_meta.ptr = preferred_addr;
  shm_meta.size = bytes;
  shm_id_map_[preferred_addr] = shm_meta;
  return preferred_addr;
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
    case MemoryType::PIN_SHM:
      FreePinShmMemory(ptr);
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
  auto shmid = shm_id_map_[ptr].id;
  shmctl(shmid, IPC_RMID, nullptr);
  shm_id_map_.erase(ptr);
}
void CachingAllocator::FreePinShmMemory(void* ptr) {
  munmap(ptr, shm_id_map_[ptr].size);
  shm_unlink(shm_id_map_[ptr].name);
  close(shm_id_map_[ptr].id);
  FreePinMemory(ptr);
  shm_id_map_.erase(ptr);
}

CachingAllocator::CachingAllocator(size_t bytes, MemoryType type, int device_id)
    : max_bytes_(bytes),
      allocated_bytes_(0),
      type_(type),
      device_id_(device_id) {}

CachingAllocator::~CachingAllocator() { FreeCached(); }

extern "C" {
void* TorchAllocate(size_t bytes) {
  InitCachingAllocator(CachingAllocator::MemoryType::PIN_SHM);
  void* ptr = kCachingAllocator->Allocate(bytes);
  return ptr;
}

void TorchFree(void* ptr) {
  InitCachingAllocator(CachingAllocator::MemoryType::PIN_SHM);
  LOG_FATAL_IF(kCachingAllocator->IsAllocated(ptr) == false,
               "Attempted to free unallocated memory");
  kCachingAllocator->Free(ptr);
}
}