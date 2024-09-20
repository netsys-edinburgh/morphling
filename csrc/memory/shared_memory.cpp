#include "shared_memory.h"

#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/types.h>

#include "utils/logger.h"

void* OpenSharedMemory(const char* name, size_t size) {
  int shm_fd = shm_open(name, O_RDWR, 0666);
  LOG_FATAL_IF(shm_fd == -1,
               "shm_open failed. name: {}, size: {}; errno: {}, message: {}",
               name, size, errno, strerror(errno));

  void* ptr =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  LOG_FATAL_IF(ptr == MAP_FAILED,
               "mmap failed. name: {}, size: {}; errno: {}, message: {}", name,
               size, errno, strerror(errno));
  return ptr;
}

void CloseSharedMemory(void* ptr, size_t size) {
  int ret = munmap(ptr, size);
  LOG_FATAL_IF(ret == -1,
               "munmap failed. ptr: {0:x}, size: {}; errno: {}, message: {}",
               ptr, size, errno, strerror(errno));
}