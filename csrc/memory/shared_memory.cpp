#include "shared_memory.h"

#include <fcntl.h>

void* OpenSharedMemory(const char* name, size_t size) {
  int shm_fd = shm_open(name, O_RDWR, 0666);
  if (shm_fd == -1) {
    LOG_ERROR << "shm_open failed. name: " << name << ", size: " << size
              << "; errno: " << errno << ", message: " << strerror(errno);
    return nullptr;
  }

  void* ptr =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    LOG_ERROR << "mmap failed. name: " << name << ", size: " << size
              << "; errno: " << errno << ", message: " << strerror(errno);
    close(shm_fd);
    return nullptr;
  }
  close(shm_fd);
  return ptr;
}

void CloseSharedMemory(void* ptr, size_t size) {
  int ret = munmap(ptr, size);
  LOG_FATAL_IF(ret == -1) << "munmap failed. ptr: " << ptr << ", size: " << size
                          << "; errno: " << errno
                          << ", message: " << strerror(errno);
}

std::tuple<void*, int> AttachSharedMemory(const char* name, size_t size) {
  int shm_fd = shm_open(name, O_RDWR, 0666);
  if (shm_fd == -1) {
    LOG_ERROR << "shm_open failed. name: " << name << ", size: " << size
              << "; errno: " << errno << ", message: " << strerror(errno);
    return {nullptr, -1};
  }

  void* ptr =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    LOG_ERROR << "mmap failed. name: " << name << ", size: " << size
              << "; errno: " << errno << ", message: " << strerror(errno);
    close(shm_fd);
    return {nullptr, -1};
  }
  return {ptr, shm_fd};
}

void DetachSharedMemory(void* ptr, int fd, size_t size) {
  int ret = munmap(ptr, size);
  LOG_FATAL_IF(ret == -1) << "munmap failed. ptr: " << ptr << ", size: " << size
                          << "; errno: " << errno
                          << ", message: " << strerror(errno);

  ret = close(fd);
  LOG_FATAL_IF(ret == -1) << "close failed. fd: " << fd << ", errno: " << errno
                          << ", message: " << strerror(errno);
}
