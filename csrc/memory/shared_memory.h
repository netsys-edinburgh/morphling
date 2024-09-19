#pragma once

#include <unistd.h>

void* OpenSharedMemory(const char* name, size_t size);
void CloseSharedMemory(void* ptr, size_t size);

struct ShmDeleter {
  void operator()(void* ptr) const { CloseSharedMemory(ptr, size); }
  size_t size;
};