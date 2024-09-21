#pragma once

#include <unistd.h>

#include <cstring>  // for memset
#include <rttr/registration>
#include <rttr/type>
#include <string>

#include "common/types_and_defs.h"

void* OpenSharedMemory(const char* name, size_t size);
void CloseSharedMemory(void* ptr, size_t size);

struct ShmDeleter {
  void operator()(void* ptr) const { CloseSharedMemory(ptr, size); }
  size_t size;
};
