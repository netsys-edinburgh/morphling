#include "interceptor.h"

#include <dlfcn.h>
#include <stdio.h>

#include <cstdint>

IMPL_INTERCEPTOR(sgemm, float, "libmkl_rt.so")
IMPL_INTERCEPTOR(sgemm_batch, float, "libmkl_rt.so")

bool CheckBufferOffloaded(const void* buffer, size_t size) {
  // when the buffer is offloaded, the first and last uint32_t is the same
  const uint32_t* buffer_uint32 = reinterpret_cast<const uint32_t*>(buffer);
  const uint32_t first_num = buffer_uint32[0];
  const uint32_t last_num = buffer_uint32[size / sizeof(uint32_t) - 1];
  return (first_num == last_num) && (first_num < 0xFFFF) && (last_num < 0xFFFF);
}