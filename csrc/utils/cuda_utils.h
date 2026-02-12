// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;           \
      std::exit(err);                                                  \
    }                                                                  \
  } while (0)

bool IsDevicePointer(const void* ptr);
int GetDeviceCount();
std::size_t GetTotalDeviceMemory(int device_id);
std::size_t GetFreeDeviceMemory(int device_id);

#define DEVICE_CACHE_LIMIT(gid) GetTotalDeviceMemory(gid) * 0.7
#define NUM_DEVICES GetDeviceCount()
