#pragma once

#include <torch/torch.h>

#include <cstring>
#include <mutex>

#include "caching_allocator.h"

struct TorchCachingAllocator : public torch::Allocator {
  // For Torch Interface
  torch::DataPtr allocate(size_t n) override {
    void* data = TorchAllocate(n);
    return {data, data, &TorchFree, torch::DeviceType::CPU};
  }

  void copy_data(void* dest, const void* src, size_t count) const override {
    LOG_DEBUG << "Copy data from " << src << " to " << dest
              << ", size: " << count;
    memcpy(dest, src, count);
  }

  // // Optional: Handle deallocation (if needed)
  // void deallocate(void* ptr) override {
  //   Free(ptr);  // Custom deallocation logic
  // }
};

void ActivatePinShmTorchAllocator();
