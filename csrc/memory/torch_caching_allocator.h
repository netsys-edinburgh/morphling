#pragma once

#include <torch/torch.h>

#include "caching_allocator.h"

struct TorchCachingAllocator : public torch::Allocator {
  // For Torch Interface
  torch::DataPtr allocate(size_t n) override {
    void* data = TorchAllocate(n);
    return {data, data, &TorchFree, torch::DeviceType::CPU};
  }

  void copy_data(void* dest, const void* src, size_t count) const override {
    // if (type_ == MemoryType::CUDA) {
    //   cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice);
    // } else if (type_ == MemoryType::PIN_SHM) {
    //   cudaMemcpy(dest, src, count, cudaMemcpyHostToHost);
    // } else {
    //   memcpy(dest, src, count);
    // }
    memcpy(dest, src, count);
  }

  // // Optional: Handle deallocation (if needed)
  // void deallocate(void* ptr) override {
  //   Free(ptr);  // Custom deallocation logic
  // }
};

// extern std::unique_ptr<TorchCachingAllocator> kTorchCachingAllocator;

class ReplaceTorchAllocatorOnLoad {
 public:
  ReplaceTorchAllocatorOnLoad() {
    InitCachingAllocator(MemoryType::PIN_SHM);
    torch_caching_allocator_ = new TorchCachingAllocator();
    LOG_DEBUG("Replace torch allocator with caching allocator");
    torch::SetAllocator(torch::DeviceType::CPU, torch_caching_allocator_);
    LOG_DEBUG("Torch allocator replaced");
  }

 private:
  TorchCachingAllocator* torch_caching_allocator_;
};

// Create a static instance of this class
static ReplaceTorchAllocatorOnLoad kReplaceTorchAllocatorOnLoad;