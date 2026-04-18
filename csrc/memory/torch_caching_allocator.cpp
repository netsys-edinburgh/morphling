#include "torch_caching_allocator.h"

std::once_flag g_activate_pin_shm_torch_allocator_flag;
TorchCachingAllocator* g_torch_caching_allocator = nullptr;

void ActivatePinShmTorchAllocator() {
  std::call_once(g_activate_pin_shm_torch_allocator_flag, []() {
    InitCachingAllocator(MemoryType::PIN_SHM);
    g_torch_caching_allocator = new TorchCachingAllocator();
    LOG_INFO << "ActivatePinShmTorchAllocator: replacing torch CPU allocator";
    torch::SetAllocator(torch::DeviceType::CPU, g_torch_caching_allocator);
    LOG_INFO << "ActivatePinShmTorchAllocator: torch allocator replaced";
  });
}
