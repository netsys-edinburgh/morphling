#include <gtest/gtest.h>

#include <cstdlib>

#include "memory/caching_allocator.h"
#include "memory/torch_caching_allocator.h"

TEST(TorchCachingAllocatorActivationTest,
     ActivatePinShmTorchAllocatorInitializesAndTracksTensorAllocation) {
  ASSERT_EQ(setenv("MORPHLING_PIN_SIZE", "1048576", 1), 0);

  ActivatePinShmTorchAllocator();
  ActivatePinShmTorchAllocator();

  ASSERT_NE(kCachingAllocator, nullptr);
  EXPECT_EQ(kCachingAllocator->GetType(), MemoryType::PIN_SHM);

  auto tensor = torch::zeros({32}, torch::TensorOptions().dtype(torch::kFloat));
  ASSERT_NE(tensor.data_ptr(), nullptr);
  EXPECT_TRUE(kCachingAllocator->IsAllocated(tensor.data_ptr()));
}
