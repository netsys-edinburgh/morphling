#pragma once
#include <torch/torch.h>

#include <cstdint>

#define PARAM_META_FILE "param_meta.json"

#define PARAM_SHM_NAME "/emulator_shm"
#define ALIGNMENT 512

#define SHM_SIZE (100L * 1024L * 1024L * 1024L)
#define MAX_TASKS 100

typedef std::uint32_t TensorID;
typedef std::size_t HashID;
typedef std::size_t NodeID;
typedef std::uint64_t GraphID;
typedef std::uint64_t RequestID;

#define KB 1024
#define MB (KB * KB)
#define GB (KB * KB * KB)

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kMeta)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

#define FLOAT32_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat32).device(target)
#define FLOAT16_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat16).device(target)
#define FAKE_TENSOR_SIZES torch::IntArrayRef({1})

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};
