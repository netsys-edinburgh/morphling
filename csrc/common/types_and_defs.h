#pragma once
#include <cstdint>

#define PARAM_META_FILE "param_meta.json"
#define BUFFER_SHM_NAME "/emulator_shm_buffer"
#define PARAM_SHM_NAME "/emulator_shm_param"
#define ALIGNMENT 512

typedef std::uint32_t TensorID;
typedef std::size_t HashID;
typedef std::size_t NodeID;
typedef std::uint64_t GraphID;
typedef std::uint64_t RequestID;

#define KB 1024
#define MB (KB * KB)
#define GB (KB * KB * KB)

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};
