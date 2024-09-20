#pragma once

#include <rttr/registration.h>

#include <filesystem>
#include <string>
#include <unordered_map>

#include "archer_prio_aio_handle.h"
#include "archer_tensor_index.h"
#include "memory/caching_allocator.h"
#include "utils/noncopyable.h"
struct ParamMeta {
  uint32_t id;
  size_t size;
  // size_t shm_offset;
  size_t file_offset;
};

// Register the struct with RTTR
RTTR_REGISTRATION {
  rttr::registration::class_<ParamMeta>("ParamMeta")
      .property("id", &ParamMeta::id)
      .property("size", &ParamMeta::size)
      // .property("shm_offset", &ParamMeta::shm_offset)
      .property("file_offset", &ParamMeta::file_offset);
}

typedef std::unordered_map<std::string, ParamMeta> ParamMetaMap;

class CheckpointHandle : public noncopyable {
 public:
  explicit CheckpointHandle(const std::filesystem::path& prefix);
  ~CheckpointHandle() = default;

  void ReadCheckpoint();

 private:
  std::filesystem::path GetFilePathByID(uint32_t file_id) const;

  std::vector<uint32_t> FindIDsSameSize(size_t size);
  std::tuple<uint64_t, std::unordered_map<size_t, size_t>> ComputeShmOffsets();
  std::tuple<uint64_t, std::unordered_map<std::string, size_t>>
  ComputePinOffsets();

 private:
  void* buffer_;
  std::filesystem::path prefix_;
  ArcherTensorIndex tensor_index_;
  ArcherPrioAioHandle prio_aio_handle_;
  ParamMetaMap param_meta_map_;
  std::unique_ptr<CachingAllocator> allocator_;
  ParamShmMap param_shm_map_;
  std::once_flag flag_;
};
