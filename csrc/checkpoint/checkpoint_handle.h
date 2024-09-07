#pragma once

#include <string>
#include <unordered_map>

#include "archer_prio_aio_handle.h"
#include "archer_tensor_index.h"
#include "utils/noncopyable.h"

class CheckpointHandle : public noncopyable {
 public:
  explicit CheckpointHandle(const std::string& prefix);
  ~CheckpointHandle() = default;

  void ReadCheckpoint(
      std::unordered_map<std::string, uint64_t>& name_offset_map,
      std::unordered_map<std::string, uint64_t>& name_id_map);

 private:
  void* buffer_;
  std::string prefix_;
  ArcherTensorIndex tensor_index_;
  ArcherPrioAioHandle prio_aio_handle_;
};
