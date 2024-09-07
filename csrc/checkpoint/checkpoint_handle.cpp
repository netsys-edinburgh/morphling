#include "checkpoint_handle.h"

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "archer_tensor_handle.h"

CheckpointHandle::CheckpointHandle(const std::string& prefix)
    : prefix_(prefix), prio_aio_handle_(prefix) {
  buffer_ = nullptr;

  if (prefix_.back() != '/') {
    prefix_ += '/';
  }
}

void CheckpointHandle::ReadCheckpoint(
    std::unordered_map<std::string, uint64_t>& name_offset_map,
    std::unordered_map<std::string, uint64_t>& name_id_map) {
  int file_id = 0;  // FIXME: hard code only one file
  auto param_filename =
      prefix_ + std::string(ARCHER_PARAM_NAME) + "_" + std::to_string(file_id);
  auto index_filename = prefix_ + std::string(ARCHER_IHDEX_NAME);

  // get param_filename file size
  struct stat st;
  if (stat(param_filename.c_str(), &st) == -1) {
    LOG(FATAL) << "Invalid prefix: " << param_filename << " does not exist";
  }
  auto file_size = st.st_size;

  CHECK(buffer_ == nullptr) << "Buffer is not null, should only load once";
  cudaHostAlloc(&buffer_, file_size, cudaHostAllocDefault);

  tensor_index_.Deserialize(index_filename.c_str());

  for (auto& [name, buffer_offset] : name_offset_map) {
    auto id = name_id_map[name];
    auto tensor_meta = tensor_index_[id];
    auto file_id = tensor_meta.file_id;

    auto file_offset = tensor_meta.offset;
    auto num_bytes = tensor_meta.size;

    param_filename = prefix_ + std::string(ARCHER_PARAM_NAME) + "_" +
                     std::to_string(file_id);

    prio_aio_handle_.Read(param_filename, buffer_ + buffer_offset, false,
                          num_bytes, file_offset);
  }
}