// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_tensor_handle.h"

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <torch/script.h>

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

const char* ARCHER_PARAM_NAME = "archer_param";
const char* ARCHER_IHDEX_NAME = "archer_index";

ArcherTensorHandle::ArcherTensorHandle(const std::string& prefix)
    : prefix_(prefix), prio_aio_handle_(prefix), file_id_(0), file_offset_(0) {
  // InitLogger();
  google::InitGoogleLogging("morphling");

  if (prefix_.back() != '/') {
    prefix_ += '/';
  }

  struct stat st;
  if (stat(prefix_.c_str(), &st) != -1 && !S_ISDIR(st.st_mode)) {
    LOG(FATAL) << "Invalid prefix: " << prefix_ << " is not a directory";
  }
  if (stat(prefix_.c_str(), &st) == -1) {
    LOG(WARNING) << "Invalid prefix: " << prefix_
                 << " does not exist, creating";
    mkdir(prefix_.c_str(), 0777);
  }

  auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);
  if (access(ckpt_index_path.c_str(), F_OK) != -1) {
    LOG(INFO) << "Loading index file from " << ckpt_index_path;
    tensor_index_->Deserialize(ckpt_index_path.c_str());
    is_serialized_ = true;
  } else {
    LOG(INFO) << "Index file " << ckpt_index_path
              << " does not exist, creating";
  }
  LOG(INFO) << "Index file size " << tensor_index_->size();
}

void ArcherTensorHandle::OffloadTensor(torch::Tensor& tensor,
                                       const std::uint32_t tensor_id) {
  StoreTensor(tensor_id, tensor);

  auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);

  std::unique_lock<std::mutex> lock(mutex_);
  tensor_index_->Serialize(ckpt_index_path.c_str());
}

bool ArcherTensorHandle::IsTensorOffloaded(const std::uint32_t tensor_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = tensor_index_->find(tensor_id);
  // ARCHER_LOG_DEBUG("Check tensor {} {}", tensor_id, it ==
  // tensor_index_->end());
  bool is_offloaded = it != tensor_index_->end();
  if (is_offloaded) {
    it->second.id = tensor_id;
  }
  return is_offloaded;
}

void ArcherTensorHandle::StoreTensor(const std::uint32_t tensor_id,
                                     torch::Tensor& buffer) {
  auto it = tensor_index_->find(tensor_id);
  bool tensor_exists = (it != tensor_index_->end());

  std::unique_lock<std::mutex> lock(mutex_);
  TensorStorageMeta tensor_meta{file_id_, file_offset_, buffer.nbytes(),
                                buffer.sizes().vec()};
  tensor_meta.options = buffer.options();
  tensor_meta.id = tensor_id;

  auto num_bytes = buffer.nbytes();
  std::int64_t num_bytes_aligned =
      (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);

  if (tensor_exists) {
    // size must be the same if found
    if (it->second.size != buffer.nbytes()) {
      LOG(FATAL) << "Tensor " << tensor_id << " size mismatch "
                 << it->second.size << " != " << buffer.nbytes();
    }
    tensor_meta = it->second;
  }

  file_offset_ += tensor_exists ? 0 : num_bytes_aligned;

  tensor_index_->insert(std::make_pair(tensor_id, tensor_meta));

  auto filename = GetIndexFileName(tensor_meta.file_id);

  lock.unlock();
  prio_aio_handle_.Write(filename, buffer.data_ptr(), false, tensor_meta.size,
                         tensor_meta.offset);
}

int64_t ArcherTensorHandle::GetTensorSizeAligned(
    const std::uint32_t tensor_id) const {
  auto it = tensor_index_->find(tensor_id);
  if (it == tensor_index_->end()) {
    LOG(FATAL) << "Tensor not found " << tensor_id;
  }
  auto num_bytes = it->second.size;
  std::int64_t num_bytes_aligned =
      (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);
  return num_bytes_aligned;
}

torch::TensorOptions ArcherTensorHandle::GetTensorOptions(
    const std::uint32_t tensor_id) const {
  auto it = tensor_index_->find(tensor_id);
  if (it == tensor_index_->end()) {
    LOG(FATAL) << "Tensor not found " << tensor_id;
  }
  return it->second.options;
}

void ArcherTensorHandle::SetTensor(std::uint32_t tensor_id,
                                   torch::Tensor& buffer,
                                   const torch::Device& device) {
  auto it = tensor_index_->find(tensor_id);
  if (it == tensor_index_->end()) {
    LOG(FATAL) << "Tensor not found " << tensor_id;
  }
  // FIXME: this is may creates extra copy of data, need to be confirmed
  // optimized CANNOT use shallow copy here, e.g., buffer =
  // it->second.tensor.to(DEFAULT_CUDA_DEVICE);

  buffer.set_data(it->second.tensor.to(device).to(buffer.dtype()));
}

void ArcherTensorHandle::SetTensor(std::uint32_t tensor_id,
                                   torch::Tensor& buffer) {
  auto it = tensor_index_->find(tensor_id);
  if (it == tensor_index_->end()) {
    LOG(FATAL) << "Tensor not found " << tensor_id;
  }
  if (buffer.dtype() != it->second.tensor.dtype()) {
    std::ostringstream oss;
    oss << buffer.dtype() << " -> " << it->second.tensor.dtype();
    DLOG(INFO) << "Tensor dtype mismatch " << tensor_id << " " << oss.str();
    buffer.set_data(it->second.tensor.to(buffer.dtype()));
  } else {
    buffer.set_data(it->second.tensor);
  }
  DLOG(INFO) << "Set tensor to device " << tensor_id << " " << buffer.device();
}

std::string ArcherTensorHandle::GetIndexFileName(
    const std::uint32_t file_id) const {
  return prefix_ + std::string(ARCHER_PARAM_NAME) + "_" +
         std::to_string(file_id);
}

std::uint32_t ArcherTensorHandle::GetTensorId(void* tensor) const {
  auto it = tensor_to_id_.find(tensor);
  if (it == tensor_to_id_.end()) {
    LOG(FATAL) << "Tensor not found " << std::hex << (void*)tensor;
    return UINT32_MAX;
  }
  return it->second;
}

void ArcherTensorHandle::ReadTensor(const uint32_t tensor_id, void* memory_ptr,
                                    bool on_demand) {
  auto it = tensor_index_->find(tensor_id);
  if (it == tensor_index_->end()) {
    LOG(FATAL) << "Tensor not found " << tensor_id;
  }

  auto tensor_meta = it->second;
  auto filename = GetIndexFileName(tensor_meta.file_id);

  prio_aio_handle_.Read(filename, memory_ptr, on_demand, tensor_meta.size,
                        tensor_meta.offset);
}
