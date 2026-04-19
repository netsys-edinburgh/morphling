#include "shm_attachment_manager.h"

#include <vector>

void* ShmAttachmentManager::GetOrAttach(const std::string& name, size_t size) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = attachments_.find(name);
  if (it != attachments_.end()) {
    it->second.refcount++;
    return it->second.ptr;
  }

  auto [ptr, fd] = AttachSharedMemory(name.c_str(), size);
  Attachment a;
  a.ptr = ptr;
  a.fd = fd;
  a.size = size;
  a.refcount = 1;
  attachments_[name] = a;
  return ptr;
}

void ShmAttachmentManager::Release(const std::string& name) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = attachments_.find(name);
  if (it == attachments_.end()) {
    return;
  }

  it->second.refcount--;
  if (it->second.refcount > 0) {
    return;
  }

  DetachSharedMemory(it->second.ptr, it->second.fd, it->second.size);
  attachments_.erase(it);
}

ShmAttachmentManager::~ShmAttachmentManager() {
  std::vector<ShmAttachmentManager::Attachment> pending;
  {
    std::lock_guard<std::mutex> lock(mu_);
    pending.reserve(attachments_.size());
    for (const auto& entry : attachments_) {
      pending.push_back(entry.second);
    }
    attachments_.clear();
  }

  for (const auto& attachment : pending) {
    if (attachment.ptr) {
      DetachSharedMemory(attachment.ptr, attachment.fd, attachment.size);
    }
  }
}
