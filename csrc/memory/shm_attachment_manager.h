#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

#include "shared_memory.h"

class ShmAttachmentManager {
 public:
  void* GetOrAttach(const std::string& name, size_t size);
  void Release(const std::string& name);
  ~ShmAttachmentManager();

 private:
  struct Attachment {
    void* ptr = nullptr;
    int fd = -1;
    size_t size = 0;
    int refcount = 0;
  };

  std::unordered_map<std::string, Attachment> attachments_;
  std::mutex mu_;
};
