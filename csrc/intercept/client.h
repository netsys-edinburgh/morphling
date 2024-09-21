#pragma once

#include <grpcpp/grpcpp.h>
// #include <torch/torch.h>

#include "memory/caching_allocator.h"
#include "morphling.grpc.pb.h"
#include "morphling.pb.h"
#include "utils/logger.h"
#include "utils/noncopyable.h"

#define SET_SHM_INFO_REPEAT(request, ptr)                           \
  do {                                                              \
    auto* info = request.add_##ptr##_info();                        \
    auto size = kCachingAllocator->GetShmSize(ptr);                 \
    auto name = kCachingAllocator->GetShmName(ptr);                 \
    info->set_size(size);                                           \
    info->set_name(name);                                           \
    LOG_DEBUG("Set {} info: size: {}, name: {}", #ptr, size, name); \
  } while (0);

#define SET_SHM_INFO(request, ptr)                                  \
  do {                                                              \
    auto* info = request.mutable_##ptr##_info();                    \
    auto size = kCachingAllocator->GetShmSize(ptr);                 \
    auto name = kCachingAllocator->GetShmName(ptr);                 \
    info->set_size(size);                                           \
    info->set_name(name);                                           \
    LOG_DEBUG("Set {} info: size: {}, name: {}", #ptr, size, name); \
  } while (0);

class MemoryManagerClient : public noncopyable {
 public:
  MemoryManagerClient(std::shared_ptr<grpc::Channel> channel)
      : stub_(morphling::MemoryManager::NewStub(channel)) {}

  MemoryManagerClient() {
    char* server_address = std::getenv("MORPHLING_SERVER_ADDRESS");
    LOG_FATAL_IF(server_address == nullptr,
                 "MORPHLING_SERVER_ADDRESS is not set");
    auto channel =
        grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    stub_ = morphling::MemoryManager::NewStub(channel);
  }

  // return <param_name, <shm_name, size>>, need explicit for python types
  std::unordered_map<std::string, std::tuple<std::string, size_t>>
  GetModelParam();
  void ScheduleGemmSync(void* a, void* b, void* c, void* task);

  // void SetTensorShm(torch::Tensor& tensor, const std::string& name);

 private:
  std::unique_ptr<morphling::MemoryManager::Stub> stub_;
  std::unordered_map<std::string, void*> shm_map_;
};

extern std::unique_ptr<MemoryManagerClient> kMemoryManagerClient;

static void InitMemoryManagerClient() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    // get env var MORPHLING_SERVER_ADDRERSS
    char* server_address = std::getenv("MORPHLING_SERVER_ADDRESS");
    LOG_FATAL_IF(server_address == nullptr,
                 "MORPHLING_SERVER_ADDRESS is not set");
    auto channel =
        grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    LOG_WARN_IF(kMemoryManagerClient != nullptr,
                "MemoryManagerClient is already initialized");
    if (kMemoryManagerClient == nullptr) {
      kMemoryManagerClient = std::make_unique<MemoryManagerClient>(channel);
    }
  });
}