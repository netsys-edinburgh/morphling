#pragma once

#include <grpcpp/grpcpp.h>

#include "morphling.grpc.pb.h"
#include "morphling.pb.h"
#include "utils/noncopyable.h"

// class MemoryManagerClient : public noncopyable {
//  public:
//   MemoryManagerClient(std::string server_address) : stub_(nullptr) {
//     grpc::ChannelArguments args;
//     args.SetMaxReceiveMessageSize(-1);
//     auto channel = grpc::CreateCustomChannel(
//         server_address, grpc::InsecureChannelCredentials(), args);
//     stub_ = morphling::MemoryManager::NewStub(channel);
//   }

//   void LoadParamAsync(const morphling::LoadParamRequest* request,
//                       morphling::LoadParamResponse* response) {
//     grpc::ClientContext context;
//     stub_->LoadParamAsync(&context, *request, response);
//   }

//   void ConfirmParamLoaded(const morphling::LoadParamRequest* request,
//                           morphling::LoadParamResponse* response) {
//     grpc::ClientContext context;
//     stub_->ConfirmParamLoaded(&context, *request, response);
//   }

//  private:
//   std::unique_ptr<morphling::MemoryManager::Stub> stub_;
// };

// extern std::unique_ptr<MemoryManagerClient> kMemoryManagerClient;

// inline void InitMemoryManagerClient(std::shared_ptr<grpc::Channel> channel) {
//   // run only once, thread-safe
//   if (kMemoryManagerClient == nullptr) {
//     kMemoryManagerClient = std::make_unique<MemoryManagerClient>(channel);
//   }
// }