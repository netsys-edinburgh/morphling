#include "client.h"

#include <uuid/uuid.h>

#include "memory/shared_memory.h"

std::unique_ptr<MemoryManagerClient> kMemoryManagerClient = nullptr;

void MemoryManagerClient::ScheduleGemmSync(void* a, void* b, void* c,
                                           void* task) {
  morphling::ScheduleGemmRequest request;
  SET_SHM_INFO_REPEAT(request, a)
  SET_SHM_INFO_REPEAT(request, b)
  SET_SHM_INFO_REPEAT(request, c)
  SET_SHM_INFO(request, task)

  char uuid_str[37];
  uuid_t bin_uuid;
  uuid_generate_random(bin_uuid);
  uuid_unparse(bin_uuid, uuid_str);

  request.set_task_id(uuid_str);

  morphling::ScheduleGemmResponse response;
  grpc::ClientContext context;
  auto status = stub_->ScheduleGemmSync(&context, request, &response);

  LOG_FATAL_IF(!status.ok(), "Failed to schedule gemm task");
  LOG_FATAL_IF(response.rsp().code() != 0, "Failed to schedule gemm task: %s",
               response.rsp().message().c_str());
}

ParamShmMap MemoryManagerClient::GetModelParam() {
  morphling::ModelParamRequest request;
  morphling::ModelParamResponse response;
  grpc::ClientContext context;
  stub_->GetModelParam(&context, request, &response);

  ParamShmMap param_map;
  for (const auto& param : response.param_info()) {
    auto shm_name = param.shm_name();
    auto size = param.size();
    auto param_name = param.param_name();
    param_map[param_name] = {shm_name, size};
    shm_map_[param_name] = OpenSharedMemory(shm_name.c_str(), size);
  }

  return param_map;
}

// void MemoryManagerClient::SetTensorShm(torch::Tensor& tensor,
//                                        const std::string& name) {
//   auto size = tensor.numel() * tensor.element_size();
//   tensor = torch::from_blob(shm_map_[name], tensor.sizes(), tensor.strides(),
//                             DoNothingDeleter<void>{}, tensor.options());
// }