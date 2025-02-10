#include "client.h"

#include "common/generator.h"
#include "common/rttr_registration.h"
#include "memory/shared_memory.h"

std::unique_ptr<MemoryManagerClient> kMemoryManagerClient = nullptr;

void MemoryManagerClient::ScheduleGemmSync(void* a, void* b, void* c,
                                           void* task) {
  morphling::ScheduleGemmRequest request;
  SET_SHM_INFO_REPEAT(request, a)
  SET_SHM_INFO_REPEAT(request, b)
  SET_SHM_INFO_REPEAT(request, c)
  SET_SHM_INFO(request, task)

  auto task_id = GenUUID();
  request.set_task_id(task_id);

  LOG_DEBUG("ScheduleGemmSync: task_id: {}", task_id);

  morphling::ScheduleGemmResponse response;
  grpc::ClientContext context;
  auto status = stub_->ScheduleGemmSync(&context, request, &response);

  LOG_FATAL_IF(!status.ok(), "Failed to schedule gemm task {}", task_id);
  LOG_FATAL_IF(response.rsp().code() != 0,
               "Failed to schedule gemm task: {} {}", task_id,
               response.rsp().message().c_str());
}

std::unordered_map<std::string, std::tuple<std::string, size_t>>
MemoryManagerClient::GetModelParam() {
  morphling::ModelParamRequest request;
  morphling::ModelParamResponse response;
  grpc::ClientContext context;
  stub_->GetModelParam(&context, request, &response);

  std::unordered_map<std::string, std::tuple<std::string, size_t>> param_map;
  for (const auto& param : response.param_info()) {
    auto shm_name = param.shm_name();
    auto size = param.size();
    auto param_name = param.param_name();

    // void* ptr = OpenSharedMemory(shm_name.c_str(), size);
    // param_map[param_name] = {
    //     .id = -1, .ptr = ptr, .size = size, .name = shm_name};
    param_map[param_name] = std::make_tuple(shm_name, size);
    // shm_map_[param_name] = ptr;
  }

  return param_map;
}

// void MemoryManagerClient::SetTensorShm(torch::Tensor& tensor,
//                                        const std::string& name) {
//   auto size = tensor.numel() * tensor.element_size();
//   tensor = torch::from_blob(shm_map_[name], tensor.sizes(), tensor.strides(),
//                             DoNothingDeleter<void>{}, tensor.options());
// }
