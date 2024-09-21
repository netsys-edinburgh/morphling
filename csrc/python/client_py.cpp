#include <torch/extension.h>

#include "intercept/client.h"
#include "memory/shared_memory.h"

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   py::class_<MemoryManagerClient>(m, "MemoryManagerClient")
//       .def(py::init<>())
//       .def("get_model_param", &MemoryManagerClient::GetModelParam,
//            "Get model parameter from server");
//   //   .def("schedule_gemm_sync", &MemoryManagerClient::ScheduleGemmSync,
//   //        "Schedule gemm task on server")

//   m.def(
//       "set_tensor_shm",
//       [](torch::Tensor& tensor, std::string& name, size_t size) {
//         void* ptr = OpenSharedMemory(name.c_str(), size);
//         // deleter does nothing since memory is managed by emulator shared
//         memory tensor = torch::from_blob(ptr, tensor.sizes(),
//         tensor.strides(),
//                                   DoNothingDeleter<void>{},
//                                   tensor.options());
//       },
//       "Set tensor shared memory");
// }