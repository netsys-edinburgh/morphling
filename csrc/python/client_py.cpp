#include <torch/extension.h>

#include "intercept/client.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<MemoryManagerClient>(m, "MemoryManagerClient")
      .def(py::init<>())
      .def("get_model_param", &MemoryManagerClient::GetModelParam,
           "Get model parameter from server")
      //   .def("schedule_gemm_sync", &MemoryManagerClient::ScheduleGemmSync,
      //        "Schedule gemm task on server")
      .def("set_tensor_shm", &MemoryManagerClient::SetTensorShm,
           "Set tensor shared memory");
}