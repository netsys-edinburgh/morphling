#include <torch/extension.h>

#include "intercept/client.h"
#include "intercept/interceptor.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<MemoryManagerClient>(m, "MemoryManagerClient")
      .def(py::init<>())
      .def("get_model_param", &MemoryManagerClient::GetModelParam,
           "Get model parameter from server");

  m.def("sgemm_", &sgemm_, "sgemm_");
}