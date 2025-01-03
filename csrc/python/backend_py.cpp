#include <torch/extension.h>

#include "backend/proxy_cli.h"
#include "backend/proxy_svr.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ProxySvr>(m, "ProxySvr")
      .def(py::init<>())
      .def("initialize", &ProxySvr::Initialize)
      .def("start", &ProxySvr::Start)
      .def("dispatch_matmul_async", &ProxySvr::DispatchMatMulAsync)
      .def("wait_matmul", &ProxySvr::WaitMatMul);

  py::class_<ProxyCli>(m, "ProxyCli")
      .def(py::init<>())
      .def("initialize", &ProxyCli::Initialize)
      .def("start", &ProxyCli::Start);
}
