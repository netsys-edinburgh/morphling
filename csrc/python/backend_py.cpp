#include <torch/extension.h>

#include "backend/mqtt_server.h"
#include "backend/mqtt_worker.h"
#include "backend/proxy_cli.h"
#include "backend/proxy_svr.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ProxySvr>(m, "ProxySvr")
      .def(py::init<>())
      .def("initialize", &ProxySvr::Initialize)
      .def("start", &ProxySvr::Start)
      .def("async_dispatch_matmul", &ProxySvr::DispatchMatMulAsync)
      .def("wait_matmul", &ProxySvr::WaitMatMul);

  py::class_<ProxyCli>(m, "ProxyCli")
      .def(py::init<>())
      .def("initialize", &ProxyCli::Initialize)
      .def("start", &ProxyCli::Start);

  py::class_<MQTTWorker>(m, "MQTTWorker")
      .def(py::init<const std::string&>())
      //   .def(py::init<const std::unordered_map<std::string, uint64_t>&>())
      //   .def("subscribe", &MQTTWorker::Subscribe, "Handle request message")
      .def("start", &MQTTWorker::Start, "Start MQTT worker");

  py::class_<MQTTServer>(m, "MQTTServer")
      .def(py::init<int64_t>())
      .def(py::init<>())
      //   .def("subscribe", &MQTTServer::Subscribe, "Handle request message")
      .def("start", &MQTTServer::Start, "Start MQTT server")
      .def("async_dispatch_matmul", &MQTTServer::DispatchMatMulAsync,
           "Dispatch matmul to devices")
      .def("wait_matmul", &MQTTServer::WaitMatMul, "Wait for matmul response");
}
