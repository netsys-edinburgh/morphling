#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "backend/mqtt_server.h"
#include "backend/mqtt_worker.h"
#include "backend/proxy_cli.h"
#include "backend/proxy_svr.h"

namespace py = pybind11;

using morphling::backend::ProxyCli;
using morphling::backend::ProxySvr;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<DeviceMode>(m, "DeviceMode")
      .value("BARRIER", DeviceMode::BARRIER)
      .value("DYNAMIC", DeviceMode::DYNAMIC)
      .export_values();

  py::class_<ProxySvr>(m, "ProxySvr")
      .def(py::init<>())
      .def("initialize", &ProxySvr::Initialize)
      .def("start", &ProxySvr::Start)
      .def("set_cache_enabled", &ProxySvr::SetCacheEnabled)
      .def("async_dispatch_matmul", &ProxySvr::DispatchMatMulAsync)
      .def("wait_matmul",
           [](ProxySvr& self, int oid) {
             py::gil_scoped_release release;
             return self.WaitMatMul(oid);
           })
      .def("get_connection_count", &ProxySvr::GetConnectionCount)
      .def("get_registered_device_count", &ProxySvr::GetRegisteredDeviceCount)
      .def("is_barrier_met", &ProxySvr::IsBarrierMet)
      .def("get_queue_size", &ProxySvr::GetQueueSize)
      .def("get_device_mode", &ProxySvr::GetDeviceMode);

  py::class_<ProxyCli>(m, "ProxyCli")
      .def(py::init<>())
      .def("initialize", &ProxyCli::Initialize, py::arg("cfg_file"),
           py::arg("device_id") = 0)
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
      .def(
          "wait_matmul",
          [](MQTTServer& self, int oid) {
            py::gil_scoped_release release;
            return self.WaitMatMul(oid);
          },
          "Wait for matmul response");
}
