#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <string>

#include "backend/proxy_cli.h"
#include "backend/proxy_svr.h"
#include "core/pytorch_defs.h"
#include "scheduler/amqp_dispatcher.h"
#include "scheduler/amqp_worker.h"

namespace py = pybind11;

using morphling::backend::ProxyCli;
using morphling::backend::ProxySvr;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<DeviceMode>(m, "DeviceMode")
      .value("BARRIER", DeviceMode::BARRIER)
      .value("DYNAMIC", DeviceMode::DYNAMIC)
      .export_values();

  py::class_<AMQPBackend>(m, "AMQPBackend")
      .def(py::init<const std::string&, uint32_t>())
      .def("sync_dispatch_matmul", &AMQPBackend::DispatchMatMul,
           "Dispatch matmul to devices");

  py::class_<ProxySvr>(m, "ProxySvr")
      .def(py::init<>())
      .def("initialize", &ProxySvr::Initialize)
      .def("start", &ProxySvr::Start)
      .def("async_dispatch_matmul", &ProxySvr::DispatchMatMulAsync)
      .def("wait_matmul",
           [](ProxySvr& self, int oid) {
             py::gil_scoped_release release;
             return self.WaitMatMul(oid);
           })
      .def("get_connection_count", &ProxySvr::GetConnectionCount)
      .def("flush_perf_log", &ProxySvr::FlushPerfLog)
      .def("is_barrier_met", &ProxySvr::IsBarrierMet)
      .def("get_queue_size", &ProxySvr::GetQueueSize)
      .def("get_device_mode", &ProxySvr::GetDeviceMode);

  py::class_<ProxyCli>(m, "ProxyCli")
      .def(py::init<>())
      .def("initialize", &ProxyCli::Initialize)
      .def("start", &ProxyCli::Start);

  // define a function equal to torch::matmul
  m.def(
      "_custom_matmul",
      [](MatMulRequestMessage& a, int gid) {
        cudaSetDevice(gid);

        cublasHandle_t handle;
        cublasCreate(&handle);

        auto a_shape = a.mat_shape[0];
        auto b_shape = a.mat_shape[1];

        auto* a_ptr = a.mat_ptr[0];
        auto* b_ptr = a.mat_ptr[1];

        float alpha = 1.0f;
        float beta = 0.0f;

        int64_t m = std::get<0>(a_shape);
        int64_t k = std::get<1>(a_shape);
        int64_t n = std::get<1>(b_shape);

        void* d_A;
        void* d_B;
        void* d_C;

        cudaMalloc(&d_B, m * k * sizeof(float));
        cudaMalloc(&d_B, k * n * sizeof(float));
        cudaMalloc(&d_C, m * n * sizeof(float));

        cudaMemcpy(d_B, a_ptr, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, b_ptr, k * n * sizeof(float), cudaMemcpyHostToDevice);

        // memory is row major, cublas is column major
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    reinterpret_cast<float*>(d_B), n,
                    reinterpret_cast<float*>(d_A), k, &beta,
                    reinterpret_cast<float*>(d_C), n);

        auto result = torch::from_blob(d_C, {m, n});

        cudaFree(d_B);
        cudaFree(d_B);
        return result.to(torch::kCPU);
      },
      "Matrix multiplication");
  //   m.def(
  //       "_custom_matmul",
  //       [](torch::Tensor& a, torch::Tensor& b, int gid) {
  //         std::cout << "a: " << a << std::endl;
  //         std::cout << "b: " << b << std::endl;

  //         // auto a_contiguous = a.contiguous();
  //         // auto b_contiguous = b.contiguous();

  //         void* a_ptr = a.data_ptr();
  //         void* b_ptr = b.data_ptr();

  //         void* d_B;
  //         void* d_B;

  //         cudaMalloc(&d_B, a.numel() * sizeof(float));
  //         cudaMalloc(&d_B, a.numel() * sizeof(float));

  //         cudaMemcpy(d_B, a_ptr, a.numel() * sizeof(float),
  //                    cudaMemcpyHostToDevice);
  //         cudaMemcpy(d_B, b_ptr, b.numel() * sizeof(float),
  //                    cudaMemcpyHostToDevice);

  //         auto gpu_a = torch::from_blob(d_B, a.sizes(),
  //                                       a.options().device(torch::kCUDA,
  //                                       gid));
  //         auto gpu_b = torch::from_blob(d_B, b.sizes(),
  //                                       b.options().device(torch::kCUDA,
  //                                       gid));

  //         std::cout << "gpu_a: " << gpu_a << std::endl;
  //         std::cout << "gpu_b: " << gpu_b << std::endl;

  //         // perform matrix multiplication
  //         auto result = torch::matmul(gpu_a, gpu_b);

  //         std::cout << "result: " << result << std::endl;

  //         cudaFree(d_B);
  //         cudaFree(d_B);

  //         return result.to(torch::kCPU);
  //       },
  //       "Matrix multiplication");

  py::class_<AMQPWorker>(m, "AMQPWorker")
      .def(py::init<const std::string&, uint32_t>())
      .def("handle_req", &AMQPWorker::HandleReq, "Handle request message");

  py::class_<MatMulRequestMessage>(m, "MatMulRequestMessage")
      .def(py::init<>())
      .def_readwrite("row", &MatMulRequestMessage::row)
      .def_readwrite("col", &MatMulRequestMessage::col)
      .def_readwrite("ld", &MatMulRequestMessage::ld)
      .def_readwrite("mat", &MatMulRequestMessage::mat)
      .def_readwrite("mat_ptr", &MatMulRequestMessage::mat_ptr)
      .def_readwrite("mat_shape", &MatMulRequestMessage::mat_shape)
      // convert std::string to python bytes
      .def("set_mat", &MatMulRequestMessage::SetMat)
      .def("Serialize",
           [](MatMulRequestMessage& msg) { return py::bytes(msg.Serialize()); })
      .def("Deserialize", &MatMulRequestMessage::Deserialize);

  py::class_<MatMulResponseMessage>(m, "MatMulResponseMessage")
      .def(py::init<>())
      .def_readwrite("row", &MatMulResponseMessage::row)
      .def_readwrite("col", &MatMulResponseMessage::col)
      .def_readwrite("ld", &MatMulResponseMessage::ld)
      .def_readwrite("mat", &MatMulResponseMessage::mat)
      .def(
          "Serialize",
          [](MatMulResponseMessage& msg) { return py::bytes(msg.Serialize()); })
      .def("Deserialize", &MatMulResponseMessage::Deserialize);
}
