#include <torch/extension.h>

#include "checkpoint/archer_tensor_handle.h"
#include "checkpoint/checkpoint_handle.h"
#include "intercept/client.h"
#include "memory/shared_memory.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ArcherTensorHandle>(m, "ArcherTensorHandle")
      .def(py::init<const std::string&>())
      .def("is_tensor_index_initialized",
           &ArcherTensorHandle::IsTensorIndexInitialized,
           "Check if tensor index is initialized")
      .def("offload_tensor", &ArcherTensorHandle::OffloadTensor,
           "Offload tensor to disk")
      .def("is_tensor_offloaded", &ArcherTensorHandle::IsTensorOffloaded,
           "Check if tensor is offloaded");
  m.def(
      "set_tensor_shm",
      [](torch::Tensor& tensor, std::string& name, size_t size) {
        void* ptr = OpenSharedMemory(name.c_str(), size);
        // deleter does nothing since memory is managed by emulator shared
        // memory
        LOG_DEBUG("Set tensor shm: name: {}, size: {}, ptr: {}", name, size,
                  ptr);
        tensor = torch::from_blob(ptr, tensor.sizes(), tensor.strides(),
                                  DoNothingDeleter<void>{}, tensor.options());
      },
      "Set tensor shared memory");

  py::class_<MemoryManagerClient>(m, "MemoryManagerClient")
      .def(py::init<>())
      .def("get_model_param", &MemoryManagerClient::GetModelParam,
           "Get model parameter from server");
  //   py::class_<CheckpointHandle>(m, "CheckpointHandle")
  //       .def(py::init<const std::string&>())
  //       .def("read_checkpoint", &CheckpointHandle::ReadCheckpoint,
  //            "Read checkpoint from disk");
}