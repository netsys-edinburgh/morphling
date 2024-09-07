#include <torch/extension.h>

#include "archer_tensor_handle.h"
#include "checkpoint_handle.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ArcherTensorHandle>(m, "tensor_handle")
      .def(py::init<const std::string&>())
      .def("is_tensor_index_initialized",
           &ArcherTensorHandle::IsTensorIndexInitialized,
           "Check if tensor index is initialized")
      .def("offload_tensor", &ArcherTensorHandle::OffloadTensor,
           "Offload tensor to disk")
      .def("is_tensor_offloaded", &ArcherTensorHandle::IsTensorOffloaded,
           "Check if tensor is offloaded");

  py::class_<CheckpointHandle>(m, "CheckpointHandle")
      .def(py::init<const std::string&>())
      .def("read_checkpoint", &CheckpointHandle::ReadCheckpoint,
           "Read checkpoint from disk");
}