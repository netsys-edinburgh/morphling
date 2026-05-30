#include <torch/extension.h>

#include "checkpoint/archer_tensor_handle.h"
#include "memory/shared_memory.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "set_tensor_shm",
      [](torch::Tensor& tensor, std::string& name, size_t size) {
        void* ptr = OpenSharedMemory(name.c_str(), size);
        LOG_DEBUG << "set_tensor_shm: name: " << name << ", size: " << size
                  << ", ptr: " << ptr;
        tensor.set_data(torch::from_blob(ptr, tensor.sizes(), tensor.strides(),
                                         DoNothingDeleter<void>{},
                                         tensor.options()));
      },
      "Set tensor shared memory");

  pybind11::class_<ArcherTensorHandle>(m, "ArcherTensorHandle")
      .def(pybind11::init<const std::string&>(), pybind11::arg("prefix"))
      .def("is_tensor_index_initialized",
           &ArcherTensorHandle::IsTensorIndexInitialized)
      .def("is_tensor_offloaded", &ArcherTensorHandle::IsTensorOffloaded,
           pybind11::arg("tensor_id"))
      .def("offload_tensor", &ArcherTensorHandle::OffloadTensor,
           pybind11::arg("tensor"), pybind11::arg("tensor_id"));
}
