#include <torch/extension.h>

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
      "Set tensor shared memory from a named POSIX shm segment");
}
