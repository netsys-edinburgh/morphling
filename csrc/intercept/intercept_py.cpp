#include <torch/extension.h>

namespace py = pybind11;

// define pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "multiplication",
      [](const torch::Tensor& t1, const torch::Tensor& t2) {
        return t1.matmul(t2);
      },
      "Matrix multiplication");
}