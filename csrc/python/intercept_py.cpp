#include <torch/extension.h>

#include "intercept/interceptor.h"

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sgemm_", &sgemm_,
// "sgemm_"); }