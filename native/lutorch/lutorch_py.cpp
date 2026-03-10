#include <pybind11/pybind11.h>

// Forward declaration; implementation lives in lutorch.cu
void PB_LUTorchManager(pybind11::module& m);

// Entry point for the standalone `lutorch_cuda` extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    PB_LUTorchManager(m);
}

