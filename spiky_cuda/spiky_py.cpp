#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void PB_SPNetDataManagerF(py::module& m);
void PB_ANDNDataManagerF(py::module& m);
#ifdef BUILD_INTEGERS_VERSION
void PB_SPNetDataManagerI(py::module& m);
void PB_ANDNDataManagerI(py::module& m);
#endif
void PB_SynapseGrowthLowLevelEngine(py::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Spiky Module";
    PB_SPNetDataManagerF(m);
    PB_ANDNDataManagerF(m);
    #ifdef BUILD_INTEGERS_VERSION
    PB_SPNetDataManagerI(m);
    PB_ANDNDataManagerI(m);
    #endif
    PB_SynapseGrowthLowLevelEngine(m);
}
