#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void PB_SPNetDataManagerF(py::module& m);
void PB_LUTDataManagerF(py::module& m);
void PB_LProjectionDataManagerF(py::module& m);
#ifdef BUILD_INTEGERS_VERSION
void PB_SPNetDataManagerI(py::module& m);
void PB_LUTDataManagerI(py::module& m);
void PB_LProjectionDataManagerI(py::module& m);
#endif
void PB_SynapseGrowthLowLevelEngine(py::module& m);
void PB_DenseToSparseConverter(py::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Spiky Module";
    PB_SPNetDataManagerF(m);
    PB_LUTDataManagerF(m);
    PB_LProjectionDataManagerF(m);
    #ifdef BUILD_INTEGERS_VERSION
    PB_SPNetDataManagerI(m);
    PB_LUTDataManagerI(m);
    PB_LProjectionDataManagerI(m);
    #endif
    PB_SynapseGrowthLowLevelEngine(m);
    PB_DenseToSparseConverter(m);
}
