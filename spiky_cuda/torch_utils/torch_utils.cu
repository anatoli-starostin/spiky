#include <string>
#include "../misc/misc.h"
#include "torch_utils.h"

namespace {
#include "aux/torch_utils_kernels_logic.cu"
}

namespace py = pybind11;

class __attribute__((visibility("hidden"))) DenseToSparseConverterNative {
public:
    DenseToSparseConverterNative() 
        #ifdef ENABLE_PROFILING
        : profiler(N_TORCH_UTILS_PROFILER_OPS)
        #endif
    {
        #ifdef ENABLE_PROFILING
        profiler.register_operation_type(TORCH_UTILS_DENSE_TO_SPARSE_PROFILER_OP, "torch_utils::dense_to_sparse_32");
        profiler.register_operation_type(TORCH_UTILS_COUNT_NONZERO_PROFILER_OP, "torch_utils::count_nonzero");
        #endif
    }

    ~DenseToSparseConverterNative() {
    }

    void dense_to_sparse_32(
        torch::Tensor source,
        torch::Tensor target_indices,
        torch::Tensor target_values,
        torch::Tensor counter_buffer,
        bool erase_input = false
    ) {
        PROF_START(TORCH_UTILS_DENSE_TO_SPARSE_PROFILER_OP);

        // Validate source tensor
        if (source.dim() != 1) {
            throw py::value_error("source tensor must be 1D");
        }

        if (!source.is_contiguous()) {
            throw py::value_error("source tensor must be contiguous");
        }

        // Validate target_indices tensor
        if (target_indices.dim() != 1) {
            throw py::value_error("target_indices tensor must be 1D");
        }

        if (target_indices.dtype() != torch::kInt64) {
            throw py::value_error("target_indices must be int64");
        }

        if (!target_indices.is_contiguous()) {
            throw py::value_error("target_indices tensor must be contiguous");
        }

        // Validate target_values tensor
        if (target_values.dim() != 1) {
            throw py::value_error("target_values tensor must be 1D");
        }

        if (target_values.dtype() != source.dtype()) {
            throw py::value_error("target_values must have the same dtype as source");
        }

        if (!target_values.is_contiguous()) {
            throw py::value_error("target_values tensor must be contiguous");
        }

        // Check that source and target_values are 32-bit (4 bytes)
        if (source.element_size() != 4) {
            throw py::value_error("source tensor must be 32-bit (element_size == 4)");
        }

        if ((source.numel() % 4) != 0) {
            throw py::value_error("source tensor size must be divisable by 4");
        }

        if (target_values.element_size() != 4) {
            throw py::value_error("target_values tensor must be 32-bit (element_size == 4)");
        }

        // Validate counter_buffer tensor
        if (counter_buffer.dim() != 1) {
            throw py::value_error("counter_buffer tensor must be 1D");
        }

        // Check that counter_buffer is int32
        if (counter_buffer.dtype() != torch::kInt32) {
            throw py::value_error("counter_buffer must be int32");
        }

        if (counter_buffer.numel() < 1) {
            throw py::value_error("counter_buffer must have at least 1 element");
        }

        if (!counter_buffer.is_contiguous()) {
            throw py::value_error("counter_buffer tensor must be contiguous");
        }

        // Check device compatibility
        if (source.device() != target_indices.device() || 
            source.device() != target_values.device() ||
            source.device() != counter_buffer.device()) {
            throw py::value_error("All tensors must be on the same device");
        }

        int64_t numel = source.numel();

        // Get device information
        int device = -1;
        if (source.device().is_cuda()) {
            device = source.device().index();
        }

        // Get counter pointer from buffer
        uint32_t* counter_ptr = reinterpret_cast<uint32_t*>(counter_buffer.data_ptr());

        // Initialize counter to zero
        if (device == -1) {
            *counter_ptr = 0;
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            CU_CHECK(cudaMemset(counter_ptr, 0, sizeof(uint32_t)));
            #endif
        }

        // Launch kernel
        // Process elements in quads (groups of 4)
        uint64_t n_quads = numel >> 2;
        uint32_t tpb = 1024;  // Threads per block (power of 2)
        uint32_t num_blocks = static_cast<uint32_t>((n_quads + tpb - 1) / tpb);
        dim3 numBlocks(num_blocks, 1);
        uint32_t shared_mem_size = tpb * sizeof(uint32_t);

        // Use GRID_CALL_SHARED_MEM macro - device must be in scope
        GRID_CALL_SHARED_MEM(
            numBlocks, densify, tpb, shared_mem_size,
            reinterpret_cast<int4*>(source.data_ptr()),
            n_quads,
            reinterpret_cast<int32_t*>(target_values.data_ptr()),
            reinterpret_cast<int64_t*>(target_indices.data_ptr()),
            counter_ptr,
            erase_input,
            device
        );

        PROF_END(TORCH_UTILS_DENSE_TO_SPARSE_PROFILER_OP);
    }

    int32_t count_nonzero(
        torch::Tensor source,
        torch::Tensor aux_buffer
    ) {
        PROF_START(TORCH_UTILS_COUNT_NONZERO_PROFILER_OP);

        // Validate source tensor
        if (source.dim() != 1) {
            throw py::value_error("source tensor must be 1D");
        }

        if (!source.is_contiguous()) {
            throw py::value_error("source tensor must be contiguous");
        }

        // Check that source is 32-bit (4 bytes)
        if (source.element_size() != 4) {
            throw py::value_error("source tensor must be 32-bit (element_size == 4)");
        }

        if ((source.numel() % 4) != 0) {
            throw py::value_error("source tensor size must be divisable by 4");
        }

        // Validate aux_buffer tensor
        if (aux_buffer.dim() != 1) {
            throw py::value_error("aux_buffer tensor must be 1D");
        }

        // Check that aux_buffer is int32
        if (aux_buffer.dtype() != torch::kInt32) {
            throw py::value_error("aux_buffer must be int32");
        }

        if (aux_buffer.numel() < 1) {
            throw py::value_error("aux_buffer must have at least 1 element");
        }

        if (!aux_buffer.is_contiguous()) {
            throw py::value_error("aux_buffer tensor must be contiguous");
        }

        // Check device compatibility
        if (source.device() != aux_buffer.device()) {
            throw py::value_error("All tensors must be on the same device");
        }

        int64_t numel = source.numel();

        // Get device information
        int device = -1;
        if (source.device().is_cuda()) {
            device = source.device().index();
        }

        // Get aux_buffer pointer
        uint32_t* aux_ptr = reinterpret_cast<uint32_t*>(aux_buffer.data_ptr());

        // Initialize aux_buffer to zero
        if (device == -1) {
            *aux_ptr = 0;
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            CU_CHECK(cudaMemset(aux_ptr, 0, sizeof(uint32_t)));
            #endif
        }

        // Launch count_nonzero kernel
        // Process elements in quads (groups of 4)
        uint64_t n_quads = numel >> 2;
        uint32_t tpb = 1024;  // Threads per block
        uint32_t num_blocks = static_cast<uint32_t>((n_quads + tpb - 1) / tpb);
        dim3 numBlocks(num_blocks, 1);
        uint32_t shared_mem_size = tpb * sizeof(uint32_t);

        // Count non-zero elements using count_nonzero kernel
        GRID_CALL_SHARED_MEM(
            numBlocks, count_nonzero, tpb, shared_mem_size,
            reinterpret_cast<int4*>(source.data_ptr()),
            n_quads,
            aux_ptr,
            device
        );

        // Copy result from aux_buffer and return
        uint32_t result = 0;
        if (device == -1) {
            result = *aux_ptr;
        } else {
            #ifndef NO_CUDA
            CU_CHECK(cudaMemcpy(&result, aux_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            #endif
        }
        PROF_END(TORCH_UTILS_COUNT_NONZERO_PROFILER_OP);
        return result;
    }

    std::string get_profiling_stats() {
        #ifdef ENABLE_PROFILING
        return profiler.get_stats_as_string();
        #else
        return "profiler is disabled";
        #endif
    }

private:
    #ifdef ENABLE_PROFILING
    SimpleProfiler profiler;
    #endif
};

void PB_DenseToSparseConverter(py::module& m) {
    py::class_<DenseToSparseConverterNative>(m, "DenseToSparseConverterNative")
        .def(py::init<>())
        .def("dense_to_sparse_32", &DenseToSparseConverterNative::dense_to_sparse_32,
            "Convert dense 32-bit tensor to sparse format",
            py::arg("source"),
            py::arg("target_indices"),
            py::arg("target_values"),
            py::arg("counter_buffer"),
            py::arg("erase_input") = false)
        .def("count_nonzero", &DenseToSparseConverterNative::count_nonzero,
            "Count non-zero elements in a dense 32-bit tensor",
            py::arg("source"),
            py::arg("aux_buffer"))
        .def("get_profiling_stats", &DenseToSparseConverterNative::get_profiling_stats);
}

