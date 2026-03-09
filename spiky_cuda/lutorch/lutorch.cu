#include <tuple>
#include "../misc/misc.h"
#include "lutorch.h"

namespace py = pybind11;

#ifndef NO_CUDA
template <typename scalar_t>
static __device__ __forceinline__ scalar_t lutorch_abs(scalar_t v) {
    return v >= static_cast<scalar_t>(0) ? v : -v;
}

template <typename scalar_t>
static __device__ __forceinline__ scalar_t lutorch_delta(
    const scalar_t* x_ptr,
    int64_t batch_index,
    int64_t x_stride0,
    int64_t x_stride1,
    int64_t anchor_a,
    int64_t anchor_b
) {
    const int64_t base = batch_index * x_stride0;
    return x_ptr[base + anchor_a * x_stride1] - x_ptr[base + anchor_b * x_stride1];
}

template <typename scalar_t>
__global__ void anchor_pairs_lookup_forward_na1_kernel(
    const scalar_t* x_ptr,
    int64_t batch_size,
    int64_t x_stride0,
    int64_t x_stride1,
    const int64_t* anchor_pairs_a_ptr,
    const int64_t* anchor_pairs_b_ptr,
    int64_t n_tables,
    int64_t n_anchor_pairs,
    scalar_t cmp_eps,
    int64_t* lookup_indices_ptr,
    int64_t* lookup_alt_indices_ptr,
    scalar_t* lookup_alt_deltas_ptr,
    int64_t* anchor1_ids_ptr,
    int64_t* anchor2_ids_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * n_tables;
    if (linear_tid >= total) {
        return;
    }

    int64_t b = linear_tid / n_tables;
    int64_t t = linear_tid - b * n_tables;
    int64_t table_offset = t * n_anchor_pairs;

    int64_t lookup_idx = 0;
    scalar_t min_abs_delta = static_cast<scalar_t>(0);
    scalar_t min_delta = static_cast<scalar_t>(0);
    int64_t min_anchor_a = 0;
    int64_t min_anchor_b = 0;
    int64_t min_bit_pos = 0;

    for (int64_t p = 0; p < n_anchor_pairs; ++p) {
        int64_t anchor_a = anchor_pairs_a_ptr[table_offset + p];
        int64_t anchor_b = anchor_pairs_b_ptr[table_offset + p];
        scalar_t delta = lutorch_delta(x_ptr, b, x_stride0, x_stride1, anchor_a, anchor_b);

        if (delta > cmp_eps) {
            lookup_idx |= (static_cast<int64_t>(1) << p);
        }

        scalar_t abs_delta = lutorch_abs(delta);
        if ((p == 0) || (abs_delta < min_abs_delta)) {
            min_abs_delta = abs_delta;
            min_delta = delta;
            min_anchor_a = anchor_a;
            min_anchor_b = anchor_b;
            min_bit_pos = p;
        }
    }

    lookup_indices_ptr[linear_tid] = lookup_idx;
    lookup_alt_indices_ptr[linear_tid] = lookup_idx ^ (static_cast<int64_t>(1) << min_bit_pos);
    lookup_alt_deltas_ptr[linear_tid] = min_delta;
    if (anchor1_ids_ptr != nullptr) {
        anchor1_ids_ptr[linear_tid] = min_anchor_a;
        anchor2_ids_ptr[linear_tid] = min_anchor_b;
    }
}

template <typename scalar_t>
__global__ void anchor_pairs_lookup_eval_forward_na1_kernel(
    const scalar_t* x_ptr,
    int64_t batch_size,
    int64_t x_stride0,
    int64_t x_stride1,
    const int64_t* anchor_pairs_a_ptr,
    const int64_t* anchor_pairs_b_ptr,
    int64_t n_tables,
    int64_t n_anchor_pairs,
    scalar_t cmp_eps,
    int64_t* lookup_indices_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * n_tables;
    if (linear_tid >= total) {
        return;
    }

    int64_t b = linear_tid / n_tables;
    int64_t t = linear_tid - b * n_tables;
    int64_t table_offset = t * n_anchor_pairs;

    int64_t lookup_idx = 0;

    for (int64_t p = 0; p < n_anchor_pairs; ++p) {
        int64_t anchor_a = anchor_pairs_a_ptr[table_offset + p];
        int64_t anchor_b = anchor_pairs_b_ptr[table_offset + p];
        scalar_t delta = lutorch_delta(x_ptr, b, x_stride0, x_stride1, anchor_a, anchor_b);

        if (delta > cmp_eps) {
            lookup_idx |= (static_cast<int64_t>(1) << p);
        }
    }

    lookup_indices_ptr[linear_tid] = lookup_idx;
}
#endif

class SPIKY_HIDDEN LUTorchManager {
public:
    LUTorchManager()
    #ifdef ENABLE_PROFILING
        : profiler(N_LUTORCH_PROFILER_OPS)
    #endif
    {
        #ifdef ENABLE_PROFILING
        #ifndef NO_CUDA
        profiler.register_operation_type(
            LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_NA1_PROFILER_OP,
            "lutorch::anchor_pairs_lookup_forward_na1"
        );
        profiler.register_operation_type(
            LUTORCH_MANAGER_ANCHOR_PAIRS_EVAL_FORWARD_NA1_PROFILER_OP,
            "lutorch::anchor_pairs_lookup_eval_forward_no_alternatives"
        );
        #endif
        #endif
    }

    ~LUTorchManager() {
    }

#ifndef NO_CUDA
    py::tuple
    anchor_pairs_lookup_forward_na1(
        const torch::Tensor& x,
        const torch::Tensor& anchor_pairs_a,
        const torch::Tensor& anchor_pairs_b,
        double cmp_eps,
        bool save_anchor_ids = true,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_NA1_PROFILER_OP);

        if (x.dim() != 2) {
            throw py::value_error("x must be 2D [batch_size, input_dim]");
        }
        if (!x.is_cuda()) {
            throw py::value_error("x must be CUDA tensor");
        }
        if (!x.is_floating_point()) {
            throw py::value_error("x must be floating point tensor");
        }

        if (anchor_pairs_a.dim() != 2 || anchor_pairs_b.dim() != 2) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be 2D [n_tables, n_anchor_pairs]");
        }
        if (anchor_pairs_a.sizes() != anchor_pairs_b.sizes()) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must have the same shape");
        }
        if (anchor_pairs_a.dtype() != torch::kInt64 || anchor_pairs_b.dtype() != torch::kInt64) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be int64");
        }
        if (!anchor_pairs_a.is_contiguous() || !anchor_pairs_b.is_contiguous()) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be contiguous");
        }
        if (!anchor_pairs_a.is_cuda() || !anchor_pairs_b.is_cuda()) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be CUDA tensors");
        }

        if (x.device() != anchor_pairs_a.device() ||
            x.device() != anchor_pairs_b.device()) {
            throw py::value_error("All tensors must be on the same CUDA device");
        }

        const int64_t batch_size = x.size(0);
        const int64_t x_stride0 = x.stride(0);
        const int64_t x_stride1 = x.stride(1);
        const int64_t n_tables = anchor_pairs_a.size(0);
        const int64_t n_anchor_pairs = anchor_pairs_a.size(1);
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }

        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto opts_x = torch::TensorOptions().dtype(x.dtype()).device(x.device());

        torch::Tensor lookup_indices = torch::empty({batch_size, n_tables}, opts_i64);
        torch::Tensor lookup_alt_indices = torch::empty({batch_size, n_tables, 1}, opts_i64);
        torch::Tensor lookup_alt_deltas = torch::empty({batch_size, n_tables, 1}, opts_x);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_tables;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        torch::Tensor anchor1_ids;
        torch::Tensor anchor2_ids;
        int64_t* anchor1_ids_ptr = nullptr;
        int64_t* anchor2_ids_ptr = nullptr;
        if (save_anchor_ids) {
            anchor1_ids = torch::empty({batch_size, n_tables, 1}, opts_i64);
            anchor2_ids = torch::empty({batch_size, n_tables, 1}, opts_i64);
            anchor1_ids_ptr = reinterpret_cast<int64_t*>(anchor1_ids.data_ptr());
            anchor2_ids_ptr = reinterpret_cast<int64_t*>(anchor2_ids.data_ptr());
        }
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_forward_na1_kernel", [&] {
            anchor_pairs_lookup_forward_na1_kernel<scalar_t><<<blocks, threads>>>(
                reinterpret_cast<const scalar_t*>(x.data_ptr()),
                batch_size,
                x_stride0,
                x_stride1,
                reinterpret_cast<const int64_t*>(anchor_pairs_a.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor_pairs_b.data_ptr()),
                n_tables,
                n_anchor_pairs,
                static_cast<scalar_t>(cmp_eps),
                reinterpret_cast<int64_t*>(lookup_indices.data_ptr()),
                reinterpret_cast<int64_t*>(lookup_alt_indices.data_ptr()),
                reinterpret_cast<scalar_t*>(lookup_alt_deltas.data_ptr()),
                anchor1_ids_ptr,
                anchor2_ids_ptr
            );
        });
        CU_CHECK(cudaGetLastError());
        py::tuple out(5);
        out[0] = lookup_indices;
        out[1] = lookup_alt_indices;
        out[2] = lookup_alt_deltas;
        if(save_anchor_ids) {
            out[3] = anchor1_ids;
            out[4] = anchor2_ids;
        } else {
            out[3] = py::none();
            out[4] = py::none();
        }
        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_NA1_PROFILER_OP);
        return out;
    }

    torch::Tensor
    anchor_pairs_lookup_eval_forward_na1(
        const torch::Tensor& x,
        const torch::Tensor& anchor_pairs_a,
        const torch::Tensor& anchor_pairs_b,
        double cmp_eps,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_EVAL_FORWARD_NA1_PROFILER_OP);

        if (x.dim() != 2) {
            throw py::value_error("x must be 2D [batch_size, input_dim]");
        }
        if (!x.is_cuda()) {
            throw py::value_error("x must be CUDA tensor");
        }
        if (!x.is_floating_point()) {
            throw py::value_error("x must be floating point tensor");
        }

        if (anchor_pairs_a.dim() != 2 || anchor_pairs_b.dim() != 2) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be 2D [n_tables, n_anchor_pairs]");
        }
        if (anchor_pairs_a.sizes() != anchor_pairs_b.sizes()) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must have the same shape");
        }
        if (anchor_pairs_a.dtype() != torch::kInt64 || anchor_pairs_b.dtype() != torch::kInt64) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be int64");
        }
        if (!anchor_pairs_a.is_contiguous() || !anchor_pairs_b.is_contiguous()) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be contiguous");
        }
        if (!anchor_pairs_a.is_cuda() || !anchor_pairs_b.is_cuda()) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b must be CUDA tensors");
        }

        if (x.device() != anchor_pairs_a.device() ||
            x.device() != anchor_pairs_b.device()) {
            throw py::value_error("All tensors must be on the same CUDA device");
        }

        const int64_t batch_size = x.size(0);
        const int64_t x_stride0 = x.stride(0);
        const int64_t x_stride1 = x.stride(1);
        const int64_t n_tables = anchor_pairs_a.size(0);
        const int64_t n_anchor_pairs = anchor_pairs_a.size(1);
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }

        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(x.device());

        torch::Tensor lookup_indices = torch::empty({batch_size, n_tables}, opts_i64);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);

        int64_t total = batch_size * n_tables;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_eval_forward_na1_kernel", [&] {
            anchor_pairs_lookup_eval_forward_na1_kernel<scalar_t><<<blocks, threads>>>(
                reinterpret_cast<const scalar_t*>(x.data_ptr()),
                batch_size,
                x_stride0,
                x_stride1,
                reinterpret_cast<const int64_t*>(anchor_pairs_a.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor_pairs_b.data_ptr()),
                n_tables,
                n_anchor_pairs,
                static_cast<scalar_t>(cmp_eps),
                reinterpret_cast<int64_t*>(lookup_indices.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_EVAL_FORWARD_NA1_PROFILER_OP);
        return lookup_indices;
    }
#endif

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

void PB_LUTorchManager(py::module& m) {
    py::class_<LUTorchManager>(m, "LUTorchManager")
        .def(py::init<>())
        #ifndef NO_CUDA
        .def(
            "anchor_pairs_lookup_forward_na1",
            &LUTorchManager::anchor_pairs_lookup_forward_na1,
            py::arg("x"),
            py::arg("anchor_pairs_a"),
            py::arg("anchor_pairs_b"),
            py::arg("cmp_eps"),
            py::arg("save_anchor_ids") = true,
            py::arg("threads_per_block") = 256
        )
        .def(
            "anchor_pairs_lookup_eval_forward_na1",
            &LUTorchManager::anchor_pairs_lookup_eval_forward_na1,
            py::arg("x"),
            py::arg("anchor_pairs_a"),
            py::arg("anchor_pairs_b"),
            py::arg("cmp_eps"),
            py::arg("threads_per_block") = 256
        )
        #endif
        .def("get_profiling_stats", &LUTorchManager::get_profiling_stats);
}
