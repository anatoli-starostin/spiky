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
static __device__ __forceinline__ scalar_t lutorch_sign(scalar_t v) {
    if (v > static_cast<scalar_t>(0)) {
        return static_cast<scalar_t>(1);
    }
    if (v < static_cast<scalar_t>(0)) {
        return static_cast<scalar_t>(-1);
    }
    return static_cast<scalar_t>(0);
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

template <typename scalar_t>
__global__ void anchor_pairs_lookup_backward_na1_kernel(
    int64_t total,
    const int64_t* anchor1_ids_ptr,
    const int64_t* anchor2_ids_ptr,
    const scalar_t* lookup_alt_deltas_ptr,
    const int64_t* batch_offset_ptr,
    const scalar_t* grad_main_ptr,
    const scalar_t* grad_alt_ptr,
    int64_t grad_main_stride0,
    int64_t grad_main_stride1,
    int64_t batch_size,
    int64_t n_tables,
    bool inv_l1,
    scalar_t* x_grad_flat_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= total) {
        return;
    }

    int64_t bt = linear_tid;
    int64_t b = bt / n_tables;
    int64_t t = bt - b * n_tables;

    scalar_t delta = lookup_alt_deltas_ptr[linear_tid];
    scalar_t minus_uncertainty_derivative = static_cast<scalar_t>(0);
    if (inv_l1) {
        scalar_t one_plus_abs = static_cast<scalar_t>(1) + lutorch_abs(delta);
        minus_uncertainty_derivative =
            static_cast<scalar_t>(0.5) * lutorch_sign(delta) / (one_plus_abs * one_plus_abs);
    } else {
        scalar_t one_plus_sq = static_cast<scalar_t>(1) + delta * delta;
        minus_uncertainty_derivative = delta / (one_plus_sq * one_plus_sq);
    }

    scalar_t grad_main = grad_main_ptr[b * grad_main_stride0 + t * grad_main_stride1];
    scalar_t grad_alt = grad_alt_ptr[linear_tid];
    scalar_t du = (grad_main - grad_alt) * minus_uncertainty_derivative;

    int64_t idx1 = batch_offset_ptr[linear_tid] + anchor1_ids_ptr[linear_tid];
    int64_t idx2 = batch_offset_ptr[linear_tid] + anchor2_ids_ptr[linear_tid];
    atomicAdd(x_grad_flat_ptr + idx1, du);
    atomicAdd(x_grad_flat_ptr + idx2, -du);
}

template <typename scalar_t>
__global__ void lprojection_backward_na1_nonsmooth_weights_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const int64_t* table_indices_flat_ptr,
    const int64_t* lookup_indices_flat_ptr,
    int64_t grad_output_stride0,
    int64_t grad_output_stride1,
    int64_t grad_output_stride2,
    scalar_t* weights_grad_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = total_bt * n_outputs;
    if (linear_tid >= total) {
        return;
    }
    int64_t bt = linear_tid / n_outputs;
    int64_t o = linear_tid - bt * n_outputs;
    int64_t b = bt / n_tables;
    int64_t t = bt - b * n_tables;
    scalar_t g = grad_output_ptr[b * grad_output_stride0 + t * grad_output_stride1 + o * grad_output_stride2];
    int64_t table = table_indices_flat_ptr[bt];
    int64_t entry = lookup_indices_flat_ptr[bt];
    int64_t widx = (table * n_entries + entry) * n_outputs + o;
    atomicAdd(weights_grad_ptr + widx, g);
}

template <typename scalar_t>
__global__ void lprojection_backward_na1_smooth_weights_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const int64_t* table_indices_flat_ptr,
    const int64_t* lookup_indices_flat_ptr,
    const int64_t* table_indices_alt_flat_ptr,
    const int64_t* lookup_alt_indices_flat_ptr,
    const scalar_t* main_weight_flat_ptr,
    const scalar_t* alt_weight_flat_ptr,
    int64_t grad_output_stride0,
    int64_t grad_output_stride1,
    int64_t grad_output_stride2,
    scalar_t* weights_grad_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = total_bt * n_outputs;
    if (linear_tid >= total) {
        return;
    }
    int64_t bt = linear_tid / n_outputs;
    int64_t o = linear_tid - bt * n_outputs;
    int64_t b = bt / n_tables;
    int64_t t = bt - b * n_tables;
    scalar_t g = grad_output_ptr[b * grad_output_stride0 + t * grad_output_stride1 + o * grad_output_stride2];
    scalar_t g_main = g * main_weight_flat_ptr[bt];
    scalar_t g_alt = g * alt_weight_flat_ptr[bt];

    int64_t table_main = table_indices_flat_ptr[bt];
    int64_t entry_main = lookup_indices_flat_ptr[bt];
    int64_t widx_main = (table_main * n_entries + entry_main) * n_outputs + o;
    atomicAdd(weights_grad_ptr + widx_main, g_main);

    int64_t table_alt = table_indices_alt_flat_ptr[bt];
    int64_t entry_alt = lookup_alt_indices_flat_ptr[bt];
    int64_t widx_alt = (table_alt * n_entries + entry_alt) * n_outputs + o;
    atomicAdd(weights_grad_ptr + widx_alt, g_alt);
}

template <typename scalar_t>
__global__ void lprojection_backward_na1_carriers_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const scalar_t* weights_ptr,
    const int64_t* table_indices_flat_ptr,
    const int64_t* lookup_indices_flat_ptr,
    const int64_t* table_indices_alt_flat_ptr,
    const int64_t* lookup_alt_indices_flat_ptr,
    int64_t grad_output_stride0,
    int64_t grad_output_stride1,
    int64_t grad_output_stride2,
    scalar_t* lookup_indices_grad_c_grad_ptr,
    scalar_t* lookup_alt_indices_grad_c_grad_ptr
) {
    int64_t bt = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bt >= total_bt) {
        return;
    }
    int64_t b = bt / n_tables;
    int64_t t = bt - b * n_tables;
    int64_t table_main = table_indices_flat_ptr[bt];
    int64_t entry_main = lookup_indices_flat_ptr[bt];
    int64_t table_alt = table_indices_alt_flat_ptr[bt];
    int64_t entry_alt = lookup_alt_indices_flat_ptr[bt];
    scalar_t acc_main = static_cast<scalar_t>(0);
    scalar_t acc_alt = static_cast<scalar_t>(0);
    for (int64_t o = 0; o < n_outputs; ++o) {
        scalar_t g = grad_output_ptr[b * grad_output_stride0 + t * grad_output_stride1 + o * grad_output_stride2];
        int64_t widx_main = (table_main * n_entries + entry_main) * n_outputs + o;
        int64_t widx_alt = (table_alt * n_entries + entry_alt) * n_outputs + o;
        acc_main += g * weights_ptr[widx_main];
        acc_alt += g * weights_ptr[widx_alt];
    }
    lookup_indices_grad_c_grad_ptr[bt] = acc_main;
    lookup_alt_indices_grad_c_grad_ptr[bt] = acc_alt;
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
        profiler.register_operation_type(
            LUTORCH_MANAGER_ANCHOR_PAIRS_BACKWARD_NA1_PROFILER_OP,
            "lutorch::anchor_pairs_lookup_backward_na1"
        );
        profiler.register_operation_type(
            LUTORCH_MANAGER_LPROJECTION_BACKWARD_NA1_PROFILER_OP,
            "lutorch::lprojection_backward_na1"
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

    torch::Tensor
    anchor_pairs_lookup_backward_na1(
        const torch::Tensor& x,
        const torch::Tensor& anchor1_ids,
        const torch::Tensor& anchor2_ids,
        const torch::Tensor& lookup_alt_deltas,
        const torch::Tensor& batch_offset,
        const torch::Tensor& grad_main,
        const torch::Tensor& grad_alt,
        bool inv_l1,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_BACKWARD_NA1_PROFILER_OP);

        if (x.dim() != 2) {
            throw py::value_error("x must be 2D [batch_size, input_dim]");
        }
        if (!x.is_cuda()) {
            throw py::value_error("x must be CUDA tensor");
        }
        if (!x.is_floating_point()) {
            throw py::value_error("x must be floating point tensor");
        }
        if (anchor1_ids.dtype() != torch::kInt64 || anchor2_ids.dtype() != torch::kInt64) {
            throw py::value_error("anchor1_ids and anchor2_ids must be int64");
        }
        if (batch_offset.dtype() != torch::kInt64) {
            throw py::value_error("batch_offset must be int64");
        }
        if (grad_main.dtype() != x.dtype() || grad_alt.dtype() != x.dtype()) {
            throw py::value_error("grad_main and grad_alt must have the same dtype as x");
        }
        if (lookup_alt_deltas.dtype() != x.dtype()) {
            throw py::value_error("lookup_alt_deltas must have the same dtype as x");
        }
        if (anchor1_ids.device() != x.device() || anchor2_ids.device() != x.device() ||
            lookup_alt_deltas.device() != x.device() || batch_offset.device() != x.device() ||
            grad_main.device() != x.device() || grad_alt.device() != x.device()) {
            throw py::value_error("All tensors must be on the same CUDA device");
        }
        if (anchor1_ids.numel() != anchor2_ids.numel() ||
            anchor1_ids.numel() != lookup_alt_deltas.numel() ||
            anchor1_ids.numel() != batch_offset.numel() ||
            anchor1_ids.numel() != grad_alt.numel()) {
            throw py::value_error("anchor1_ids, anchor2_ids, lookup_alt_deltas, batch_offset, grad_alt must have equal numel");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }

        int64_t batch_size = x.size(0);
        int64_t input_dim = x.size(1);
        int64_t n_tables = grad_main.size(1);
        if (grad_main.size(0) != batch_size) {
            throw py::value_error("grad_main first dimension must match x batch size");
        }
        if (grad_alt.numel() != batch_size * n_tables) {
            throw py::value_error("grad_alt numel must be batch_size * n_tables for n_alternatives=1");
        }

        auto opts_x = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor x_grad_flat = torch::zeros({batch_size * input_dim}, opts_x);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_tables;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        int64_t grad_main_stride0 = grad_main.stride(0);
        int64_t grad_main_stride1 = grad_main.stride(1);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_backward_na1_kernel", [&] {
            anchor_pairs_lookup_backward_na1_kernel<scalar_t><<<blocks, threads>>>(
                total,
                reinterpret_cast<const int64_t*>(anchor1_ids.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor2_ids.data_ptr()),
                reinterpret_cast<const scalar_t*>(lookup_alt_deltas.data_ptr()),
                reinterpret_cast<const int64_t*>(batch_offset.data_ptr()),
                reinterpret_cast<const scalar_t*>(grad_main.data_ptr()),
                reinterpret_cast<const scalar_t*>(grad_alt.data_ptr()),
                grad_main_stride0,
                grad_main_stride1,
                batch_size,
                n_tables,
                inv_l1,
                reinterpret_cast<scalar_t*>(x_grad_flat.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_BACKWARD_NA1_PROFILER_OP);
        return x_grad_flat;
    }

    py::tuple
    lprojection_backward_na1_nonsmooth(
        const torch::Tensor& grad_output,
        const torch::Tensor& weights,
        const torch::Tensor& lookup_indices,
        const torch::Tensor& lookup_alt_indices,
        const torch::Tensor& table_indices_flat,
        const torch::Tensor& table_indices_alt_flat,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_LPROJECTION_BACKWARD_NA1_PROFILER_OP);

        if (!grad_output.is_cuda() || !weights.is_cuda() || !lookup_indices.is_cuda() ||
            !lookup_alt_indices.is_cuda() || !table_indices_flat.is_cuda() || !table_indices_alt_flat.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (grad_output.dtype() != weights.dtype()) {
            throw py::value_error("grad_output and weights must have same dtype");
        }
        if (!grad_output.is_floating_point()) {
            throw py::value_error("grad_output/weights must be floating point");
        }
        if (lookup_indices.dtype() != torch::kInt64 || lookup_alt_indices.dtype() != torch::kInt64 ||
            table_indices_flat.dtype() != torch::kInt64 || table_indices_alt_flat.dtype() != torch::kInt64) {
            throw py::value_error("indices tensors must be int64");
        }
        if (lookup_indices.dim() != 2 || lookup_alt_indices.dim() != 3 || lookup_alt_indices.size(2) != 1) {
            throw py::value_error("lookup_indices must be [B,T], lookup_alt_indices must be [B,T,1]");
        }
        if (weights.dim() != 3 || grad_output.dim() != 3) {
            throw py::value_error("weights must be [T,E,O], grad_output must be [B,T,O]");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }
        if (grad_output.device() != weights.device()) {
            throw py::value_error("grad_output and weights must be on same device");
        }
        if (lookup_indices.device() != weights.device() || lookup_alt_indices.device() != weights.device() ||
            table_indices_flat.device() != weights.device() || table_indices_alt_flat.device() != weights.device()) {
            throw py::value_error("all tensors must be on the same CUDA device");
        }

        int64_t batch_size = lookup_indices.size(0);
        int64_t n_tables = lookup_indices.size(1);
        int64_t n_entries = weights.size(1);
        int64_t n_outputs = weights.size(2);
        int64_t total_bt = batch_size * n_tables;
        if (table_indices_flat.numel() != total_bt || table_indices_alt_flat.numel() != total_bt) {
            throw py::value_error("table_indices_*_flat must have numel == B*T");
        }
        if (grad_output.size(0) != batch_size || grad_output.size(1) != n_tables || grad_output.size(2) != n_outputs) {
            throw py::value_error("grad_output shape mismatch");
        }
        if (lookup_alt_indices.numel() != total_bt) {
            throw py::value_error("lookup_alt_indices numel mismatch for n_alternatives=1");
        }

        auto opts = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());
        torch::Tensor weights_grad = torch::zeros_like(weights);
        torch::Tensor lookup_indices_grad_c_grad = torch::empty({batch_size, n_tables}, opts);
        torch::Tensor lookup_alt_indices_grad_c_grad = torch::empty({batch_size, n_tables, 1}, opts);

        int device = weights.device().index();
        c10::cuda::CUDAGuard guard(device);
        int threads = static_cast<int>(threads_per_block);
        int blocks_w = static_cast<int>(((total_bt * n_outputs) + threads - 1) / threads);
        int blocks_c = static_cast<int>((total_bt + threads - 1) / threads);
        int64_t go_s0 = grad_output.stride(0);
        int64_t go_s1 = grad_output.stride(1);
        int64_t go_s2 = grad_output.stride(2);

        AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "lprojection_backward_na1_nonsmooth", [&] {
            lprojection_backward_na1_nonsmooth_weights_kernel<scalar_t><<<blocks_w, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_indices.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(weights_grad.data_ptr())
            );
            lprojection_backward_na1_carriers_kernel<scalar_t><<<blocks_c, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_indices.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_alt_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_alt_indices.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_indices_grad_c_grad.data_ptr()),
                reinterpret_cast<scalar_t*>(lookup_alt_indices_grad_c_grad.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        PROF_END(LUTORCH_MANAGER_LPROJECTION_BACKWARD_NA1_PROFILER_OP);
        return py::make_tuple(weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad);
    }

    py::tuple
    lprojection_backward_na1_smooth(
        const torch::Tensor& grad_output,
        const torch::Tensor& weights,
        const torch::Tensor& lookup_indices,
        const torch::Tensor& lookup_alt_indices,
        const torch::Tensor& table_indices_flat,
        const torch::Tensor& table_indices_alt_flat,
        const torch::Tensor& main_weight,
        const torch::Tensor& alt_weight,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_LPROJECTION_BACKWARD_NA1_PROFILER_OP);

        if (!main_weight.is_cuda() || !alt_weight.is_cuda()) {
            throw py::value_error("main_weight and alt_weight must be CUDA");
        }
        if (main_weight.dtype() != weights.dtype() || alt_weight.dtype() != weights.dtype()) {
            throw py::value_error("main_weight/alt_weight must have same dtype as weights");
        }
        if (main_weight.device() != weights.device() || alt_weight.device() != weights.device()) {
            throw py::value_error("main_weight and alt_weight must be on same device as weights");
        }
        if (main_weight.dim() != 2 || alt_weight.dim() != 3 || alt_weight.size(2) != 1) {
            throw py::value_error("main_weight must be [B,T], alt_weight must be [B,T,1]");
        }

        if (!grad_output.is_cuda() || !weights.is_cuda() || !lookup_indices.is_cuda() ||
            !lookup_alt_indices.is_cuda() || !table_indices_flat.is_cuda() || !table_indices_alt_flat.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (grad_output.dtype() != weights.dtype()) {
            throw py::value_error("grad_output and weights must have same dtype");
        }
        if (!grad_output.is_floating_point()) {
            throw py::value_error("grad_output/weights must be floating point");
        }
        if (lookup_indices.dtype() != torch::kInt64 || lookup_alt_indices.dtype() != torch::kInt64 ||
            table_indices_flat.dtype() != torch::kInt64 || table_indices_alt_flat.dtype() != torch::kInt64) {
            throw py::value_error("indices tensors must be int64");
        }
        if (lookup_indices.dim() != 2 || lookup_alt_indices.dim() != 3 || lookup_alt_indices.size(2) != 1) {
            throw py::value_error("lookup_indices must be [B,T], lookup_alt_indices must be [B,T,1]");
        }
        if (weights.dim() != 3 || grad_output.dim() != 3) {
            throw py::value_error("weights must be [T,E,O], grad_output must be [B,T,O]");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }
        if (grad_output.device() != weights.device()) {
            throw py::value_error("grad_output and weights must be on same device");
        }

        int64_t batch_size = lookup_indices.size(0);
        int64_t n_tables = lookup_indices.size(1);
        int64_t n_entries = weights.size(1);
        int64_t n_outputs = weights.size(2);
        int64_t total_bt = batch_size * n_tables;
        if (table_indices_flat.numel() != total_bt || table_indices_alt_flat.numel() != total_bt) {
            throw py::value_error("table_indices_*_flat must have numel == B*T");
        }
        if (grad_output.size(0) != batch_size || grad_output.size(1) != n_tables || grad_output.size(2) != n_outputs) {
            throw py::value_error("grad_output shape mismatch");
        }
        if (lookup_alt_indices.numel() != total_bt) {
            throw py::value_error("lookup_alt_indices numel mismatch for n_alternatives=1");
        }

        auto opts = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());
        torch::Tensor weights_grad = torch::zeros_like(weights);
        torch::Tensor lookup_indices_grad_c_grad = torch::empty({batch_size, n_tables}, opts);
        torch::Tensor lookup_alt_indices_grad_c_grad = torch::empty({batch_size, n_tables, 1}, opts);

        int threads = static_cast<int>(threads_per_block);
        int blocks_w = static_cast<int>(((total_bt * n_outputs) + threads - 1) / threads);
        int blocks_c = static_cast<int>((total_bt + threads - 1) / threads);
        int64_t go_s0 = grad_output.stride(0);
        int64_t go_s1 = grad_output.stride(1);
        int64_t go_s2 = grad_output.stride(2);

        int device = weights.device().index();
        c10::cuda::CUDAGuard guard(device);
        AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "lprojection_backward_na1_smooth_weights", [&] {
            lprojection_backward_na1_smooth_weights_kernel<scalar_t><<<blocks_w, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_indices.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_alt_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_alt_indices.data_ptr()),
                reinterpret_cast<const scalar_t*>(main_weight.data_ptr()),
                reinterpret_cast<const scalar_t*>(alt_weight.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(weights_grad.data_ptr())
            );
            lprojection_backward_na1_carriers_kernel<scalar_t><<<blocks_c, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_indices.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_alt_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_alt_indices.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_indices_grad_c_grad.data_ptr()),
                reinterpret_cast<scalar_t*>(lookup_alt_indices_grad_c_grad.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        PROF_END(LUTORCH_MANAGER_LPROJECTION_BACKWARD_NA1_PROFILER_OP);
        return py::make_tuple(weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad);
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
        .def(
            "anchor_pairs_lookup_backward_na1",
            &LUTorchManager::anchor_pairs_lookup_backward_na1,
            py::arg("x"),
            py::arg("anchor1_ids"),
            py::arg("anchor2_ids"),
            py::arg("lookup_alt_deltas"),
            py::arg("batch_offset"),
            py::arg("grad_main"),
            py::arg("grad_alt"),
            py::arg("inv_l1"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "lprojection_backward_na1_nonsmooth",
            &LUTorchManager::lprojection_backward_na1_nonsmooth,
            py::arg("grad_output"),
            py::arg("weights"),
            py::arg("lookup_indices"),
            py::arg("lookup_alt_indices"),
            py::arg("table_indices_flat"),
            py::arg("table_indices_alt_flat"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "lprojection_backward_na1_smooth",
            &LUTorchManager::lprojection_backward_na1_smooth,
            py::arg("grad_output"),
            py::arg("weights"),
            py::arg("lookup_indices"),
            py::arg("lookup_alt_indices"),
            py::arg("table_indices_flat"),
            py::arg("table_indices_alt_flat"),
            py::arg("main_weight"),
            py::arg("alt_weight"),
            py::arg("threads_per_block") = 256
        )
        #endif
        .def("get_profiling_stats", &LUTorchManager::get_profiling_stats);
}
