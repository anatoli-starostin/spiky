#include <tuple>
#include <ATen/AccumulateType.h>
#include "common_misc.h"
#include "lprojection.h"

// bf16/fp16 reductions accumulate in fp32 (at::acc_type<scalar_t,true> is float for Half/
// BFloat16 and the type itself for float/double, so fp32/fp64 stay byte-identical). The three
// kernels that accumulate a grad via atomicAdd write into an accumulator buffer of this scalar
// type; the wrapper allocates it and casts back to the input dtype.
static inline at::ScalarType lutorch_acc_scalar_type(at::ScalarType t) {
    return (t == at::kHalf || t == at::kBFloat16) ? at::kFloat : t;
}

namespace py = pybind11;

#ifndef NO_CUDA
// Generic backward kernel matching Python fallback semantics for any n_alternatives.
template <typename scalar_t, typename acc_t = at::acc_type<scalar_t, true>>
__global__ void anchor_pairs_lookup_backward_all_kernel(
    int64_t total,
    const int64_t* anchor1_ids_ptr,
    const int64_t* anchor2_ids_ptr,
    const scalar_t* lookup_alt_deltas_ptr,
    const int64_t* batch_offset_ptr,
    const scalar_t* grad_main_ptr,
    const scalar_t* grad_alt_ptr,
    int64_t grad_main_stride0,
    int64_t grad_main_stride1,
    int64_t n_tables,
    int64_t n_alternatives,
    bool inv_l1,
    acc_t* x_grad_flat_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= total) {
        return;
    }

    int64_t bt = linear_tid / n_alternatives;
    int64_t b = bt / n_tables;
    int64_t t = bt - b * n_tables;

    acc_t delta = static_cast<acc_t>(lookup_alt_deltas_ptr[linear_tid]);
    acc_t minus_uncertainty_derivative = static_cast<acc_t>(0);
    if (inv_l1) {
        acc_t one_plus_abs = static_cast<acc_t>(1) + (delta < acc_t(0) ? -delta : delta);
        minus_uncertainty_derivative =
            static_cast<acc_t>(0.5) * static_cast<acc_t>((delta > acc_t(0)) - (delta < acc_t(0)))
            / (one_plus_abs * one_plus_abs);
    } else {
        acc_t one_plus_sq = static_cast<acc_t>(1) + delta * delta;
        minus_uncertainty_derivative = delta / (one_plus_sq * one_plus_sq);
    }

    acc_t grad_main = static_cast<acc_t>(grad_main_ptr[b * grad_main_stride0 + t * grad_main_stride1]);
    acc_t grad_alt = static_cast<acc_t>(grad_alt_ptr[linear_tid]);
    acc_t du = (grad_main - grad_alt) * minus_uncertainty_derivative / static_cast<acc_t>(n_alternatives);

    int64_t idx1 = batch_offset_ptr[linear_tid] + anchor1_ids_ptr[linear_tid];
    int64_t idx2 = batch_offset_ptr[linear_tid] + anchor2_ids_ptr[linear_tid];
    atomicAdd(x_grad_flat_ptr + idx1, du);
    atomicAdd(x_grad_flat_ptr + idx2, -du);
}

// Cell-index kernels (index_t = int32_t or int64_t): only the STORED index tensors are index_t. Every index is
// loaded into an int64_t local and all offset arithmetic -- (table * n_entries + entry) * n_outputs + o -- stays
// int64, because rows * d_out can exceed 2^31 even when every stored index fits int32.
template <typename scalar_t, typename index_t = int64_t, typename acc_t = at::acc_type<scalar_t, true>>
__global__ void lprojection_backward_na1_nonsmooth_weights_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const index_t* table_indices_flat_ptr,
    const index_t* lookup_indices_flat_ptr,
    int64_t grad_output_stride0,
    int64_t grad_output_stride1,
    int64_t grad_output_stride2,
    acc_t* weights_grad_ptr
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
    int64_t table = static_cast<int64_t>(table_indices_flat_ptr[bt]);
    int64_t entry = static_cast<int64_t>(lookup_indices_flat_ptr[bt]);
    int64_t widx = (table * n_entries + entry) * n_outputs + o;
    atomicAdd(weights_grad_ptr + widx, static_cast<acc_t>(g));  // fp32 accumulation for bf16/fp16
}

template <typename scalar_t>
__global__ void lprojection_forward_smooth_weights_kernel(
    int64_t total_bt,
    int64_t n_alternatives,
    bool l1_uncertainty,
    const scalar_t* lookup_alt_deltas_ptr, // [B*T*A] contiguous flattened
    scalar_t* main_weight_ptr,             // [B*T]
    scalar_t* alt_weight_ptr               // [B*T*A]
) {
    int64_t bt = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bt >= total_bt) {
        return;
    }
    using acc_t = at::acc_type<scalar_t, true>;
    acc_t inv_n_alt = static_cast<acc_t>(1.0) / static_cast<acc_t>(n_alternatives);
    acc_t uncertainty_sum = static_cast<acc_t>(0);
    int64_t base = bt * n_alternatives;
    for (int64_t a = 0; a < n_alternatives; ++a) {
        acc_t d = static_cast<acc_t>(lookup_alt_deltas_ptr[base + a]);
        acc_t u;
        if (l1_uncertainty) {
            u = static_cast<acc_t>(0.5) / (static_cast<acc_t>(1.0) + (d < acc_t(0) ? -d : d));
        } else {
            u = static_cast<acc_t>(0.5) / (static_cast<acc_t>(1.0) + d * d);
        }
        alt_weight_ptr[base + a] = static_cast<scalar_t>(u * inv_n_alt);
        uncertainty_sum += u;
    }
    main_weight_ptr[bt] = static_cast<scalar_t>(static_cast<acc_t>(1.0) - uncertainty_sum * inv_n_alt);
}

template <typename scalar_t, typename index_t = int64_t>
__global__ void lprojection_forward_smooth_output_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    int64_t n_alternatives,
    const scalar_t* weights_ptr,                 // [T,E,O] contiguous
    const index_t* table_indices_flat_ptr,       // [B*T]
    const index_t* lookup_indices_flat_ptr,      // [B*T]
    const index_t* table_indices_alt_flat_ptr,   // [B*T*A]
    const index_t* lookup_alt_indices_flat_ptr,  // [B*T*A]
    const scalar_t* main_weight_ptr,             // [B*T]
    const scalar_t* alt_weight_ptr,              // [B*T*A]
    scalar_t* output_ptr                         // [B,T,O] contiguous
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

    using acc_t = at::acc_type<scalar_t, true>;
    int64_t table_main = table_indices_flat_ptr[bt];
    int64_t entry_main = lookup_indices_flat_ptr[bt];
    acc_t acc = static_cast<acc_t>(weights_ptr[(table_main * n_entries + entry_main) * n_outputs + o])
                * static_cast<acc_t>(main_weight_ptr[bt]);

    int64_t base = bt * n_alternatives;
    for (int64_t a = 0; a < n_alternatives; ++a) {
        int64_t bta = base + a;
        int64_t table_alt = table_indices_alt_flat_ptr[bta];
        int64_t entry_alt = lookup_alt_indices_flat_ptr[bta];
        acc_t w = static_cast<acc_t>(weights_ptr[(table_alt * n_entries + entry_alt) * n_outputs + o]);
        acc += w * static_cast<acc_t>(alt_weight_ptr[bta]);
    }

    output_ptr[(b * n_tables + t) * n_outputs + o] = static_cast<scalar_t>(acc);
}

template <typename scalar_t, typename index_t = int64_t, typename acc_t = at::acc_type<scalar_t, true>>
__global__ void lprojection_backward_na1_smooth_weights_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const index_t* table_indices_flat_ptr,
    const index_t* lookup_indices_flat_ptr,
    const index_t* table_indices_alt_flat_ptr,
    const index_t* lookup_alt_indices_flat_ptr,
    const scalar_t* main_weight_flat_ptr,
    const scalar_t* alt_weight_flat_ptr,
    int64_t grad_output_stride0,
    int64_t grad_output_stride1,
    int64_t grad_output_stride2,
    acc_t* weights_grad_ptr
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
    acc_t g = static_cast<acc_t>(grad_output_ptr[b * grad_output_stride0 + t * grad_output_stride1 + o * grad_output_stride2]);
    acc_t g_main = g * static_cast<acc_t>(main_weight_flat_ptr[bt]);
    acc_t g_alt = g * static_cast<acc_t>(alt_weight_flat_ptr[bt]);

    int64_t table_main = table_indices_flat_ptr[bt];
    int64_t entry_main = lookup_indices_flat_ptr[bt];
    int64_t widx_main = (table_main * n_entries + entry_main) * n_outputs + o;
    atomicAdd(weights_grad_ptr + widx_main, g_main);

    int64_t table_alt = table_indices_alt_flat_ptr[bt];
    int64_t entry_alt = lookup_alt_indices_flat_ptr[bt];
    int64_t widx_alt = (table_alt * n_entries + entry_alt) * n_outputs + o;
    atomicAdd(weights_grad_ptr + widx_alt, g_alt);
}

template <typename scalar_t, typename index_t = int64_t>
__global__ void lprojection_backward_na1_carriers_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const scalar_t* weights_ptr,
    const index_t* table_indices_flat_ptr,
    const index_t* lookup_indices_flat_ptr,
    const index_t* table_indices_alt_flat_ptr,
    const index_t* lookup_alt_indices_flat_ptr,
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
    using acc_t = at::acc_type<scalar_t, true>;
    acc_t acc_main = static_cast<acc_t>(0);
    acc_t acc_alt = static_cast<acc_t>(0);
    for (int64_t o = 0; o < n_outputs; ++o) {
        acc_t g = static_cast<acc_t>(grad_output_ptr[b * grad_output_stride0 + t * grad_output_stride1 + o * grad_output_stride2]);
        int64_t widx_main = (table_main * n_entries + entry_main) * n_outputs + o;
        int64_t widx_alt = (table_alt * n_entries + entry_alt) * n_outputs + o;
        acc_main += g * static_cast<acc_t>(weights_ptr[widx_main]);
        acc_alt += g * static_cast<acc_t>(weights_ptr[widx_alt]);
    }
    lookup_indices_grad_c_grad_ptr[bt] = static_cast<scalar_t>(acc_main);
    lookup_alt_indices_grad_c_grad_ptr[bt] = static_cast<scalar_t>(acc_alt);
}
#endif

// The stored cell-index tensors of the entry points lutorch_ex calls (lprojection_forward_smooth,
// lprojection_backward_na1_nonsmooth, lprojection_backward_na1_smooth): int32 or int64, all four the same dtype (one
// index_t per launch). The kernels read them through data_ptr<index_t>(), which re-checks the dtype.
static void check_cell_index_dtypes(const torch::Tensor& lookup_indices, const torch::Tensor& lookup_alt_indices,
                                    const torch::Tensor& table_indices_flat, const torch::Tensor& table_indices_alt_flat) {
    auto dt = lookup_indices.scalar_type();
    if ((dt != torch::kInt32 && dt != torch::kInt64) || lookup_alt_indices.scalar_type() != dt ||
        table_indices_flat.scalar_type() != dt || table_indices_alt_flat.scalar_type() != dt) {
        throw py::value_error("indices tensors must all be int32 or all be int64");
    }
}

class SPIKY_HIDDEN LUTorchManager {
public:
    LUTorchManager() {}

    ~LUTorchManager() {
    }

#ifndef NO_CUDA
    py::tuple
    lprojection_forward_smooth(
        const torch::Tensor& weights,
        const torch::Tensor& lookup_indices,
        const torch::Tensor& lookup_alt_indices,
        const torch::Tensor& lookup_alt_deltas,
        const torch::Tensor& table_indices_flat,
        const torch::Tensor& table_indices_alt_flat,
        bool l1_uncertainty,
        int64_t threads_per_block = 256
    ) {

        if (!weights.is_cuda() || !lookup_indices.is_cuda() || !lookup_alt_indices.is_cuda() ||
            !lookup_alt_deltas.is_cuda() || !table_indices_flat.is_cuda() || !table_indices_alt_flat.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!weights.is_floating_point() || lookup_alt_deltas.dtype() != weights.dtype()) {
            throw py::value_error("weights/lookup_alt_deltas must be floating with same dtype");
        }
        check_cell_index_dtypes(lookup_indices, lookup_alt_indices, table_indices_flat, table_indices_alt_flat);
        if (weights.dim() != 3 || lookup_indices.dim() != 2 || lookup_alt_indices.dim() != 3 || lookup_alt_deltas.dim() != 3) {
            throw py::value_error("weights [T,E,O], lookup_indices [B,T], lookup_alt_indices/deltas [B,T,A] required");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }
        if (lookup_indices.device() != weights.device() || lookup_alt_indices.device() != weights.device() ||
            lookup_alt_deltas.device() != weights.device() || table_indices_flat.device() != weights.device() ||
            table_indices_alt_flat.device() != weights.device()) {
            throw py::value_error("all tensors must be on the same CUDA device");
        }

        int64_t batch_size = lookup_indices.size(0);
        int64_t n_tables = lookup_indices.size(1);
        int64_t n_entries = weights.size(1);
        int64_t n_outputs = weights.size(2);
        int64_t n_alternatives = lookup_alt_indices.size(2);
        int64_t total_bt = batch_size * n_tables;
        int64_t total_bta = total_bt * n_alternatives;

        if (lookup_alt_deltas.numel() != total_bta || lookup_alt_indices.numel() != total_bta) {
            throw py::value_error("lookup_alt_indices and lookup_alt_deltas numel must be B*T*A");
        }
        if (table_indices_flat.numel() != total_bt || table_indices_alt_flat.numel() != total_bta) {
            throw py::value_error("table_indices_flat must be B*T and table_indices_alt_flat must be B*T*A");
        }

        auto opts = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());
        torch::Tensor main_weight = torch::empty({batch_size, n_tables}, opts);
        torch::Tensor alt_weight = torch::empty({batch_size, n_tables, n_alternatives}, opts);
        torch::Tensor output = torch::empty({batch_size, n_tables, n_outputs}, opts);

        int device = weights.device().index();
        c10::cuda::CUDAGuard guard(device);
        int threads = static_cast<int>(threads_per_block);
        int blocks_bt = static_cast<int>((total_bt + threads - 1) / threads);
        int blocks_out = static_cast<int>(((total_bt * n_outputs) + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                        weights.scalar_type(), "lprojection_forward_smooth", [&] {
            lprojection_forward_smooth_weights_kernel<scalar_t><<<blocks_bt, threads>>>(
                total_bt,
                n_alternatives,
                l1_uncertainty,
                reinterpret_cast<const scalar_t*>(lookup_alt_deltas.data_ptr()),
                reinterpret_cast<scalar_t*>(main_weight.data_ptr()),
                reinterpret_cast<scalar_t*>(alt_weight.data_ptr())
            );
            AT_DISPATCH_INDEX_TYPES(lookup_indices.scalar_type(), "lprojection_forward_smooth_index", [&] {
                lprojection_forward_smooth_output_kernel<scalar_t, index_t><<<blocks_out, threads>>>(
                    total_bt,
                    n_tables,
                    n_outputs,
                    n_entries,
                    n_alternatives,
                    reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                    table_indices_flat.data_ptr<index_t>(),
                    lookup_indices.data_ptr<index_t>(),
                    table_indices_alt_flat.data_ptr<index_t>(),
                    lookup_alt_indices.data_ptr<index_t>(),
                    reinterpret_cast<const scalar_t*>(main_weight.data_ptr()),
                    reinterpret_cast<const scalar_t*>(alt_weight.data_ptr()),
                    reinterpret_cast<scalar_t*>(output.data_ptr())
                );
            });
        });
        CU_CHECK(cudaGetLastError());

        return py::make_tuple(output, main_weight, alt_weight);
    }

    // Backward for generic n_alternatives (matching Python fallback semantics).
    torch::Tensor
    anchor_pairs_lookup_backward_all(
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
        int64_t n_alternatives = lookup_alt_deltas.size(2);
        if (grad_alt.numel() != batch_size * n_tables * n_alternatives) {
            throw py::value_error("grad_alt numel must be batch_size * n_tables * n_alternatives");
        }

        // Accumulate the scatter in fp32 for bf16/fp16 (and so atomicAdd runs on float, which
        // is well-supported); fp32/fp64 keep their own dtype so this is byte-identical there.
        auto acc_scalar = lutorch_acc_scalar_type(x.scalar_type());
        auto opts_acc = torch::TensorOptions().dtype(acc_scalar).device(x.device());
        torch::Tensor x_grad_flat = torch::zeros({batch_size * input_dim}, opts_acc);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_tables * n_alternatives;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        int64_t grad_main_stride0 = grad_main.stride(0);
        int64_t grad_main_stride1 = grad_main.stride(1);

        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                        x.scalar_type(), "anchor_pairs_lookup_backward_all_kernel", [&] {
            anchor_pairs_lookup_backward_all_kernel<scalar_t><<<blocks, threads>>>(
                total,
                reinterpret_cast<const int64_t*>(anchor1_ids.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor2_ids.data_ptr()),
                reinterpret_cast<const scalar_t*>(lookup_alt_deltas.data_ptr()),
                reinterpret_cast<const int64_t*>(batch_offset.data_ptr()),
                reinterpret_cast<const scalar_t*>(grad_main.data_ptr()),
                reinterpret_cast<const scalar_t*>(grad_alt.data_ptr()),
                grad_main_stride0,
                grad_main_stride1,
                n_tables,
                n_alternatives,
                inv_l1,
                reinterpret_cast<at::acc_type<scalar_t, true>*>(x_grad_flat.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        return x_grad_flat.to(x.dtype());
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
        check_cell_index_dtypes(lookup_indices, lookup_alt_indices, table_indices_flat, table_indices_alt_flat);
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
        // Weight grad accumulates via atomicAdd across the batch: in fp32 for bf16/fp16 (and so
        // atomicAdd runs on float), cast back to the weight dtype. fp32/fp64 are byte-identical.
        auto acc_scalar = lutorch_acc_scalar_type(weights.scalar_type());
        torch::Tensor weights_grad = torch::zeros(
            weights.sizes(), torch::TensorOptions().dtype(acc_scalar).device(weights.device()));
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

        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                        weights.scalar_type(), "lprojection_backward_na1_nonsmooth", [&] {
          AT_DISPATCH_INDEX_TYPES(lookup_indices.scalar_type(), "lprojection_backward_na1_nonsmooth_index", [&] {
            lprojection_backward_na1_nonsmooth_weights_kernel<scalar_t, index_t><<<blocks_w, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                table_indices_flat.data_ptr<index_t>(),
                lookup_indices.data_ptr<index_t>(),
                go_s0, go_s1, go_s2,
                reinterpret_cast<at::acc_type<scalar_t, true>*>(weights_grad.data_ptr())
            );
            lprojection_backward_na1_carriers_kernel<scalar_t, index_t><<<blocks_c, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                table_indices_flat.data_ptr<index_t>(),
                lookup_indices.data_ptr<index_t>(),
                table_indices_alt_flat.data_ptr<index_t>(),
                lookup_alt_indices.data_ptr<index_t>(),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_indices_grad_c_grad.data_ptr()),
                reinterpret_cast<scalar_t*>(lookup_alt_indices_grad_c_grad.data_ptr())
            );
          });
        });
        CU_CHECK(cudaGetLastError());
        return py::make_tuple(weights_grad.to(weights.dtype()), lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad);
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
        check_cell_index_dtypes(lookup_indices, lookup_alt_indices, table_indices_flat, table_indices_alt_flat);
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
        // Weight grad accumulates via atomicAdd across the batch: fp32 buffer for bf16/fp16.
        auto acc_scalar = lutorch_acc_scalar_type(weights.scalar_type());
        torch::Tensor weights_grad = torch::zeros(
            weights.sizes(), torch::TensorOptions().dtype(acc_scalar).device(weights.device()));
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
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                        weights.scalar_type(), "lprojection_backward_na1_smooth_weights", [&] {
          AT_DISPATCH_INDEX_TYPES(lookup_indices.scalar_type(), "lprojection_backward_na1_smooth_index", [&] {
            lprojection_backward_na1_smooth_weights_kernel<scalar_t, index_t><<<blocks_w, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                table_indices_flat.data_ptr<index_t>(),
                lookup_indices.data_ptr<index_t>(),
                table_indices_alt_flat.data_ptr<index_t>(),
                lookup_alt_indices.data_ptr<index_t>(),
                reinterpret_cast<const scalar_t*>(main_weight.data_ptr()),
                reinterpret_cast<const scalar_t*>(alt_weight.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<at::acc_type<scalar_t, true>*>(weights_grad.data_ptr())
            );
            lprojection_backward_na1_carriers_kernel<scalar_t, index_t><<<blocks_c, threads>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                table_indices_flat.data_ptr<index_t>(),
                lookup_indices.data_ptr<index_t>(),
                table_indices_alt_flat.data_ptr<index_t>(),
                lookup_alt_indices.data_ptr<index_t>(),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_indices_grad_c_grad.data_ptr()),
                reinterpret_cast<scalar_t*>(lookup_alt_indices_grad_c_grad.data_ptr())
            );
          });
        });
        CU_CHECK(cudaGetLastError());
        return py::make_tuple(weights_grad.to(weights.dtype()), lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad);
    }
#endif
};

void PB_LUTorchManager(py::module& m) {
    py::class_<LUTorchManager>(m, "LUTorchManager")
        .def(py::init<>())
        #ifndef NO_CUDA
        .def(
            "anchor_pairs_lookup_backward_all",
            &LUTorchManager::anchor_pairs_lookup_backward_all,
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
        .def(
            "lprojection_forward_smooth",
            &LUTorchManager::lprojection_forward_smooth,
            py::arg("weights"),
            py::arg("lookup_indices"),
            py::arg("lookup_alt_indices"),
            py::arg("lookup_alt_deltas"),
            py::arg("table_indices_flat"),
            py::arg("table_indices_alt_flat"),
            py::arg("l1_uncertainty"),
            py::arg("threads_per_block") = 256
        )
        #endif
        ;

    // Singleton: one manager for all lprojection ops.
    // Never destroyed (intentional leak) to avoid segfault on exit when CUDA/PyTorch tear down first.
    m.def(
        "get_lutorch_manager",
        []() -> LUTorchManager* {
            static LUTorchManager* instance = new LUTorchManager();
            return instance;
        },
        py::return_value_policy::reference
    );
}
