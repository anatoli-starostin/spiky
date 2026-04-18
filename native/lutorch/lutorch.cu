#include <tuple>
#include "../common/misc.h"
#include "lutorch.h"
#include <ATen/cuda/CUDAContext.h>

#ifndef NO_CUDA
#include <cuda_fp8.h>
#endif

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
__global__ void anchor_pairs_lookup_eval_forward_kernel(
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
__global__ void anchor_pairs_lookup_forward_na2_kernel(
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
    scalar_t big = static_cast<scalar_t>(1e30);
    scalar_t min1_abs_delta = big;
    scalar_t min2_abs_delta = big;
    scalar_t min1_delta = static_cast<scalar_t>(0);
    scalar_t min2_delta = static_cast<scalar_t>(0);
    int64_t min1_anchor_a = 0;
    int64_t min1_anchor_b = 0;
    int64_t min2_anchor_a = 0;
    int64_t min2_anchor_b = 0;
    int64_t min1_bit_pos = 0;
    int64_t min2_bit_pos = 0;

    for (int64_t p = 0; p < n_anchor_pairs; ++p) {
        int64_t anchor_a = anchor_pairs_a_ptr[table_offset + p];
        int64_t anchor_b = anchor_pairs_b_ptr[table_offset + p];
        scalar_t delta = lutorch_delta(x_ptr, b, x_stride0, x_stride1, anchor_a, anchor_b);

        if (delta > cmp_eps) {
            lookup_idx |= (static_cast<int64_t>(1) << p);
        }

        scalar_t abs_delta = lutorch_abs(delta);
        if (abs_delta < min1_abs_delta) {
            min2_abs_delta = min1_abs_delta;
            min2_delta = min1_delta;
            min2_anchor_a = min1_anchor_a;
            min2_anchor_b = min1_anchor_b;
            min2_bit_pos = min1_bit_pos;

            min1_abs_delta = abs_delta;
            min1_delta = delta;
            min1_anchor_a = anchor_a;
            min1_anchor_b = anchor_b;
            min1_bit_pos = p;
        } else if (abs_delta < min2_abs_delta) {
            min2_abs_delta = abs_delta;
            min2_delta = delta;
            min2_anchor_a = anchor_a;
            min2_anchor_b = anchor_b;
            min2_bit_pos = p;
        }
    }

    int64_t base = linear_tid * 2;
    lookup_indices_ptr[linear_tid] = lookup_idx;
    lookup_alt_indices_ptr[base + 0] = lookup_idx ^ (static_cast<int64_t>(1) << min1_bit_pos);
    lookup_alt_indices_ptr[base + 1] = lookup_idx ^ (static_cast<int64_t>(1) << min2_bit_pos);
    lookup_alt_deltas_ptr[base + 0] = min1_delta;
    lookup_alt_deltas_ptr[base + 1] = min2_delta;
    if (anchor1_ids_ptr != nullptr) {
        anchor1_ids_ptr[base + 0] = min1_anchor_a;
        anchor2_ids_ptr[base + 0] = min1_anchor_b;
        anchor1_ids_ptr[base + 1] = min2_anchor_a;
        anchor2_ids_ptr[base + 1] = min2_anchor_b;
    }
}


template <typename scalar_t>
__global__ void anchor_pairs_lookup_forward_na3_kernel(
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
    scalar_t big = static_cast<scalar_t>(1e30);
    scalar_t min1_abs_delta = big;
    scalar_t min2_abs_delta = big;
    scalar_t min3_abs_delta = big;
    scalar_t min1_delta = static_cast<scalar_t>(0);
    scalar_t min2_delta = static_cast<scalar_t>(0);
    scalar_t min3_delta = static_cast<scalar_t>(0);
    int64_t min1_anchor_a = 0;
    int64_t min1_anchor_b = 0;
    int64_t min2_anchor_a = 0;
    int64_t min2_anchor_b = 0;
    int64_t min3_anchor_a = 0;
    int64_t min3_anchor_b = 0;
    int64_t min1_bit_pos = 0;
    int64_t min2_bit_pos = 0;
    int64_t min3_bit_pos = 0;

    for (int64_t p = 0; p < n_anchor_pairs; ++p) {
        int64_t anchor_a = anchor_pairs_a_ptr[table_offset + p];
        int64_t anchor_b = anchor_pairs_b_ptr[table_offset + p];
        scalar_t delta = lutorch_delta(x_ptr, b, x_stride0, x_stride1, anchor_a, anchor_b);

        if (delta > cmp_eps) {
            lookup_idx |= (static_cast<int64_t>(1) << p);
        }

        scalar_t abs_delta = lutorch_abs(delta);
        if (abs_delta < min1_abs_delta) {
            min3_abs_delta = min2_abs_delta;
            min3_delta = min2_delta;
            min3_anchor_a = min2_anchor_a;
            min3_anchor_b = min2_anchor_b;
            min3_bit_pos = min2_bit_pos;

            min2_abs_delta = min1_abs_delta;
            min2_delta = min1_delta;
            min2_anchor_a = min1_anchor_a;
            min2_anchor_b = min1_anchor_b;
            min2_bit_pos = min1_bit_pos;

            min1_abs_delta = abs_delta;
            min1_delta = delta;
            min1_anchor_a = anchor_a;
            min1_anchor_b = anchor_b;
            min1_bit_pos = p;
        } else if (abs_delta < min2_abs_delta) {
            min3_abs_delta = min2_abs_delta;
            min3_delta = min2_delta;
            min3_anchor_a = min2_anchor_a;
            min3_anchor_b = min2_anchor_b;
            min3_bit_pos = min2_bit_pos;

            min2_abs_delta = abs_delta;
            min2_delta = delta;
            min2_anchor_a = anchor_a;
            min2_anchor_b = anchor_b;
            min2_bit_pos = p;
        } else if (abs_delta < min3_abs_delta) {
            min3_abs_delta = abs_delta;
            min3_delta = delta;
            min3_anchor_a = anchor_a;
            min3_anchor_b = anchor_b;
            min3_bit_pos = p;
        }
    }

    int64_t base = linear_tid * 3;
    lookup_indices_ptr[linear_tid] = lookup_idx;
    lookup_alt_indices_ptr[base + 0] = lookup_idx ^ (static_cast<int64_t>(1) << min1_bit_pos);
    lookup_alt_indices_ptr[base + 1] = lookup_idx ^ (static_cast<int64_t>(1) << min2_bit_pos);
    lookup_alt_indices_ptr[base + 2] = lookup_idx ^ (static_cast<int64_t>(1) << min3_bit_pos);
    lookup_alt_deltas_ptr[base + 0] = min1_delta;
    lookup_alt_deltas_ptr[base + 1] = min2_delta;
    lookup_alt_deltas_ptr[base + 2] = min3_delta;
    if (anchor1_ids_ptr != nullptr) {
        anchor1_ids_ptr[base + 0] = min1_anchor_a;
        anchor2_ids_ptr[base + 0] = min1_anchor_b;
        anchor1_ids_ptr[base + 1] = min2_anchor_a;
        anchor2_ids_ptr[base + 1] = min2_anchor_b;
        anchor1_ids_ptr[base + 2] = min3_anchor_a;
        anchor2_ids_ptr[base + 2] = min3_anchor_b;
    }
}


// Generic forward kernel for n_alternatives == n_anchor_pairs.
// Produces all alternatives by flipping each bit position once, without sorting.
template <typename scalar_t>
__global__ void anchor_pairs_lookup_forward_all_kernel(
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
    int64_t base = linear_tid * n_anchor_pairs;

    int64_t lookup_idx = 0;

    // First pass: compute lookup index bits and store all deltas / anchor ids.
    for (int64_t p = 0; p < n_anchor_pairs; ++p) {
        int64_t anchor_a = anchor_pairs_a_ptr[table_offset + p];
        int64_t anchor_b = anchor_pairs_b_ptr[table_offset + p];
        scalar_t delta = lutorch_delta(x_ptr, b, x_stride0, x_stride1, anchor_a, anchor_b);

        if (delta > cmp_eps) {
            lookup_idx |= (static_cast<int64_t>(1) << p);
        }

        lookup_alt_deltas_ptr[base + p] = delta;
        if (anchor1_ids_ptr != nullptr) {
            anchor1_ids_ptr[base + p] = anchor_a;
            anchor2_ids_ptr[base + p] = anchor_b;
        }
    }

    // Second pass: fill alternative indices by flipping each bit.
    lookup_indices_ptr[linear_tid] = lookup_idx;
    for (int64_t p = 0; p < n_anchor_pairs; ++p) {
        lookup_alt_indices_ptr[base + p] = lookup_idx ^ (static_cast<int64_t>(1) << p);
    }
}

// Generic backward kernel matching Python fallback semantics for any n_alternatives.
template <typename scalar_t>
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
    scalar_t uncertainty_bias,
    scalar_t* x_grad_flat_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= total) {
        return;
    }

    int64_t bt = linear_tid / n_alternatives;
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
    scalar_t du = (grad_main - grad_alt) * minus_uncertainty_derivative / static_cast<scalar_t>(n_alternatives);

    int64_t idx1 = batch_offset_ptr[linear_tid] + anchor1_ids_ptr[linear_tid];
    int64_t idx2 = batch_offset_ptr[linear_tid] + anchor2_ids_ptr[linear_tid];
    atomicAdd(x_grad_flat_ptr + idx1, du);
    atomicAdd(x_grad_flat_ptr + idx2, -du);
}

// =====================================================================
// Tiny Anchor Pairs Lookup — int16/int32 specialization for the
// BitPermutationLUT path. Fixed: n_alternatives=1, cmp_eps=0,
// uncertainty_mode=INVERSE_L1, uncertainty_bias=0.5, n_anchor_pairs <= 16,
// input_dim <= 32767.
// =====================================================================

// Forward: one thread per (b, t). Writes int16 indices + float delta + int16
// anchor1/2 ids (used for scatter in backward).
template <typename scalar_t>
__global__ void tiny_apl_fwd_kernel(
    const scalar_t* __restrict__ x_ptr,
    int32_t batch_size,
    int32_t input_dim,
    const int16_t* __restrict__ a_ptr,         // [n_tables, n_anchor_pairs]
    const int16_t* __restrict__ b_ptr,
    int32_t n_tables,
    int32_t n_anchor_pairs,
    int16_t*       __restrict__ lookup_idx_ptr,   // [B, n_tables]
    int16_t*       __restrict__ alt_idx_ptr,      // [B, n_tables]
    scalar_t*      __restrict__ alt_delta_ptr,    // [B, n_tables]
    int16_t*       __restrict__ anchor1_ids_ptr,  // [B, n_tables]
    int16_t*       __restrict__ anchor2_ids_ptr
) {
    int32_t tid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t total = batch_size * n_tables;
    if (tid >= total) return;

    int32_t b = tid / n_tables;
    int32_t t = tid - b * n_tables;
    int32_t table_offset = t * n_anchor_pairs;
    const scalar_t* xb = x_ptr + b * input_dim;

    int32_t lookup_idx = 0;
    scalar_t min_abs_delta = static_cast<scalar_t>(0);
    scalar_t min_delta = static_cast<scalar_t>(0);
    int16_t min_a = 0, min_b = 0;
    int32_t min_pos = 0;

    #pragma unroll 1
    for (int32_t p = 0; p < n_anchor_pairs; ++p) {
        int16_t aa = a_ptr[table_offset + p];
        int16_t bb = b_ptr[table_offset + p];
        scalar_t delta = xb[aa] - xb[bb];

        if (delta > static_cast<scalar_t>(0)) {
            lookup_idx |= (1 << p);
        }

        scalar_t abs_d = delta > static_cast<scalar_t>(0) ? delta : -delta;
        if (p == 0 || abs_d < min_abs_delta) {
            min_abs_delta = abs_d;
            min_delta = delta;
            min_a = aa;
            min_b = bb;
            min_pos = p;
        }
    }

    lookup_idx_ptr[tid]  = static_cast<int16_t>(lookup_idx);
    alt_idx_ptr[tid]     = static_cast<int16_t>(lookup_idx ^ (1 << min_pos));
    alt_delta_ptr[tid]   = min_delta;
    anchor1_ids_ptr[tid] = min_a;
    anchor2_ids_ptr[tid] = min_b;
}

// Backward: one thread per (b, t). INVERSE_L1 (bias=0.5 fixed), n_alt=1.
template <typename scalar_t>
__global__ void tiny_apl_bwd_kernel(
    int32_t total,   // B * n_tables
    int32_t input_dim,
    int32_t n_tables,
    const int16_t* __restrict__ anchor1_ids_ptr,   // [B, n_tables]
    const int16_t* __restrict__ anchor2_ids_ptr,
    const scalar_t* __restrict__ lookup_alt_deltas_ptr,  // [B, n_tables]
    const scalar_t* __restrict__ grad_main_ptr,          // [B, n_tables]
    const scalar_t* __restrict__ grad_alt_ptr,           // [B, n_tables]
    const scalar_t* __restrict__ grad_direct_ptr,        // [B, n_tables] or nullptr
    scalar_t*       __restrict__ x_grad_flat_ptr         // [B * input_dim]
) {
    int32_t tid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int32_t b = tid / n_tables;

    scalar_t delta = lookup_alt_deltas_ptr[tid];
    scalar_t abs_d = delta > static_cast<scalar_t>(0) ? delta : -delta;
    scalar_t sign_d = delta > static_cast<scalar_t>(0) ? static_cast<scalar_t>(1)
                      : (delta < static_cast<scalar_t>(0) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(0));
    scalar_t one_plus_abs = static_cast<scalar_t>(1) + abs_d;
    scalar_t minus_uncertainty_derivative =
        static_cast<scalar_t>(0.5) * sign_d / (one_plus_abs * one_plus_abs);

    scalar_t gmain = grad_main_ptr[tid];
    scalar_t galt  = grad_alt_ptr[tid];
    scalar_t du = (gmain - galt) * minus_uncertainty_derivative;
    if (grad_direct_ptr != nullptr) {
        du += grad_direct_ptr[tid];
    }

    int32_t idx1 = b * input_dim + static_cast<int32_t>(anchor1_ids_ptr[tid]);
    int32_t idx2 = b * input_dim + static_cast<int32_t>(anchor2_ids_ptr[tid]);
    atomicAdd(x_grad_flat_ptr + idx1, du);
    atomicAdd(x_grad_flat_ptr + idx2, -du);
}

// =====================================================================
// BitPermutationLUT dominance-gather forward kernel.
// Reads 1-bit weights packed as int32 blocks (output_nap bits per entry,
// padded to ceil(output_nap/32) blocks), accumulates signed votes into
// per-head canonical-pair dominance outputs. Int all the way — every
// weight ∈ {±1}, sum of K terms is in [-K, K]. Caller applies any float
// scaling (e.g. 0.5/sqrt(N_votes_per_pair)).
// Assumes CANONICAL_DISTINCT sampling → per-slot sign is always +1, so
// no sign tensor is needed.
// One thread per (b, h, p). No atomics.
// =====================================================================

__global__ void bit_perm_lut_dom_gather_fwd_kernel(
    int32_t batch_size,
    int32_t n_heads,
    int32_t tph,
    int32_t n_blocks,          // ceil(output_nap / 32)
    int32_t P,                 // canonical pair count
    int32_t K,                 // inv_idx last dim
    int32_t table_dim,         // 2 ** n_anchor_pairs (entries per table)
    int32_t output_nap,
    const int16_t* __restrict__ lookup_indices_ptr,  // [B, n_heads*tph]
    const int32_t* __restrict__ bit_weights_ptr,     // [n_heads*tph, table_dim, n_blocks]
    const int32_t* __restrict__ inv_idx_ptr,         // [n_heads, P, K]; -1 is padding
    int32_t*       __restrict__ out_ptr              // [B, n_heads, P] int32
) {
    int32_t tid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t total = batch_size * n_heads * P;
    if (tid >= total) return;

    int32_t p = tid % P;
    int32_t rest = tid / P;
    int32_t h = rest % n_heads;
    int32_t b = rest / n_heads;

    int32_t inv_base = h * P * K + p * K;
    int32_t li_base = b * n_heads * tph + h * tph;

    int32_t sum = 0;
    #pragma unroll 1
    for (int32_t k = 0; k < K; ++k) {
        int32_t slot_idx = inv_idx_ptr[inv_base + k];
        if (slot_idx < 0) break;  // padding marker

        int32_t table_within = slot_idx / output_nap;
        int32_t slot_within  = slot_idx - table_within * output_nap;
        int32_t table_global = h * tph + table_within;

        int16_t entry = lookup_indices_ptr[li_base + table_within];

        int32_t block_idx = slot_within >> 5;            // / 32
        int32_t bit_idx   = slot_within & 31;            // % 32
        int32_t w_offset  = (table_global * table_dim + static_cast<int32_t>(entry)) * n_blocks + block_idx;
        int32_t block     = bit_weights_ptr[w_offset];
        int32_t bit       = (block >> bit_idx) & 1;
        int32_t w_val     = 2 * bit - 1;                 // 0 -> -1, 1 -> +1

        sum += w_val;
    }
    out_ptr[b * n_heads * P + h * P + p] = sum;
}

// Backward: project grad_out through bit weights to lookup-index carriers.
// One thread per (b, n) where n = h*tph + t. Each thread accumulates over k
// in [0, output_nap): sign_main/alt ∈ {-1, +1} from bit_weights at entry_main/alt,
// grad_slot = scale * grad_out[b, h, pair_idx_per_slot[h, t, k]].
//   grad_main[b, n]    = sum_k grad_slot * sign_main[b, n, k]
//   grad_alt [b, n, 0] = sum_k grad_slot * sign_alt [b, n, k]
// Bit weights carry NO gradient (discrete); returned through carriers only.
__global__ void bit_perm_lut_dom_gather_bwd_kernel(
    int32_t batch_size,
    int32_t n_heads,
    int32_t tph,
    int32_t n_blocks,          // ceil(output_nap / 32)
    int32_t P,                 // canonical pair count
    int32_t table_dim,         // 2 ** input_nap
    int32_t output_nap,
    float scale,
    const int16_t* __restrict__ lookup_indices_ptr,      // [B, n_heads*tph]
    const int16_t* __restrict__ lookup_alt_indices_ptr,  // [B, n_heads*tph, 1]
    const int32_t* __restrict__ bit_weights_ptr,         // [n_heads*tph, table_dim, n_blocks]
    const int32_t* __restrict__ pair_idx_ptr,            // [n_heads, tph, output_nap]
    const float*   __restrict__ grad_out_ptr,            // [B, n_heads, P]
    float*         __restrict__ grad_main_ptr,           // [B, n_heads*tph]
    float*         __restrict__ grad_alt_ptr             // [B, n_heads*tph, 1]
) {
    int32_t tid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t total = batch_size * n_heads * tph;
    if (tid >= total) return;

    int32_t n = tid % (n_heads * tph);
    int32_t b = tid / (n_heads * tph);
    int32_t h = n / tph;
    int32_t t = n - h * tph;

    int32_t entry_main = static_cast<int32_t>(lookup_indices_ptr[b * n_heads * tph + n]);
    int32_t entry_alt  = static_cast<int32_t>(lookup_alt_indices_ptr[b * n_heads * tph + n]);

    int32_t w_base_main = (n * table_dim + entry_main) * n_blocks;
    int32_t w_base_alt  = (n * table_dim + entry_alt)  * n_blocks;
    int32_t pair_base   = (h * tph + t) * output_nap;
    int32_t grad_base   = b * n_heads * P + h * P;

    float grad_main = 0.0f;
    float grad_alt  = 0.0f;

    int32_t cached_block_idx = -1;
    int32_t block_main = 0;
    int32_t block_alt  = 0;

    for (int32_t k = 0; k < output_nap; ++k) {
        int32_t block_idx = k >> 5;         // / 32
        int32_t bit_idx   = k & 31;         // % 32
        if (block_idx != cached_block_idx) {
            block_main = bit_weights_ptr[w_base_main + block_idx];
            block_alt  = bit_weights_ptr[w_base_alt  + block_idx];
            cached_block_idx = block_idx;
        }
        int32_t bit_main = (block_main >> bit_idx) & 1;
        int32_t bit_alt  = (block_alt  >> bit_idx) & 1;
        float sign_main = 2.0f * static_cast<float>(bit_main) - 1.0f;
        float sign_alt  = 2.0f * static_cast<float>(bit_alt)  - 1.0f;

        int32_t p = pair_idx_ptr[pair_base + k];
        float g_slot = scale * grad_out_ptr[grad_base + p];

        grad_main += g_slot * sign_main;
        grad_alt  += g_slot * sign_alt;
    }

    grad_main_ptr[b * n_heads * tph + n] = grad_main;
    grad_alt_ptr [b * n_heads * tph + n] = grad_alt;
}

// =====================================================================
// Bit packing kernel: pack ±1 signs into int32 blocks of 32 bits each.
// One thread per (n, e, block_idx). Replaces a Python loop that issued
// one GPU op per bit.
//   signs:       [N, table_dim, output_nap] float32; positive → 1, else → 0
//   bit_weights: [N, table_dim, n_blocks]   int32    (output)
// =====================================================================
__global__ void bit_pack_signs_kernel(
    int32_t N, int32_t table_dim, int32_t output_nap, int32_t n_blocks,
    const float*   __restrict__ signs_ptr,
    int32_t*       __restrict__ bit_weights_ptr
) {
    int32_t tid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t total = N * table_dim * n_blocks;
    if (tid >= total) return;

    int32_t bi = tid % n_blocks;
    int32_t e_n = tid / n_blocks;
    int32_t e = e_n % table_dim;
    int32_t n = e_n / table_dim;

    int32_t k_start = bi * 32;
    int32_t k_end = k_start + 32;
    if (k_end > output_nap) k_end = output_nap;

    int32_t block = 0;
    int32_t sign_base = (n * table_dim + e) * output_nap;
    #pragma unroll 1
    for (int32_t k = k_start; k < k_end; ++k) {
        int32_t bit = (signs_ptr[sign_base + k] > 0.0f) ? 1 : 0;
        block |= (bit << (k - k_start));
    }
    bit_weights_ptr[(n * table_dim + e) * n_blocks + bi] = block;
}

// =====================================================================
// Weight-gradient projection: atomicAdd grad_out contributions into
// weight_grad at only the touched (n, entry_main, k) positions.
// One thread per (b, n); caller must pre-zero weight_grad.
//
//   grad_out:       [B, n_heads, P]      float32
//   lookup_indices: [B, n_heads*tph]     int16
//   pair_idx:       [n_heads, tph, output_nap] int32
//   weight_grad:    [n_heads*tph, table_dim, output_nap] float32 (zeroed)
// =====================================================================
__global__ void bit_perm_lut_weight_grad_kernel(
    int32_t B, int32_t n_heads, int32_t tph, int32_t table_dim,
    int32_t output_nap, int32_t P,
    float scale,
    const int16_t* __restrict__ lookup_indices,
    const int32_t* __restrict__ pair_idx,
    const float*   __restrict__ grad_out,
    float*         __restrict__ weight_grad
) {
    int32_t tid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t N = n_heads * tph;
    if (tid >= B * N) return;

    int32_t b = tid / N;
    int32_t n = tid - b * N;
    int32_t h = n / tph;
    int32_t t = n - h * tph;

    int32_t entry = static_cast<int32_t>(lookup_indices[b * N + n]);
    int32_t pair_base = (h * tph + t) * output_nap;
    int32_t grad_base = b * n_heads * P + h * P;
    int32_t wg_base = (n * table_dim + entry) * output_nap;

    #pragma unroll 1
    for (int32_t k = 0; k < output_nap; ++k) {
        int32_t p = pair_idx[pair_base + k];
        float g = scale * grad_out[grad_base + p];
        atomicAdd(&weight_grad[wg_base + k], g);
    }
}

// =====================================================================
// Fused fp8-latent Adam step: dequant latent/m/v, update m/v, update latent,
// clamp + requant latent to fp8 (fixed scale = 1/448), emit new m/v as float32
// scratch (per-table quantization happens in a separate pass).
//
// One thread per element at (n, e, k). Per-table Adam uses shared m/v scales
// from float32 arrays of shape [N, 1, 1] indexed by n.
//
//   latent_fp8 (inplace): [N, td, ona] fp8 (fixed scale AMAX=448)
//   m_fp8                 [N, td, ona] fp8
//   m_scale               [N] float32
//   v_fp8                 [N, td, ona] fp8
//   v_scale               [N] float32
//   weight_grad           [N, td, ona] float32 (zeroed after this call)
//   m_f32_out (written)   [N, td, ona] float32
//   v_f32_out (written)   [N, td, ona] float32
//
// Hparams are packed as fused scalars:
//   lr_over_bias1_times_b2sqrt = -lr * sqrt(bias2) / bias1
//   eps_times_b2sqrt           = eps * sqrt(bias2)
// =====================================================================
__global__ void fused_fp8_adam_kernel(
    int32_t total,
    int32_t per_table,
    float beta1, float beta2,
    float one_minus_b1, float one_minus_b2,
    float eps_times_b2sqrt,
    float lr_step_coef,     // = -lr * sqrt(bias2) / bias1
    __nv_fp8_e4m3* __restrict__ latent_fp8,    // inplace
    const __nv_fp8_e4m3* __restrict__ m_fp8,
    const float*         __restrict__ m_scale,
    const __nv_fp8_e4m3* __restrict__ v_fp8,
    const float*         __restrict__ v_scale,
    const float*         __restrict__ weight_grad,
    float*               __restrict__ m_f32_out,
    float*               __restrict__ v_f32_out
) {
    int32_t tid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int32_t n = tid / per_table;

    const float AMAX = 448.0f;
    // Dequant.
    float latent_f = static_cast<float>(latent_fp8[tid]) / AMAX;
    float m_f = static_cast<float>(m_fp8[tid]) / m_scale[n];
    float v_f = static_cast<float>(v_fp8[tid]) / v_scale[n];
    float g = weight_grad[tid];

    // Adam update.
    float m_new = beta1 * m_f + one_minus_b1 * g;
    float v_new = beta2 * v_f + one_minus_b2 * g * g;
    float denom = sqrtf(fmaxf(v_new, 0.0f)) + eps_times_b2sqrt;
    float latent_new = latent_f + lr_step_coef * m_new / denom;

    // Clamp + requant latent (fixed scale).
    latent_new = fminf(1.0f, fmaxf(-1.0f, latent_new));
    latent_fp8[tid] = __nv_fp8_e4m3(latent_new * AMAX);

    // Emit f32 m, v for per-table quantization in a follow-up kernel.
    m_f32_out[tid] = m_new;
    v_f32_out[tid] = v_new;
}

// WTA (Winner-Take-All) Lookup Kernels
// Input x is contiguous [B, C, N]. One thread per (b, c) pair.
// Forward kernels find the winner (argmax) and n_alternatives runner-ups in a single pass.
// Backward kernels scatter gradients to the winner and alt positions using the uncertainty function.

template <typename scalar_t>
__global__ void wta_lookup_forward_na1_kernel(
    const scalar_t* x_ptr,
    int64_t x_stride0,
    int64_t x_stride1,
    int64_t x_stride2,
    int64_t n_channels,
    int64_t n_inputs,
    int64_t total,
    int64_t* winner_inds_ptr,   // [B, C] flat
    int64_t* alt_inds_ptr,      // [B, C, 1] flat
    scalar_t* alt_deltas_ptr    // [B, C, 1] flat  (winner_val - alt_val)
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= total) return;

    int64_t b = linear_tid / n_channels;
    int64_t c = linear_tid - b * n_channels;
    int64_t x_base = b * x_stride0 + c * x_stride1;
    scalar_t neg_big = -static_cast<scalar_t>(1e30);

    scalar_t max1_val = x_ptr[x_base]; int64_t max1_idx = 0;
    scalar_t max2_val = neg_big;       int64_t max2_idx = 0;

    for (int64_t n = 1; n < n_inputs; ++n) {
        scalar_t val = x_ptr[x_base + n * x_stride2];
        if (val > max1_val) {
            max2_val = max1_val; max2_idx = max1_idx;
            max1_val = val;      max1_idx = n;
        } else if (val > max2_val) {
            max2_val = val; max2_idx = n;
        }
    }

    winner_inds_ptr[linear_tid] = max1_idx;
    alt_inds_ptr[linear_tid]    = max2_idx;
    alt_deltas_ptr[linear_tid]  = max1_val - max2_val;
}

template <typename scalar_t>
__global__ void wta_lookup_forward_na2_kernel(
    const scalar_t* x_ptr,
    int64_t x_stride0,
    int64_t x_stride1,
    int64_t x_stride2,
    int64_t n_channels,
    int64_t n_inputs,
    int64_t total,
    int64_t* winner_inds_ptr,   // [B, C] flat
    int64_t* alt_inds_ptr,      // [B, C, 2] flat
    scalar_t* alt_deltas_ptr    // [B, C, 2] flat
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= total) return;

    int64_t b = linear_tid / n_channels;
    int64_t c = linear_tid - b * n_channels;
    int64_t x_base = b * x_stride0 + c * x_stride1;
    scalar_t neg_big = -static_cast<scalar_t>(1e30);

    scalar_t max1_val = x_ptr[x_base]; int64_t max1_idx = 0;
    scalar_t max2_val = neg_big;       int64_t max2_idx = 0;
    scalar_t max3_val = neg_big;       int64_t max3_idx = 0;

    for (int64_t n = 1; n < n_inputs; ++n) {
        scalar_t val = x_ptr[x_base + n * x_stride2];
        if (val > max1_val) {
            max3_val = max2_val; max3_idx = max2_idx;
            max2_val = max1_val; max2_idx = max1_idx;
            max1_val = val;      max1_idx = n;
        } else if (val > max2_val) {
            max3_val = max2_val; max3_idx = max2_idx;
            max2_val = val;      max2_idx = n;
        } else if (val > max3_val) {
            max3_val = val; max3_idx = n;
        }
    }

    int64_t base_alt = linear_tid * 2;
    winner_inds_ptr[linear_tid]  = max1_idx;
    alt_inds_ptr[base_alt]       = max2_idx;
    alt_inds_ptr[base_alt + 1]   = max3_idx;
    alt_deltas_ptr[base_alt]     = max1_val - max2_val;
    alt_deltas_ptr[base_alt + 1] = max1_val - max3_val;
}

template <typename scalar_t>
__global__ void wta_lookup_forward_na3_kernel(
    const scalar_t* x_ptr,
    int64_t x_stride0,
    int64_t x_stride1,
    int64_t x_stride2,
    int64_t n_channels,
    int64_t n_inputs,
    int64_t total,
    int64_t* winner_inds_ptr,   // [B, C] flat
    int64_t* alt_inds_ptr,      // [B, C, 3] flat
    scalar_t* alt_deltas_ptr    // [B, C, 3] flat
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= total) return;

    int64_t b = linear_tid / n_channels;
    int64_t c = linear_tid - b * n_channels;
    int64_t x_base = b * x_stride0 + c * x_stride1;
    scalar_t neg_big = -static_cast<scalar_t>(1e30);

    scalar_t max1_val = x_ptr[x_base]; int64_t max1_idx = 0;
    scalar_t max2_val = neg_big;       int64_t max2_idx = 0;
    scalar_t max3_val = neg_big;       int64_t max3_idx = 0;
    scalar_t max4_val = neg_big;       int64_t max4_idx = 0;

    for (int64_t n = 1; n < n_inputs; ++n) {
        scalar_t val = x_ptr[x_base + n * x_stride2];
        if (val > max1_val) {
            max4_val = max3_val; max4_idx = max3_idx;
            max3_val = max2_val; max3_idx = max2_idx;
            max2_val = max1_val; max2_idx = max1_idx;
            max1_val = val;      max1_idx = n;
        } else if (val > max2_val) {
            max4_val = max3_val; max4_idx = max3_idx;
            max3_val = max2_val; max3_idx = max2_idx;
            max2_val = val;      max2_idx = n;
        } else if (val > max3_val) {
            max4_val = max3_val; max4_idx = max3_idx;
            max3_val = val;      max3_idx = n;
        } else if (val > max4_val) {
            max4_val = val; max4_idx = n;
        }
    }

    int64_t base_alt = linear_tid * 3;
    winner_inds_ptr[linear_tid]  = max1_idx;
    alt_inds_ptr[base_alt]       = max2_idx;
    alt_inds_ptr[base_alt + 1]   = max3_idx;
    alt_inds_ptr[base_alt + 2]   = max4_idx;
    alt_deltas_ptr[base_alt]     = max1_val - max2_val;
    alt_deltas_ptr[base_alt + 1] = max1_val - max3_val;
    alt_deltas_ptr[base_alt + 2] = max1_val - max4_val;
}

template <typename scalar_t>
__global__ void wta_lookup_backward_kernel(
    int64_t total,
    const int64_t* winner_ids_ptr,
    const int64_t* alt_ids_ptr,
    const scalar_t* alt_deltas_ptr,
    const int64_t* batch_offset_ptr,
    const scalar_t* grad_main_ptr,
    const scalar_t* grad_alt_ptr,
    int64_t grad_main_stride0,
    int64_t grad_main_stride1,
    int64_t n_channels,
    int64_t n_alternatives,
    bool inv_l1,
    scalar_t uncertainty_bias,
    scalar_t* x_grad_flat_ptr
) {
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= total) return;

    int64_t bc = linear_tid / n_alternatives;
    int64_t b  = bc / n_channels;
    int64_t c  = bc - b * n_channels;

    scalar_t delta = alt_deltas_ptr[linear_tid];
    scalar_t minus_uncertainty_derivative;
    if (inv_l1) {
        scalar_t one_plus_abs = static_cast<scalar_t>(1) + lutorch_abs(delta);
        minus_uncertainty_derivative =
            static_cast<scalar_t>(0.5) * lutorch_sign(delta) / (one_plus_abs * one_plus_abs);
    } else {
        scalar_t one_plus_sq = static_cast<scalar_t>(1) + delta * delta;
        minus_uncertainty_derivative = delta / (one_plus_sq * one_plus_sq);
    }

    scalar_t grad_main = grad_main_ptr[b * grad_main_stride0 + c * grad_main_stride1];
    scalar_t grad_alt  = grad_alt_ptr[linear_tid];
    scalar_t du = (grad_main - grad_alt) * minus_uncertainty_derivative
                  / static_cast<scalar_t>(n_alternatives);

    int64_t idx_winner = batch_offset_ptr[linear_tid] + winner_ids_ptr[linear_tid];
    int64_t idx_alt    = batch_offset_ptr[linear_tid] + alt_ids_ptr[linear_tid];
    atomicAdd(x_grad_flat_ptr + idx_winner,  du);
    atomicAdd(x_grad_flat_ptr + idx_alt,    -du);
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
__global__ void lprojection_forward_smooth_weights_kernel(
    int64_t total_bt,
    int64_t n_alternatives,
    bool l1_uncertainty,
    scalar_t uncertainty_bias,
    const scalar_t* lookup_alt_deltas_ptr, // [B*T*A] contiguous flattened
    scalar_t* main_weight_ptr,             // [B*T]
    scalar_t* alt_weight_ptr               // [B*T*A]
) {
    int64_t bt = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bt >= total_bt) {
        return;
    }
    scalar_t inv_n_alt = static_cast<scalar_t>(1.0) / static_cast<scalar_t>(n_alternatives);
    scalar_t uncertainty_sum = static_cast<scalar_t>(0);
    int64_t base = bt * n_alternatives;
    for (int64_t a = 0; a < n_alternatives; ++a) {
        scalar_t d = lookup_alt_deltas_ptr[base + a];
        scalar_t u;
        if (l1_uncertainty) {
            u = uncertainty_bias / (static_cast<scalar_t>(1.0) + lutorch_abs(d));
        } else {
            u = uncertainty_bias / (static_cast<scalar_t>(1.0) + d * d);
        }
        alt_weight_ptr[base + a] = u * inv_n_alt;
        uncertainty_sum += u;
    }
    main_weight_ptr[bt] = static_cast<scalar_t>(1.0) - uncertainty_sum * inv_n_alt;
}

template <typename scalar_t>
__global__ void lprojection_forward_smooth_output_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    int64_t n_alternatives,
    const scalar_t* weights_ptr,                 // [T,E,O] contiguous
    const int64_t* table_indices_flat_ptr,       // [B*T]
    const int64_t* lookup_indices_flat_ptr,      // [B*T]
    const int64_t* table_indices_alt_flat_ptr,   // [B*T*A]
    const int64_t* lookup_alt_indices_flat_ptr,  // [B*T*A]
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

    int64_t table_main = table_indices_flat_ptr[bt];
    int64_t entry_main = lookup_indices_flat_ptr[bt];
    scalar_t acc = weights_ptr[(table_main * n_entries + entry_main) * n_outputs + o] * main_weight_ptr[bt];

    int64_t base = bt * n_alternatives;
    for (int64_t a = 0; a < n_alternatives; ++a) {
        int64_t bta = base + a;
        int64_t table_alt = table_indices_alt_flat_ptr[bta];
        int64_t entry_alt = lookup_alt_indices_flat_ptr[bta];
        scalar_t w = weights_ptr[(table_alt * n_entries + entry_alt) * n_outputs + o];
        acc += w * alt_weight_ptr[bta];
    }

    output_ptr[(b * n_tables + t) * n_outputs + o] = acc;
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

template <typename scalar_t>
__global__ void lprojection_backward_main_carriers_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const scalar_t* weights_ptr,
    const int64_t* table_indices_flat_ptr,
    const int64_t* lookup_indices_flat_ptr,
    int64_t grad_output_stride0,
    int64_t grad_output_stride1,
    int64_t grad_output_stride2,
    scalar_t* lookup_indices_grad_c_grad_ptr
) {
    int64_t bt = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bt >= total_bt) {
        return;
    }
    int64_t b = bt / n_tables;
    int64_t t = bt - b * n_tables;
    int64_t table_main = table_indices_flat_ptr[bt];
    int64_t entry_main = lookup_indices_flat_ptr[bt];
    scalar_t acc_main = static_cast<scalar_t>(0);
    for (int64_t o = 0; o < n_outputs; ++o) {
        scalar_t g = grad_output_ptr[b * grad_output_stride0 + t * grad_output_stride1 + o * grad_output_stride2];
        int64_t widx_main = (table_main * n_entries + entry_main) * n_outputs + o;
        acc_main += g * weights_ptr[widx_main];
    }
    lookup_indices_grad_c_grad_ptr[bt] = acc_main;
}

template <typename scalar_t>
__global__ void lprojection_backward_alt_carriers_kernel(
    int64_t total_bta,
    int64_t n_tables,
    int64_t n_alternatives,
    int64_t n_outputs,
    int64_t n_entries,
    const scalar_t* grad_output_ptr,
    const scalar_t* weights_ptr,
    const int64_t* table_indices_alt_flat_ptr,
    const int64_t* lookup_alt_indices_flat_ptr,
    int64_t grad_output_stride0,
    int64_t grad_output_stride1,
    int64_t grad_output_stride2,
    scalar_t* lookup_alt_indices_grad_c_grad_ptr
) {
    int64_t bta = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bta >= total_bta) {
        return;
    }
    int64_t bt = bta / n_alternatives;
    int64_t b = bt / n_tables;
    int64_t t = bt - b * n_tables;
    int64_t table_alt = table_indices_alt_flat_ptr[bta];
    int64_t entry_alt = lookup_alt_indices_flat_ptr[bta];
    scalar_t acc_alt = static_cast<scalar_t>(0);
    for (int64_t o = 0; o < n_outputs; ++o) {
        scalar_t g = grad_output_ptr[b * grad_output_stride0 + t * grad_output_stride1 + o * grad_output_stride2];
        int64_t widx_alt = (table_alt * n_entries + entry_alt) * n_outputs + o;
        acc_alt += g * weights_ptr[widx_alt];
    }
    lookup_alt_indices_grad_c_grad_ptr[bta] = acc_alt;
}

template <typename scalar_t>
__global__ void lprojection_backward_smooth_weights_kernel(
    int64_t total_bt,
    int64_t n_tables,
    int64_t n_outputs,
    int64_t n_entries,
    int64_t n_alternatives,
    const scalar_t* grad_output_ptr,
    const int64_t* table_indices_flat_ptr,
    const int64_t* lookup_indices_flat_ptr,
    const int64_t* table_indices_alt_flat_ptr,
    const int64_t* lookup_alt_indices_flat_ptr,
    const scalar_t* main_weight_ptr,
    const scalar_t* alt_weight_ptr,
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
    scalar_t g_main = g * main_weight_ptr[bt];
    int64_t table_main = table_indices_flat_ptr[bt];
    int64_t entry_main = lookup_indices_flat_ptr[bt];
    int64_t widx_main = (table_main * n_entries + entry_main) * n_outputs + o;
    atomicAdd(weights_grad_ptr + widx_main, g_main);
    int64_t bt_a_offset = bt * n_alternatives;
    for (int64_t a = 0; a < n_alternatives; ++a) {
        int64_t bta = bt_a_offset + a;
        scalar_t g_alt = g * alt_weight_ptr[bta];
        int64_t table_alt = table_indices_alt_flat_ptr[bta];
        int64_t entry_alt = lookup_alt_indices_flat_ptr[bta];
        int64_t widx_alt = (table_alt * n_entries + entry_alt) * n_outputs + o;
        atomicAdd(weights_grad_ptr + widx_alt, g_alt);
    }
}

// ---------------------------------------------------------------------------
// Fused LUT Attention kernels (na1, non-smooth, n_alternatives=1)
// Two-phase design inspired by PRODUCT kernels:
//   Block: (O, TILE_TPH) — threadIdx.x = output dim, threadIdx.y = table slice
//   Phase 1: all threads cooperatively load feature cache
//   Phase 2: each (o, ty) accumulates weights for output o across its table slice
//   Then reduce partial sums across ty, apply SE, scatter.
// ---------------------------------------------------------------------------

// ---- Forward kernel ----
// Grid: (B*M, H), Block: (O, TILE_TPH)
// Shared: (input_dim + TILE_TPH * O) * sizeof(scalar_t)
template <typename scalar_t>
__global__ void lut_attn_fwd_na1_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weights,
    const int64_t* __restrict__ anchor_a,
    const int64_t* __restrict__ anchor_b,
    const int64_t* __restrict__ pair_rows,
    const int64_t* __restrict__ pair_cols,
    const scalar_t* __restrict__ rel_pe,
    scalar_t* __restrict__ pair_out_buf,       // [B*M, H, O]
    scalar_t* __restrict__ result,             // [B, T, H, O]
    int64_t B, int64_t T, int64_t E,
    int64_t M, int64_t H,
    int64_t tables_per_head,
    int64_t n_entries, int64_t n_anchor_pairs,
    int64_t O, int64_t pos_dim,
    bool causal, bool self_excitement,
    scalar_t cmp_eps,
    int se_mode   // 0=linear, 1=quadratic, 2=exponential
) {
    extern __shared__ char sh_raw[];
    int64_t input_dim = 2 * E + pos_dim;
    scalar_t* feat = reinterpret_cast<scalar_t*>(sh_raw);
    scalar_t* rb   = feat + input_dim;   // reduce buffer [TILE_TPH][O]

    int64_t o  = static_cast<int64_t>(threadIdx.x);   // output dim
    int64_t ty = static_cast<int64_t>(threadIdx.y);    // table slice
    int64_t TILE_TPH = static_cast<int64_t>(blockDim.y);
    int64_t bm = static_cast<int64_t>(blockIdx.x);
    int64_t h  = static_cast<int64_t>(blockIdx.y);
    if (bm >= B * M || h >= H) return;

    int64_t b = bm / M;
    int64_t m = bm - b * M;
    int64_t row_i = pair_rows[m];
    int64_t col_j = pair_cols[m];
    int64_t dist = causal ? (row_i - col_j) : lutorch_abs(row_i - col_j);

    // Phase 1: cooperatively load feature cache
    int64_t linear_tid = ty * O + o;
    int64_t block_size = TILE_TPH * O;
    const scalar_t* xi_ptr  = x + b * T * E + row_i * E;
    const scalar_t* xj_ptr  = x + b * T * E + col_j * E;
    const scalar_t* rpe_ptr = (rel_pe != nullptr) ? rel_pe + dist * pos_dim : nullptr;
    for (int64_t f = linear_tid; f < input_dim; f += block_size) {
        if (f < E)          feat[f] = xi_ptr[f];
        else if (f < 2 * E) feat[f] = xj_ptr[f - E];
        else                 feat[f] = rpe_ptr[f - 2 * E];
    }
    __syncthreads();

    // Phase 2: each thread accumulates weights for its table slice
    scalar_t acc = static_cast<scalar_t>(0);
    int64_t t_start = h * tables_per_head;
    for (int64_t tl = ty; tl < tables_per_head; tl += TILE_TPH) {
        int64_t t = t_start + tl;
        int64_t ta_off = t * n_anchor_pairs;
        int64_t lookup_idx = 0;
        for (int64_t p = 0; p < n_anchor_pairs; ++p) {
            if (feat[anchor_a[ta_off + p]] - feat[anchor_b[ta_off + p]] > cmp_eps)
                lookup_idx |= (static_cast<int64_t>(1) << p);
        }
        acc += weights[t * n_entries * O + lookup_idx * O + o];
    }

    // Reduce partial sums across ty
    rb[ty * O + o] = acc;
    __syncthreads();
    for (int64_t stride = TILE_TPH >> 1; stride > 0; stride >>= 1) {
        if (ty < stride)
            rb[ty * O + o] += rb[(ty + stride) * O + o];
        __syncthreads();
    }

    // rb[0..O-1] now has final per-output accumulated value
    scalar_t f_o = rb[o];
    if (ty == 0)
        pair_out_buf[bm * H * O + h * O + o] = f_o;

    // Self-excitement: y_o = f_o * scale(mean(|f|))
    scalar_t y_o = f_o;
    if (self_excitement) {
        if (ty == 0)
            rb[o] = lutorch_abs(f_o);
        __syncthreads();
        for (int64_t stride = O >> 1; stride > 0; stride >>= 1) {
            if (ty == 0 && o < stride)
                rb[o] += rb[o + stride];
            __syncthreads();
        }
        scalar_t s = rb[0] / static_cast<scalar_t>(O);  // mean_abs
        scalar_t scale;
        if (se_mode == 0)      scale = s;           // linear
        else if (se_mode == 1) scale = s * s;        // quadratic
        else                    scale = exp(s);       // exponential
        y_o = f_o * scale;
    }

    // Scatter-add to result
    if (ty == 0)
        atomicAdd(&result[b * T * H * O + row_i * H * O + h * O + o], y_o);
}

// ---- Backward kernel ----
// Grid: (B*M, H), Block: (O, TILE_TPH)
// Shared: (input_dim + TILE_TPH * O) * sizeof(scalar_t)
template <typename scalar_t>
__global__ void lut_attn_bwd_na1_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weights,
    const int64_t* __restrict__ anchor_a,
    const int64_t* __restrict__ anchor_b,
    const int64_t* __restrict__ pair_rows,
    const int64_t* __restrict__ pair_cols,
    const scalar_t* __restrict__ rel_pe,
    const scalar_t* __restrict__ pair_out_buf,   // [B*M, H, O]
    const scalar_t* __restrict__ result_grad,    // [B, T, H, O]
    scalar_t* __restrict__ weights_grad,
    scalar_t* __restrict__ x_grad,
    scalar_t* __restrict__ rel_pe_grad,
    int64_t B, int64_t T, int64_t E,
    int64_t M, int64_t H,
    int64_t tables_per_head,
    int64_t n_entries, int64_t n_anchor_pairs,
    int64_t O, int64_t pos_dim,
    bool causal, bool self_excitement,
    scalar_t cmp_eps, scalar_t uncertainty_bias,
    int se_mode   // 0=linear, 1=quadratic, 2=exponential
) {
    extern __shared__ char sh_raw[];
    int64_t input_dim = 2 * E + pos_dim;
    scalar_t* feat     = reinterpret_cast<scalar_t*>(sh_raw);
    scalar_t* rb       = feat + input_dim;                      // [TILE_TPH * O]
    scalar_t* sh_xgrad = rb + static_cast<int64_t>(blockDim.y) * O;  // [input_dim]

    int64_t o  = static_cast<int64_t>(threadIdx.x);
    int64_t ty = static_cast<int64_t>(threadIdx.y);
    int64_t TILE_TPH = static_cast<int64_t>(blockDim.y);
    int64_t bm = static_cast<int64_t>(blockIdx.x);
    int64_t h  = static_cast<int64_t>(blockIdx.y);
    if (bm >= B * M || h >= H) return;

    int64_t b = bm / M;
    int64_t m = bm - b * M;
    int64_t row_i = pair_rows[m];
    int64_t col_j = pair_cols[m];
    int64_t dist = causal ? (row_i - col_j) : lutorch_abs(row_i - col_j);

    // Cooperatively zero sh_xgrad and load feature cache
    int64_t linear_tid = ty * O + o;
    int64_t block_size = TILE_TPH * O;
    for (int64_t f = linear_tid; f < input_dim; f += block_size)
        sh_xgrad[f] = static_cast<scalar_t>(0);
    const scalar_t* xi_ptr  = x + b * T * E + row_i * E;
    const scalar_t* xj_ptr  = x + b * T * E + col_j * E;
    const scalar_t* rpe_ptr = (rel_pe != nullptr) ? rel_pe + dist * pos_dim : nullptr;
    for (int64_t f = linear_tid; f < input_dim; f += block_size) {
        if (f < E)          feat[f] = xi_ptr[f];
        else if (f < 2 * E) feat[f] = xj_ptr[f - E];
        else                 feat[f] = rpe_ptr[f - 2 * E];
    }
    __syncthreads();

    // Compute SE gradient (uses rb[0..O-1] only, ty==0 does reductions)
    scalar_t f_o = pair_out_buf[bm * H * O + h * O + o];
    scalar_t g_o = result_grad[b * T * H * O + row_i * H * O + h * O + o];
    scalar_t se_grad_o;
    if (self_excitement) {
        if (ty == 0) rb[o] = f_o * g_o;
        __syncthreads();
        for (int64_t stride = O >> 1; stride > 0; stride >>= 1) {
            if (ty == 0 && o < stride) rb[o] += rb[o + stride];
            __syncthreads();
        }
        scalar_t dot_fg = rb[0];
        __syncthreads();  // ensure all threads read dot_fg before ty=0 overwrites rb
        if (ty == 0) rb[o] = lutorch_abs(f_o);
        __syncthreads();
        for (int64_t stride = O >> 1; stride > 0; stride >>= 1) {
            if (ty == 0 && o < stride) rb[o] += rb[o + stride];
            __syncthreads();
        }
        scalar_t s = rb[0] / static_cast<scalar_t>(O);  // mean_abs
        // se_grad_o = g_o * A + sign(f_o) * B * dot_fg / O
        scalar_t A, B;
        if (se_mode == 0)      { A = s; B = static_cast<scalar_t>(1); }           // linear
        else if (se_mode == 1) { A = s * s; B = static_cast<scalar_t>(2) * s; }   // quadratic
        else                    { scalar_t e = exp(s); A = e; B = e; }             // exponential
        se_grad_o = g_o * A + lutorch_sign(f_o) * B * dot_fg / static_cast<scalar_t>(O);
    } else {
        se_grad_o = g_o;
    }

    // Table loop: TILE_TPH tables processed in parallel per iteration.
    // For O <= 32 each ty-row is one warp → use __shfl_down_sync (no syncs).
    // For O > 32 fall back to shared memory reduction.
    int64_t t_start = h * tables_per_head;
    bool use_warp_shuffle = (O <= 32);

    for (int64_t tl = ty; tl < tables_per_head; tl += TILE_TPH) {
        int64_t t = t_start + tl;
        int64_t ta_off = t * n_anchor_pairs;

        // Compute lookup + min-delta from feat cache
        int64_t lookup_idx = 0;
        scalar_t min_abs_delta = static_cast<scalar_t>(0);
        scalar_t min_delta = static_cast<scalar_t>(0);
        int64_t min_anc_a = 0, min_anc_b = 0, min_bit_pos = 0;

        for (int64_t p = 0; p < n_anchor_pairs; ++p) {
            int64_t a_idx = anchor_a[ta_off + p];
            int64_t b_idx = anchor_b[ta_off + p];
            scalar_t delta = feat[a_idx] - feat[b_idx];
            if (delta > cmp_eps) lookup_idx |= (static_cast<int64_t>(1) << p);
            scalar_t abs_d = lutorch_abs(delta);
            if (p == 0 || abs_d < min_abs_delta) {
                min_abs_delta = abs_d;
                min_delta = delta;
                min_anc_a = a_idx;
                min_anc_b = b_idx;
                min_bit_pos = p;
            }
        }

        // Weight gradient
        int64_t w_main = t * n_entries * O + lookup_idx * O + o;
        atomicAdd(&weights_grad[w_main], se_grad_o);

        // X gradient: reduce se_grad * (W[main] - W[alt]) across outputs
        int64_t alt_idx = lookup_idx ^ (static_cast<int64_t>(1) << min_bit_pos);
        int64_t w_alt = t * n_entries * O + alt_idx * O + o;
        scalar_t val = se_grad_o * (weights[w_main] - weights[w_alt]);

        // Two-stage reduction: warp shuffle first, then cross-warp via shared mem
        // Stage 1: intra-warp reduction with width=O for sub-warp groups
        scalar_t grad_diff;
        {
            unsigned mask = 0xffffffff;
            int shfl_width = (O <= 32) ? static_cast<int>(O) : 32;
            for (int offset = shfl_width >> 1; offset > 0; offset >>= 1)
                val += __shfl_down_sync(mask, val, offset, shfl_width);
        }
        if (O <= 32) {
            grad_diff = val;  // single warp, done
        } else {
            // Stage 2: lane 0 of each warp writes partial sum, combine
            int64_t lane = o & 31;
            int64_t warp_in_row = o >> 5;
            int64_t n_warps = O >> 5;
            if (lane == 0)
                rb[ty * n_warps + warp_in_row] = val;
            __syncthreads();
            if (o == 0) {
                grad_diff = rb[ty * n_warps];
                for (int64_t w = 1; w < n_warps; ++w)
                    grad_diff += rb[ty * n_warps + w];
            }
            __syncthreads();
        }

        if (o == 0) {
            scalar_t one_plus_abs = static_cast<scalar_t>(1) + min_abs_delta;
            scalar_t du = grad_diff * uncertainty_bias * lutorch_sign(min_delta)
                          / (one_plus_abs * one_plus_abs);
            // Accumulate into shared memory (fast ~10 cycle atomics)
            atomicAdd(&sh_xgrad[min_anc_a], du);
            atomicAdd(&sh_xgrad[min_anc_b], -du);
        }
    }

    // Flush sh_xgrad to global memory — one sync, then cooperative write
    __syncthreads();
    for (int64_t f = linear_tid; f < input_dim; f += block_size) {
        scalar_t g = sh_xgrad[f];
        if (g != static_cast<scalar_t>(0)) {
            if (f < E)
                atomicAdd(&x_grad[b * T * E + row_i * E + f], g);
            else if (f < 2 * E)
                atomicAdd(&x_grad[b * T * E + col_j * E + (f - E)], g);
            else if (rel_pe_grad != nullptr)
                atomicAdd(&rel_pe_grad[dist * pos_dim + (f - 2 * E)], g);
        }
    }
}
// =====================================================================
// PermutationalLut fused kernels:
//   forward:  raw [B, H*T, P] -> out [B, H, N]   (signed vote + scatter)
//   backward: grad_out [B, H, N] -> grad_raw [B, H*T, P]
//
// soft_mode: 0=sigmoid, 1=rational, 2=ste (hard fwd, rational bwd)
// =====================================================================

template <typename scalar_t>
static __device__ __forceinline__ scalar_t perm_signed_vote(
    scalar_t raw, scalar_t inv_T, scalar_t T_val, int soft_mode
) {
    if (soft_mode == 0) {
        // sigmoid(raw / T) - 0.5
        scalar_t s = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + ::exp(-raw * inv_T));
        return s - static_cast<scalar_t>(0.5);
    }
    // rational and ste both have the same forward in soft_mode==1; ste is handled by the caller
    // for soft_mode==2 (ste), the caller passes the hard sign(raw)*0.5 directly.
    // 0.5 * raw / (T + |raw|)
    scalar_t a = raw >= static_cast<scalar_t>(0) ? raw : -raw;
    return static_cast<scalar_t>(0.5) * raw / (T_val + a);
}

template <typename scalar_t>
static __device__ __forceinline__ scalar_t perm_signed_vote_jac(
    scalar_t raw, scalar_t inv_T, scalar_t T_val, int soft_mode
) {
    // Returns d(d)/d(raw)
    if (soft_mode == 0) {
        scalar_t s = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + ::exp(-raw * inv_T));
        return inv_T * s * (static_cast<scalar_t>(1) - s);  // (1/T) * s * (1-s)
    }
    // rational (also used for ste backward): d/draw [0.5 * raw / (T + |raw|)] = 0.5 * T / (T + |raw|)^2
    scalar_t a = raw >= static_cast<scalar_t>(0) ? raw : -raw;
    scalar_t denom = T_val + a;
    return static_cast<scalar_t>(0.5) * T_val / (denom * denom);
}

// Forward: each thread handles one (b, h, t, p) element.
// raw layout: [B, H*T, P] contiguous (B major).
// idx_a/idx_b layout: [H, T*P] contiguous.
// out layout: [B, H, N] contiguous.
template <typename scalar_t>
__global__ void perm_lut_fwd_kernel(
    int64_t B,
    int64_t H,
    int64_t T,   // tph
    int64_t P,   // output_nap
    int64_t N,   // n_outputs
    int soft_mode,
    scalar_t T_val,
    scalar_t inv_T,
    const scalar_t* __restrict__ raw_ptr,    // [B, H*T, P]
    const int64_t*   __restrict__ idx_a_ptr,  // [H, T*P]
    const int64_t*   __restrict__ idx_b_ptr,  // [H, T*P]
    scalar_t*        __restrict__ out_ptr     // [B, H, N]
) {
    int64_t total = B * H * T * P;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t p = tid % P;
    int64_t rest = tid / P;
    int64_t t = rest % T;
    int64_t rest2 = rest / T;
    int64_t h = rest2 % H;
    int64_t b = rest2 / H;

    int64_t raw_idx = (b * H + h) * T * P + t * P + p;
    int64_t idx_pos = h * (T * P) + t * P + p;
    int64_t a_dim = idx_a_ptr[idx_pos];
    int64_t b_dim = idx_b_ptr[idx_pos];

    scalar_t raw = raw_ptr[raw_idx];
    scalar_t d;
    if (soft_mode == 2) {
        // STE hard forward: sign(raw) * 0.5
        d = raw > static_cast<scalar_t>(0) ? static_cast<scalar_t>(0.5)
                                            : static_cast<scalar_t>(-0.5);
    } else {
        d = perm_signed_vote<scalar_t>(raw, inv_T, T_val, soft_mode);
    }

    int64_t out_base = (b * H + h) * N;
    atomicAdd(out_ptr + out_base + a_dim, d);
    atomicAdd(out_ptr + out_base + b_dim, -d);
}

// Backward: each thread handles one (b, h, t, p) element.
// grad_raw[b, h*T+t, p] = (grad_out[b, h, idx_a] - grad_out[b, h, idx_b]) * d(d)/d(raw)
// No atomics: each thread writes to a unique grad_raw entry.
template <typename scalar_t>
__global__ void perm_lut_bwd_kernel(
    int64_t B,
    int64_t H,
    int64_t T,
    int64_t P,
    int64_t N,
    int soft_mode,
    scalar_t T_val,
    scalar_t inv_T,
    const scalar_t* __restrict__ grad_out_ptr, // [B, H, N]
    const scalar_t* __restrict__ raw_ptr,      // [B, H*T, P]
    const int64_t*   __restrict__ idx_a_ptr,    // [H, T*P]
    const int64_t*   __restrict__ idx_b_ptr,    // [H, T*P]
    scalar_t*        __restrict__ grad_raw_ptr  // [B, H*T, P]
) {
    int64_t total = B * H * T * P;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t p = tid % P;
    int64_t rest = tid / P;
    int64_t t = rest % T;
    int64_t rest2 = rest / T;
    int64_t h = rest2 % H;
    int64_t b = rest2 / H;

    int64_t raw_idx = (b * H + h) * T * P + t * P + p;
    int64_t idx_pos = h * (T * P) + t * P + p;
    int64_t a_dim = idx_a_ptr[idx_pos];
    int64_t b_dim = idx_b_ptr[idx_pos];

    int64_t out_base = (b * H + h) * N;
    scalar_t ga = grad_out_ptr[out_base + a_dim];
    scalar_t gb = grad_out_ptr[out_base + b_dim];
    scalar_t upstream = ga - gb;  // d(loss)/d(d) = +1 for a, -1 for b

    scalar_t raw = raw_ptr[raw_idx];
    // For ste, backward uses rational (soft_mode==1) Jacobian
    int jac_mode = (soft_mode == 2) ? 1 : soft_mode;
    scalar_t jac = perm_signed_vote_jac<scalar_t>(raw, inv_T, T_val, jac_mode);

    grad_raw_ptr[raw_idx] = upstream * jac;
}

// =====================================================================
// Dominance-path variants: each pair slot contributes to ONE output
// (canonical pair index) with a precomputed sign. Halves atomics vs
// the remap trick that uses the non-dominance kernel with a dummy bucket.
// =====================================================================

// Forward: each thread handles one (b, h, t, p_slot) element.
// raw layout:      [B, H*T, P_slots] contiguous.
// pair_idx layout: [H, T*P_slots] (int64, canonical pair index ∈ [0, P_out))
// sign layout:     [H, T*P_slots] (scalar, ±1)
// out layout:      [B, H, P_out]
template <typename scalar_t>
__global__ void perm_lut_dom_fwd_kernel(
    int64_t B,
    int64_t H,
    int64_t T,         // tph
    int64_t P_slots,   // output_nap
    int64_t P_out,     // canonical pair count
    int soft_mode,
    scalar_t T_val,
    scalar_t inv_T,
    const scalar_t* __restrict__ raw_ptr,
    const int64_t*   __restrict__ pair_idx_ptr,
    const scalar_t*  __restrict__ sign_ptr,
    scalar_t*        __restrict__ out_ptr
) {
    int64_t total = B * H * T * P_slots;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t p = tid % P_slots;
    int64_t rest = tid / P_slots;
    int64_t t = rest % T;
    int64_t rest2 = rest / T;
    int64_t h = rest2 % H;
    int64_t b = rest2 / H;

    int64_t raw_idx = (b * H + h) * T * P_slots + t * P_slots + p;
    int64_t idx_pos = h * (T * P_slots) + t * P_slots + p;
    int64_t p_out = pair_idx_ptr[idx_pos];
    scalar_t s = sign_ptr[idx_pos];

    scalar_t raw = raw_ptr[raw_idx];
    scalar_t d;
    if (soft_mode == 2) {
        d = raw > static_cast<scalar_t>(0) ? static_cast<scalar_t>(0.5)
                                            : static_cast<scalar_t>(-0.5);
    } else {
        d = perm_signed_vote<scalar_t>(raw, inv_T, T_val, soft_mode);
    }

    int64_t out_base = (b * H + h) * P_out;
    atomicAdd(out_ptr + out_base + p_out, d * s);
}

// Gather-style forward: one thread per output (b, h, p).
// No atomics — each thread accumulates contributions from ~N_votes_per_pair
// slots that map to canonical pair p, then writes a unique output location.
// inv_idx  [H, P, K]: raw-slot index (or -1 padding) for each (h, p, k).
// inv_sign [H, P, K]: ±1 sign for that contribution.
template <typename scalar_t>
__global__ void perm_lut_dom_gather_fwd_kernel(
    int64_t B,
    int64_t H,
    int64_t TP,        // tph * output_nap
    int64_t P,         // canonical pair count
    int64_t K,         // padded max contributions per pair (inv_idx last dim)
    int soft_mode,
    scalar_t T_val,
    scalar_t inv_T,
    const scalar_t* __restrict__ raw_ptr,      // [B, H*TP] (flat over H and TP)
    const int64_t*   __restrict__ inv_idx_ptr,  // [H, P, K]
    const scalar_t*  __restrict__ inv_sign_ptr, // [H, P, K]
    scalar_t*        __restrict__ out_ptr       // [B, H, P]
) {
    int64_t total = B * H * P;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t p = tid % P;
    int64_t rest = tid / P;
    int64_t h = rest % H;
    int64_t b = rest / H;

    int64_t raw_base = b * H * TP + h * TP;
    int64_t inv_base = h * P * K + p * K;

    scalar_t sum = static_cast<scalar_t>(0);
    #pragma unroll 1
    for (int64_t k = 0; k < K; ++k) {
        int64_t slot = inv_idx_ptr[inv_base + k];
        if (slot < 0) break;  // padding marker; remaining are all -1
        scalar_t s = inv_sign_ptr[inv_base + k];
        scalar_t raw = raw_ptr[raw_base + slot];
        scalar_t d;
        if (soft_mode == 2) {
            d = raw > static_cast<scalar_t>(0) ? static_cast<scalar_t>(0.5)
                                                : static_cast<scalar_t>(-0.5);
        } else {
            d = perm_signed_vote<scalar_t>(raw, inv_T, T_val, soft_mode);
        }
        sum += s * d;
    }
    out_ptr[b * H * P + h * P + p] = sum;
}

// Backward: each thread handles one (b, h, t, p_slot) element.
// grad_d = sign * grad_out[pair_idx]; grad_raw = grad_d * jac(raw)
template <typename scalar_t>
__global__ void perm_lut_dom_bwd_kernel(
    int64_t B,
    int64_t H,
    int64_t T,
    int64_t P_slots,
    int64_t P_out,
    int soft_mode,
    scalar_t T_val,
    scalar_t inv_T,
    const scalar_t* __restrict__ grad_out_ptr,
    const scalar_t* __restrict__ raw_ptr,
    const int64_t*   __restrict__ pair_idx_ptr,
    const scalar_t*  __restrict__ sign_ptr,
    scalar_t*        __restrict__ grad_raw_ptr
) {
    int64_t total = B * H * T * P_slots;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t p = tid % P_slots;
    int64_t rest = tid / P_slots;
    int64_t t = rest % T;
    int64_t rest2 = rest / T;
    int64_t h = rest2 % H;
    int64_t b = rest2 / H;

    int64_t raw_idx = (b * H + h) * T * P_slots + t * P_slots + p;
    int64_t idx_pos = h * (T * P_slots) + t * P_slots + p;
    int64_t p_out = pair_idx_ptr[idx_pos];
    scalar_t s = sign_ptr[idx_pos];

    int64_t out_base = (b * H + h) * P_out;
    scalar_t upstream = s * grad_out_ptr[out_base + p_out];

    scalar_t raw = raw_ptr[raw_idx];
    int jac_mode = (soft_mode == 2) ? 1 : soft_mode;
    scalar_t jac = perm_signed_vote_jac<scalar_t>(raw, inv_T, T_val, jac_mode);

    grad_raw_ptr[raw_idx] = upstream * jac;
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
            LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP,
            "lutorch::anchor_pairs_lookup_forward"
        );
        profiler.register_operation_type(
            LUTORCH_MANAGER_ANCHOR_PAIRS_EVAL_FORWARD_PROFILER_OP,
            "lutorch::anchor_pairs_lookup_eval_forward"
        );
        profiler.register_operation_type(
            LUTORCH_MANAGER_ANCHOR_PAIRS_BACKWARD_PROFILER_OP,
            "lutorch::anchor_pairs_lookup_backward"
        );
        profiler.register_operation_type(
            LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP,
            "lutorch::lprojection_backward"
        );
        profiler.register_operation_type(
            LUTORCH_MANAGER_LPROJECTION_FORWARD_SMOOTH_PROFILER_OP,
            "lutorch::lprojection_forward_smooth"
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
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);

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
            anchor_pairs_lookup_forward_na1_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);
        return out;
    }

    torch::Tensor
    anchor_pairs_lookup_eval_forward(
        const torch::Tensor& x,
        const torch::Tensor& anchor_pairs_a,
        const torch::Tensor& anchor_pairs_b,
        double cmp_eps,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_EVAL_FORWARD_PROFILER_OP);

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

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_eval_forward_kernel", [&] {
            anchor_pairs_lookup_eval_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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

        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_EVAL_FORWARD_PROFILER_OP);
        return lookup_indices;
    }

    py::tuple
    anchor_pairs_lookup_forward_na2(
        const torch::Tensor& x,
        const torch::Tensor& anchor_pairs_a,
        const torch::Tensor& anchor_pairs_b,
        double cmp_eps,
        bool save_anchor_ids = true,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);

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
        if (n_anchor_pairs < 2) {
            throw py::value_error("n_alternatives=2 requires n_anchor_pairs >= 2");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }

        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto opts_x = torch::TensorOptions().dtype(x.dtype()).device(x.device());

        torch::Tensor lookup_indices = torch::empty({batch_size, n_tables}, opts_i64);
        torch::Tensor lookup_alt_indices = torch::empty({batch_size, n_tables, 2}, opts_i64);
        torch::Tensor lookup_alt_deltas = torch::empty({batch_size, n_tables, 2}, opts_x);

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
            anchor1_ids = torch::empty({batch_size, n_tables, 2}, opts_i64);
            anchor2_ids = torch::empty({batch_size, n_tables, 2}, opts_i64);
            anchor1_ids_ptr = reinterpret_cast<int64_t*>(anchor1_ids.data_ptr());
            anchor2_ids_ptr = reinterpret_cast<int64_t*>(anchor2_ids.data_ptr());
        }

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_forward_na2_kernel", [&] {
            anchor_pairs_lookup_forward_na2_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        if (save_anchor_ids) {
            out[3] = anchor1_ids;
            out[4] = anchor2_ids;
        } else {
            out[3] = py::none();
            out[4] = py::none();
        }
        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);
        return out;
    }

    py::tuple
    anchor_pairs_lookup_forward_na3(
        const torch::Tensor& x,
        const torch::Tensor& anchor_pairs_a,
        const torch::Tensor& anchor_pairs_b,
        double cmp_eps,
        bool save_anchor_ids = true,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);

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
        if (x.device() != anchor_pairs_a.device() || x.device() != anchor_pairs_b.device()) {
            throw py::value_error("All tensors must be on the same CUDA device");
        }

        const int64_t batch_size = x.size(0);
        const int64_t x_stride0 = x.stride(0);
        const int64_t x_stride1 = x.stride(1);
        const int64_t n_tables = anchor_pairs_a.size(0);
        const int64_t n_anchor_pairs = anchor_pairs_a.size(1);
        if (n_anchor_pairs < 3) {
            throw py::value_error("n_alternatives=3 requires n_anchor_pairs >= 3");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }

        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto opts_x = torch::TensorOptions().dtype(x.dtype()).device(x.device());

        torch::Tensor lookup_indices = torch::empty({batch_size, n_tables}, opts_i64);
        torch::Tensor lookup_alt_indices = torch::empty({batch_size, n_tables, 3}, opts_i64);
        torch::Tensor lookup_alt_deltas = torch::empty({batch_size, n_tables, 3}, opts_x);

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
            anchor1_ids = torch::empty({batch_size, n_tables, 3}, opts_i64);
            anchor2_ids = torch::empty({batch_size, n_tables, 3}, opts_i64);
            anchor1_ids_ptr = reinterpret_cast<int64_t*>(anchor1_ids.data_ptr());
            anchor2_ids_ptr = reinterpret_cast<int64_t*>(anchor2_ids.data_ptr());
        }

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_forward_na3_kernel", [&] {
            anchor_pairs_lookup_forward_na3_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        if (save_anchor_ids) {
            out[3] = anchor1_ids;
            out[4] = anchor2_ids;
        } else {
            out[3] = py::none();
            out[4] = py::none();
        }
        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);
        return out;
    }

    // Forward for generic n_alternatives == n_anchor_pairs (no sorting).
    py::tuple
    anchor_pairs_lookup_forward_all(
        const torch::Tensor& x,
        const torch::Tensor& anchor_pairs_a,
        const torch::Tensor& anchor_pairs_b,
        double cmp_eps,
        bool save_anchor_ids = true,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);

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
        if (x.device() != anchor_pairs_a.device() || x.device() != anchor_pairs_b.device()) {
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
        torch::Tensor lookup_alt_indices = torch::empty({batch_size, n_tables, n_anchor_pairs}, opts_i64);
        torch::Tensor lookup_alt_deltas = torch::empty({batch_size, n_tables, n_anchor_pairs}, opts_x);

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
            anchor1_ids = torch::empty({batch_size, n_tables, n_anchor_pairs}, opts_i64);
            anchor2_ids = torch::empty({batch_size, n_tables, n_anchor_pairs}, opts_i64);
            anchor1_ids_ptr = reinterpret_cast<int64_t*>(anchor1_ids.data_ptr());
            anchor2_ids_ptr = reinterpret_cast<int64_t*>(anchor2_ids.data_ptr());
        }

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_forward_all_kernel", [&] {
            anchor_pairs_lookup_forward_all_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        if (save_anchor_ids) {
            out[3] = anchor1_ids;
            out[4] = anchor2_ids;
        } else {
            out[3] = py::none();
            out[4] = py::none();
        }
        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_PROFILER_OP);
        return out;
    }

    // NOTE: eval for all n_alternatives cases is handled by
    // anchor_pairs_lookup_eval_forward and the shared CUDA kernel
    // anchor_pairs_lookup_eval_forward_kernel.

    py::tuple
    lprojection_forward_smooth(
        const torch::Tensor& weights,
        const torch::Tensor& lookup_indices,
        const torch::Tensor& lookup_alt_indices,
        const torch::Tensor& lookup_alt_deltas,
        const torch::Tensor& table_indices_flat,
        const torch::Tensor& table_indices_alt_flat,
        bool l1_uncertainty,
        double uncertainty_bias = 0.5,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_LPROJECTION_FORWARD_SMOOTH_PROFILER_OP);

        if (!weights.is_cuda() || !lookup_indices.is_cuda() || !lookup_alt_indices.is_cuda() ||
            !lookup_alt_deltas.is_cuda() || !table_indices_flat.is_cuda() || !table_indices_alt_flat.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!weights.is_floating_point() || lookup_alt_deltas.dtype() != weights.dtype()) {
            throw py::value_error("weights/lookup_alt_deltas must be floating with same dtype");
        }
        if (lookup_indices.dtype() != torch::kInt64 || lookup_alt_indices.dtype() != torch::kInt64 ||
            table_indices_flat.dtype() != torch::kInt64 || table_indices_alt_flat.dtype() != torch::kInt64) {
            throw py::value_error("indices tensors must be int64");
        }
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

        AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "lprojection_forward_smooth", [&] {
            lprojection_forward_smooth_weights_kernel<scalar_t><<<blocks_bt, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total_bt,
                n_alternatives,
                l1_uncertainty,
                static_cast<scalar_t>(uncertainty_bias),
                reinterpret_cast<const scalar_t*>(lookup_alt_deltas.data_ptr()),
                reinterpret_cast<scalar_t*>(main_weight.data_ptr()),
                reinterpret_cast<scalar_t*>(alt_weight.data_ptr())
            );
            lprojection_forward_smooth_output_kernel<scalar_t><<<blocks_out, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                n_alternatives,
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_indices.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_alt_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_alt_indices.data_ptr()),
                reinterpret_cast<const scalar_t*>(main_weight.data_ptr()),
                reinterpret_cast<const scalar_t*>(alt_weight.data_ptr()),
                reinterpret_cast<scalar_t*>(output.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        PROF_END(LUTORCH_MANAGER_LPROJECTION_FORWARD_SMOOTH_PROFILER_OP);
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
        double uncertainty_bias = 0.5,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_ANCHOR_PAIRS_BACKWARD_PROFILER_OP);

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

        auto opts_x = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor x_grad_flat = torch::zeros({batch_size * input_dim}, opts_x);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_tables * n_alternatives;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        int64_t grad_main_stride0 = grad_main.stride(0);
        int64_t grad_main_stride1 = grad_main.stride(1);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "anchor_pairs_lookup_backward_all_kernel", [&] {
            anchor_pairs_lookup_backward_all_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
                static_cast<scalar_t>(uncertainty_bias),
                reinterpret_cast<scalar_t*>(x_grad_flat.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        PROF_END(LUTORCH_MANAGER_ANCHOR_PAIRS_BACKWARD_PROFILER_OP);
        return x_grad_flat;
    }

    // ---- WTA Lookup ----

    py::tuple
    wta_lookup_forward_na1(
        const torch::Tensor& x,
        int64_t threads_per_block = 256
    ) {
        if (x.dim() != 3) throw py::value_error("x must be 3D [batch_size, n_channels, n_inputs]");
        if (!x.is_cuda()) throw py::value_error("x must be CUDA tensor");
        if (!x.is_floating_point()) throw py::value_error("x must be floating point tensor");
        if (x.size(2) < 2) throw py::value_error("n_inputs must be >= 2 for n_alternatives=1");
        if (threads_per_block <= 0 || threads_per_block > 1024)
            throw py::value_error("threads_per_block must be in range [1, 1024]");

        const int64_t batch_size = x.size(0);
        const int64_t n_channels = x.size(1);
        const int64_t n_inputs   = x.size(2);
        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto opts_x   = torch::TensorOptions().dtype(x.dtype()).device(x.device());

        torch::Tensor winner_inds = torch::empty({batch_size, n_channels},    opts_i64);
        torch::Tensor alt_inds    = torch::empty({batch_size, n_channels, 1}, opts_i64);
        torch::Tensor alt_deltas  = torch::empty({batch_size, n_channels, 1}, opts_x);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_channels;
        int threads = static_cast<int>(threads_per_block);
        int blocks  = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "wta_lookup_forward_na1_kernel", [&] {
            wta_lookup_forward_na1_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t*>(x.data_ptr()),
                x.stride(0), x.stride(1), x.stride(2),
                n_channels, n_inputs, total,
                reinterpret_cast<int64_t*>(winner_inds.data_ptr()),
                reinterpret_cast<int64_t*>(alt_inds.data_ptr()),
                reinterpret_cast<scalar_t*>(alt_deltas.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        py::tuple out(3);
        out[0] = winner_inds;
        out[1] = alt_inds;
        out[2] = alt_deltas;
        return out;
    }

    py::tuple
    wta_lookup_forward_na2(
        const torch::Tensor& x,
        int64_t threads_per_block = 256
    ) {
        if (x.dim() != 3) throw py::value_error("x must be 3D [batch_size, n_channels, n_inputs]");
        if (!x.is_cuda()) throw py::value_error("x must be CUDA tensor");
        if (!x.is_floating_point()) throw py::value_error("x must be floating point tensor");
        if (x.size(2) < 3) throw py::value_error("n_inputs must be >= 3 for n_alternatives=2");
        if (threads_per_block <= 0 || threads_per_block > 1024)
            throw py::value_error("threads_per_block must be in range [1, 1024]");

        const int64_t batch_size = x.size(0);
        const int64_t n_channels = x.size(1);
        const int64_t n_inputs   = x.size(2);
        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto opts_x   = torch::TensorOptions().dtype(x.dtype()).device(x.device());

        torch::Tensor winner_inds = torch::empty({batch_size, n_channels},    opts_i64);
        torch::Tensor alt_inds    = torch::empty({batch_size, n_channels, 2}, opts_i64);
        torch::Tensor alt_deltas  = torch::empty({batch_size, n_channels, 2}, opts_x);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_channels;
        int threads = static_cast<int>(threads_per_block);
        int blocks  = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "wta_lookup_forward_na2_kernel", [&] {
            wta_lookup_forward_na2_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t*>(x.data_ptr()),
                x.stride(0), x.stride(1), x.stride(2),
                n_channels, n_inputs, total,
                reinterpret_cast<int64_t*>(winner_inds.data_ptr()),
                reinterpret_cast<int64_t*>(alt_inds.data_ptr()),
                reinterpret_cast<scalar_t*>(alt_deltas.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        py::tuple out(3);
        out[0] = winner_inds;
        out[1] = alt_inds;
        out[2] = alt_deltas;
        return out;
    }

    py::tuple
    wta_lookup_forward_na3(
        const torch::Tensor& x,
        int64_t threads_per_block = 256
    ) {
        if (x.dim() != 3) throw py::value_error("x must be 3D [batch_size, n_channels, n_inputs]");
        if (!x.is_cuda()) throw py::value_error("x must be CUDA tensor");
        if (!x.is_floating_point()) throw py::value_error("x must be floating point tensor");
        if (x.size(2) < 4) throw py::value_error("n_inputs must be >= 4 for n_alternatives=3");
        if (threads_per_block <= 0 || threads_per_block > 1024)
            throw py::value_error("threads_per_block must be in range [1, 1024]");

        const int64_t batch_size = x.size(0);
        const int64_t n_channels = x.size(1);
        const int64_t n_inputs   = x.size(2);
        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto opts_x   = torch::TensorOptions().dtype(x.dtype()).device(x.device());

        torch::Tensor winner_inds = torch::empty({batch_size, n_channels},    opts_i64);
        torch::Tensor alt_inds    = torch::empty({batch_size, n_channels, 3}, opts_i64);
        torch::Tensor alt_deltas  = torch::empty({batch_size, n_channels, 3}, opts_x);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_channels;
        int threads = static_cast<int>(threads_per_block);
        int blocks  = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "wta_lookup_forward_na3_kernel", [&] {
            wta_lookup_forward_na3_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t*>(x.data_ptr()),
                x.stride(0), x.stride(1), x.stride(2),
                n_channels, n_inputs, total,
                reinterpret_cast<int64_t*>(winner_inds.data_ptr()),
                reinterpret_cast<int64_t*>(alt_inds.data_ptr()),
                reinterpret_cast<scalar_t*>(alt_deltas.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());

        py::tuple out(3);
        out[0] = winner_inds;
        out[1] = alt_inds;
        out[2] = alt_deltas;
        return out;
    }

    torch::Tensor
    wta_lookup_backward(
        const torch::Tensor& x,
        const torch::Tensor& winner_ids,
        const torch::Tensor& alt_ids,
        const torch::Tensor& alt_deltas,
        const torch::Tensor& batch_offset,
        const torch::Tensor& grad_main,
        const torch::Tensor& grad_alt,
        int64_t n_alternatives,
        bool inv_l1,
        double uncertainty_bias = 0.5,
        int64_t threads_per_block = 256
    ) {
        if (x.dim() != 3) throw py::value_error("x must be 3D [batch_size, n_channels, n_inputs]");
        if (!x.is_cuda()) throw py::value_error("x must be CUDA tensor");
        if (!x.is_floating_point()) throw py::value_error("x must be floating point tensor");
        if (winner_ids.dtype() != torch::kInt64 || alt_ids.dtype() != torch::kInt64)
            throw py::value_error("winner_ids and alt_ids must be int64");
        if (batch_offset.dtype() != torch::kInt64)
            throw py::value_error("batch_offset must be int64");
        if (grad_main.dtype() != x.dtype() || grad_alt.dtype() != x.dtype() || alt_deltas.dtype() != x.dtype())
            throw py::value_error("grad_main, grad_alt, alt_deltas must have the same dtype as x");
        if (winner_ids.numel() != alt_ids.numel() ||
            winner_ids.numel() != alt_deltas.numel() ||
            winner_ids.numel() != batch_offset.numel() ||
            winner_ids.numel() != grad_alt.numel())
            throw py::value_error("winner_ids, alt_ids, alt_deltas, batch_offset, grad_alt must have equal numel");
        if (n_alternatives < 1 || n_alternatives > 3)
            throw py::value_error("n_alternatives must be 1, 2, or 3");
        if (threads_per_block <= 0 || threads_per_block > 1024)
            throw py::value_error("threads_per_block must be in range [1, 1024]");

        const int64_t batch_size = x.size(0);
        const int64_t n_channels = x.size(1);
        const int64_t n_inputs   = x.size(2);
        auto opts_x = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor x_grad_flat = torch::zeros({batch_size * n_channels * n_inputs}, opts_x);

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);
        int64_t total = batch_size * n_channels * n_alternatives;
        int threads = static_cast<int>(threads_per_block);
        int blocks  = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "wta_lookup_backward_kernel", [&] {
            wta_lookup_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total,
                reinterpret_cast<const int64_t*>(winner_ids.data_ptr()),
                reinterpret_cast<const int64_t*>(alt_ids.data_ptr()),
                reinterpret_cast<const scalar_t*>(alt_deltas.data_ptr()),
                reinterpret_cast<const int64_t*>(batch_offset.data_ptr()),
                reinterpret_cast<const scalar_t*>(grad_main.data_ptr()),
                reinterpret_cast<const scalar_t*>(grad_alt.data_ptr()),
                grad_main.stride(0), grad_main.stride(1),
                n_channels, n_alternatives, inv_l1,
                static_cast<scalar_t>(uncertainty_bias),
                reinterpret_cast<scalar_t*>(x_grad_flat.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
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
        PROF_START(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);

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
            lprojection_backward_na1_nonsmooth_weights_kernel<scalar_t><<<blocks_w, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
            lprojection_backward_na1_carriers_kernel<scalar_t><<<blocks_c, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        PROF_END(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);
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
        PROF_START(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);

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
            lprojection_backward_na1_smooth_weights_kernel<scalar_t><<<blocks_w, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
            lprojection_backward_na1_carriers_kernel<scalar_t><<<blocks_c, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        PROF_END(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);
        return py::make_tuple(weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad);
    }

    py::tuple
    lprojection_backward_nonsmooth(
        const torch::Tensor& grad_output,
        const torch::Tensor& weights,
        const torch::Tensor& lookup_indices,
        const torch::Tensor& lookup_alt_indices,
        const torch::Tensor& table_indices_flat,
        const torch::Tensor& table_indices_alt_flat,
        int64_t threads_per_block = 256
    ) {
        PROF_START(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);
        if (!grad_output.is_cuda() || !weights.is_cuda() || !lookup_indices.is_cuda() ||
            !lookup_alt_indices.is_cuda() || !table_indices_flat.is_cuda() || !table_indices_alt_flat.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (grad_output.dtype() != weights.dtype() || !grad_output.is_floating_point()) {
            throw py::value_error("grad_output/weights must be same floating dtype");
        }
        if (lookup_indices.dtype() != torch::kInt64 || lookup_alt_indices.dtype() != torch::kInt64 ||
            table_indices_flat.dtype() != torch::kInt64 || table_indices_alt_flat.dtype() != torch::kInt64) {
            throw py::value_error("indices tensors must be int64");
        }
        if (lookup_indices.dim() != 2 || lookup_alt_indices.dim() != 3) {
            throw py::value_error("lookup_indices must be [B,T], lookup_alt_indices must be [B,T,A]");
        }
        if (weights.dim() != 3 || grad_output.dim() != 3) {
            throw py::value_error("weights must be [T,E,O], grad_output must be [B,T,O]");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }
        if (grad_output.device() != weights.device() || lookup_indices.device() != weights.device() ||
            lookup_alt_indices.device() != weights.device() || table_indices_flat.device() != weights.device() ||
            table_indices_alt_flat.device() != weights.device()) {
            throw py::value_error("all tensors must be on the same CUDA device");
        }

        int64_t batch_size = lookup_indices.size(0);
        int64_t n_tables = lookup_indices.size(1);
        int64_t n_alternatives = lookup_alt_indices.size(2);
        int64_t n_entries = weights.size(1);
        int64_t n_outputs = weights.size(2);
        int64_t total_bt = batch_size * n_tables;
        int64_t total_bta = total_bt * n_alternatives;
        if (table_indices_flat.numel() != total_bt || table_indices_alt_flat.numel() != total_bta) {
            throw py::value_error("table_indices_flat must be B*T and table_indices_alt_flat must be B*T*A");
        }
        if (lookup_alt_indices.numel() != total_bta) {
            throw py::value_error("lookup_alt_indices numel mismatch");
        }
        if (grad_output.size(0) != batch_size || grad_output.size(1) != n_tables || grad_output.size(2) != n_outputs) {
            throw py::value_error("grad_output shape mismatch");
        }

        auto opts = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());
        torch::Tensor weights_grad = torch::zeros_like(weights);
        torch::Tensor lookup_indices_grad_c_grad = torch::empty({batch_size, n_tables}, opts);
        torch::Tensor lookup_alt_indices_grad_c_grad = torch::empty({batch_size, n_tables, n_alternatives}, opts);

        int device = weights.device().index();
        c10::cuda::CUDAGuard guard(device);
        int threads = static_cast<int>(threads_per_block);
        int blocks_w = static_cast<int>(((total_bt * n_outputs) + threads - 1) / threads);
        int blocks_main = static_cast<int>((total_bt + threads - 1) / threads);
        int blocks_alt = static_cast<int>((total_bta + threads - 1) / threads);
        int64_t go_s0 = grad_output.stride(0);
        int64_t go_s1 = grad_output.stride(1);
        int64_t go_s2 = grad_output.stride(2);

        AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "lprojection_backward_nonsmooth", [&] {
            lprojection_backward_na1_nonsmooth_weights_kernel<scalar_t><<<blocks_w, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
            lprojection_backward_main_carriers_kernel<scalar_t><<<blocks_main, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_indices.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_indices_grad_c_grad.data_ptr())
            );
            lprojection_backward_alt_carriers_kernel<scalar_t><<<blocks_alt, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total_bta,
                n_tables,
                n_alternatives,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_alt_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_alt_indices.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_alt_indices_grad_c_grad.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        PROF_END(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);
        return py::make_tuple(weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad);
    }

    py::tuple
    lprojection_backward_smooth(
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
        PROF_START(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);
        if (!main_weight.is_cuda() || !alt_weight.is_cuda() ||
            main_weight.dtype() != weights.dtype() || alt_weight.dtype() != weights.dtype()) {
            throw py::value_error("main_weight and alt_weight must be CUDA and same dtype as weights");
        }
        if (main_weight.dim() != 2 || alt_weight.dim() != 3) {
            throw py::value_error("main_weight must be [B,T], alt_weight must be [B,T,A]");
        }
        if (main_weight.device() != weights.device() || alt_weight.device() != weights.device()) {
            throw py::value_error("main_weight/alt_weight device mismatch");
        }

        if (!grad_output.is_cuda() || !weights.is_cuda() || !lookup_indices.is_cuda() ||
            !lookup_alt_indices.is_cuda() || !table_indices_flat.is_cuda() || !table_indices_alt_flat.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (grad_output.dtype() != weights.dtype() || !grad_output.is_floating_point()) {
            throw py::value_error("grad_output/weights must be same floating dtype");
        }
        if (lookup_indices.dtype() != torch::kInt64 || lookup_alt_indices.dtype() != torch::kInt64 ||
            table_indices_flat.dtype() != torch::kInt64 || table_indices_alt_flat.dtype() != torch::kInt64) {
            throw py::value_error("indices tensors must be int64");
        }
        if (lookup_indices.dim() != 2 || lookup_alt_indices.dim() != 3) {
            throw py::value_error("lookup_indices must be [B,T], lookup_alt_indices must be [B,T,A]");
        }
        if (weights.dim() != 3 || grad_output.dim() != 3) {
            throw py::value_error("weights must be [T,E,O], grad_output must be [B,T,O]");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in range [1, 1024]");
        }
        if (grad_output.device() != weights.device() || lookup_indices.device() != weights.device() ||
            lookup_alt_indices.device() != weights.device() || table_indices_flat.device() != weights.device() ||
            table_indices_alt_flat.device() != weights.device()) {
            throw py::value_error("all tensors must be on the same CUDA device");
        }

        int64_t batch_size = lookup_indices.size(0);
        int64_t n_tables = lookup_indices.size(1);
        int64_t n_alternatives = lookup_alt_indices.size(2);
        int64_t n_entries = weights.size(1);
        int64_t n_outputs = weights.size(2);
        int64_t total_bt = batch_size * n_tables;
        int64_t total_bta = total_bt * n_alternatives;
        if (table_indices_flat.numel() != total_bt || table_indices_alt_flat.numel() != total_bta) {
            throw py::value_error("table_indices_flat must be B*T and table_indices_alt_flat must be B*T*A");
        }
        if (lookup_alt_indices.numel() != total_bta) {
            throw py::value_error("lookup_alt_indices numel mismatch");
        }
        if (grad_output.size(0) != batch_size || grad_output.size(1) != n_tables || grad_output.size(2) != n_outputs) {
            throw py::value_error("grad_output shape mismatch");
        }

        torch::Tensor weights_grad = torch::zeros_like(weights);
        auto opts = torch::TensorOptions().dtype(weights.dtype()).device(weights.device());
        torch::Tensor lookup_indices_grad_c_grad = torch::empty({batch_size, n_tables}, opts);
        torch::Tensor lookup_alt_indices_grad_c_grad = torch::empty({batch_size, n_tables, n_alternatives}, opts);
        int threads = static_cast<int>(threads_per_block);
        int blocks_w = static_cast<int>(((total_bt * n_outputs) + threads - 1) / threads);
        int blocks_main = static_cast<int>((total_bt + threads - 1) / threads);
        int blocks_alt = static_cast<int>((total_bta + threads - 1) / threads);
        int64_t go_s0 = grad_output.stride(0);
        int64_t go_s1 = grad_output.stride(1);
        int64_t go_s2 = grad_output.stride(2);

        int device = weights.device().index();
        c10::cuda::CUDAGuard guard(device);
        AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "lprojection_backward_smooth", [&] {
            lprojection_backward_smooth_weights_kernel<scalar_t><<<blocks_w, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                n_alternatives,
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
            lprojection_backward_main_carriers_kernel<scalar_t><<<blocks_main, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total_bt,
                n_tables,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_indices.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_indices_grad_c_grad.data_ptr())
            );
            lprojection_backward_alt_carriers_kernel<scalar_t><<<blocks_alt, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                total_bta,
                n_tables,
                n_alternatives,
                n_outputs,
                n_entries,
                reinterpret_cast<const scalar_t*>(grad_output.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(table_indices_alt_flat.data_ptr()),
                reinterpret_cast<const int64_t*>(lookup_alt_indices.data_ptr()),
                go_s0, go_s1, go_s2,
                reinterpret_cast<scalar_t*>(lookup_alt_indices_grad_c_grad.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        PROF_END(LUTORCH_MANAGER_LPROJECTION_BACKWARD_PROFILER_OP);
        return py::make_tuple(weights_grad, lookup_indices_grad_c_grad, lookup_alt_indices_grad_c_grad);
    }

    // -----------------------------------------------------------------------
    // Fused LUT Attention forward (n_alternatives=1, non-smooth)
    // -----------------------------------------------------------------------
    // Returns (pair_out_buf [B*M, H, O], result [B, T, H, O])
    py::tuple
    lut_attn_fwd_na1(
        const torch::Tensor& x,           // [B, T, E]  float32, CUDA
        const torch::Tensor& weights,     // [n_tables, n_entries, O]
        const torch::Tensor& anchor_a,    // [n_tables, n_anchor_pairs]
        const torch::Tensor& anchor_b,    // [n_tables, n_anchor_pairs]
        const torch::Tensor& pair_rows,   // [M]
        const torch::Tensor& pair_cols,   // [M]
        const c10::optional<torch::Tensor>& rel_pe_opt,  // [T, pos_dim] or nullopt
        int64_t H,
        int64_t tables_per_head,
        bool causal,
        bool self_excitement,
        double cmp_eps,
        int se_mode
    ) {
        if (!x.is_cuda()) throw py::value_error("x must be CUDA");
        if (!x.is_contiguous()) throw py::value_error("x must be contiguous");
        if (!weights.is_contiguous()) throw py::value_error("weights must be contiguous");
        if (!anchor_a.is_contiguous() || !anchor_b.is_contiguous()) throw py::value_error("anchor_a/b must be contiguous");
        if (!pair_rows.is_contiguous() || !pair_cols.is_contiguous()) throw py::value_error("pair_rows/cols must be contiguous");

        const int64_t B = x.size(0);
        const int64_t T = x.size(1);
        const int64_t E = x.size(2);
        const int64_t M = pair_rows.size(0);
        const int64_t n_tables = anchor_a.size(0);
        const int64_t n_anchor_pairs = anchor_a.size(1);
        const int64_t n_entries = weights.size(1);
        const int64_t O = weights.size(2);
        const int64_t pos_dim = rel_pe_opt.has_value() ? rel_pe_opt.value().size(1) : 0;

        auto opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor pair_out_buf = torch::zeros({B * M, H, O}, opts);
        torch::Tensor result = torch::zeros({B, T, H, O}, opts);

        torch::Tensor rel_pe;
        if (rel_pe_opt.has_value()) {
            rel_pe = rel_pe_opt.value().contiguous();
        }

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);

        int64_t input_dim = 2 * E + pos_dim;
        int64_t tile_tph = std::min(static_cast<int64_t>(1024) / O, tables_per_head);
        // Round down to power of 2
        {
            int64_t v = tile_tph;
            v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
            tile_tph = (v + 1) >> 1;
        }
        if (tile_tph < 1) tile_tph = 1;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lut_attn_fwd_na1", [&] {
            const scalar_t* rpe_ptr = rel_pe_opt.has_value()
                ? reinterpret_cast<const scalar_t*>(rel_pe.data_ptr()) : nullptr;
            dim3 grid(static_cast<unsigned>(B * M), static_cast<unsigned>(H));
            dim3 block(static_cast<unsigned>(O), static_cast<unsigned>(tile_tph));
            size_t smem = static_cast<size_t>(input_dim + tile_tph * O) * sizeof(scalar_t);
            lut_attn_fwd_na1_kernel<scalar_t><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t*>(x.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor_a.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor_b.data_ptr()),
                reinterpret_cast<const int64_t*>(pair_rows.data_ptr()),
                reinterpret_cast<const int64_t*>(pair_cols.data_ptr()),
                rpe_ptr,
                reinterpret_cast<scalar_t*>(pair_out_buf.data_ptr()),
                reinterpret_cast<scalar_t*>(result.data_ptr()),
                B, T, E, M, H,
                tables_per_head,
                n_entries, n_anchor_pairs, O, pos_dim,
                causal, self_excitement,
                static_cast<scalar_t>(cmp_eps),
                se_mode
            );
        });
        CU_CHECK(cudaGetLastError());
        return py::make_tuple(pair_out_buf, result);
    }

    // -----------------------------------------------------------------------
    // Fused LUT Attention backward (n_alternatives=1, non-smooth)
    // -----------------------------------------------------------------------
    // Returns (x_grad [B, T, E], weights_grad [n_tables, n_entries, O], rel_pe_grad or None)
    py::tuple
    lut_attn_bwd_na1(
        const torch::Tensor& x,
        const torch::Tensor& weights,
        const torch::Tensor& anchor_a,
        const torch::Tensor& anchor_b,
        const torch::Tensor& pair_rows,
        const torch::Tensor& pair_cols,
        const c10::optional<torch::Tensor>& rel_pe_opt,
        const torch::Tensor& pair_out_buf,   // [B*M, H, O]
        const torch::Tensor& result_grad,    // [B, T, H, O]
        int64_t H,
        int64_t tables_per_head,
        bool causal,
        bool self_excitement,
        double cmp_eps,
        double uncertainty_bias,
        int se_mode
    ) {
        if (!x.is_cuda()) throw py::value_error("x must be CUDA");

        const int64_t B = x.size(0);
        const int64_t T = x.size(1);
        const int64_t E = x.size(2);
        const int64_t M = pair_rows.size(0);
        const int64_t n_tables = anchor_a.size(0);
        const int64_t n_anchor_pairs = anchor_a.size(1);
        const int64_t n_entries = weights.size(1);
        const int64_t O = weights.size(2);
        const int64_t pos_dim = rel_pe_opt.has_value() ? rel_pe_opt.value().size(1) : 0;

        auto opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        torch::Tensor x_grad = torch::zeros_like(x);
        torch::Tensor weights_grad = torch::zeros_like(weights);

        torch::Tensor rel_pe;
        torch::Tensor rel_pe_grad;
        if (rel_pe_opt.has_value()) {
            rel_pe = rel_pe_opt.value().contiguous();
            rel_pe_grad = torch::zeros_like(rel_pe);
        }

        int device = x.device().index();
        c10::cuda::CUDAGuard guard(device);

        int64_t input_dim = 2 * E + pos_dim;
        int64_t tile_tph = std::min(static_cast<int64_t>(1024) / O, tables_per_head);
        {
            int64_t v = tile_tph;
            v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
            tile_tph = (v + 1) >> 1;
        }
        if (tile_tph < 1) tile_tph = 1;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lut_attn_bwd_na1", [&] {
            const scalar_t* rpe_ptr = rel_pe_opt.has_value()
                ? reinterpret_cast<const scalar_t*>(rel_pe.data_ptr()) : nullptr;
            scalar_t* rpe_grad_ptr = rel_pe_opt.has_value()
                ? reinterpret_cast<scalar_t*>(rel_pe_grad.data_ptr()) : nullptr;
            dim3 grid(static_cast<unsigned>(B * M), static_cast<unsigned>(H));
            dim3 block(static_cast<unsigned>(O), static_cast<unsigned>(tile_tph));
            size_t smem = static_cast<size_t>(input_dim + tile_tph * O + input_dim) * sizeof(scalar_t);
            lut_attn_bwd_na1_kernel<scalar_t><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t*>(x.data_ptr()),
                reinterpret_cast<const scalar_t*>(weights.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor_a.data_ptr()),
                reinterpret_cast<const int64_t*>(anchor_b.data_ptr()),
                reinterpret_cast<const int64_t*>(pair_rows.data_ptr()),
                reinterpret_cast<const int64_t*>(pair_cols.data_ptr()),
                rpe_ptr,
                reinterpret_cast<const scalar_t*>(pair_out_buf.data_ptr()),
                reinterpret_cast<const scalar_t*>(result_grad.data_ptr()),
                reinterpret_cast<scalar_t*>(weights_grad.data_ptr()),
                reinterpret_cast<scalar_t*>(x_grad.data_ptr()),
                rpe_grad_ptr,
                B, T, E, M, H,
                tables_per_head,
                n_entries, n_anchor_pairs, O, pos_dim,
                causal, self_excitement,
                static_cast<scalar_t>(cmp_eps),
                static_cast<scalar_t>(uncertainty_bias),
                se_mode
            );
        });
        CU_CHECK(cudaGetLastError());
        if (rel_pe_opt.has_value()) {
            return py::make_tuple(x_grad, weights_grad, rel_pe_grad);
        } else {
            return py::make_tuple(x_grad, weights_grad, py::none());
        }
    }
#endif

    std::string get_profiling_stats() {
        #ifdef ENABLE_PROFILING
        return profiler.get_stats_as_string();
        #else
        return "profiler is disabled";
        #endif
    }

    void reset_profiling_stats() {
        #ifdef ENABLE_PROFILING
        profiler.reset();
        #endif
    }

    #ifndef NO_CUDA
    // -----------------------------------------------------------------
    // PermutationalLut fused forward
    // raw:    [B, H*T, P]  contiguous (per-table outputs)
    // idx_a:  [H, T*P]     int64 (output endpoint indices)
    // idx_b:  [H, T*P]     int64
    // returns out: [B, H, N]
    // -----------------------------------------------------------------
    torch::Tensor perm_lut_forward(
        const torch::Tensor& raw,
        const torch::Tensor& idx_a,
        const torch::Tensor& idx_b,
        int64_t n_outputs,
        int64_t soft_mode,
        double temperature,
        int64_t threads_per_block = 256
    ) {
        if (!raw.is_cuda() || !idx_a.is_cuda() || !idx_b.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!raw.is_floating_point()) {
            throw py::value_error("raw must be floating point");
        }
        if (idx_a.dtype() != torch::kInt64 || idx_b.dtype() != torch::kInt64) {
            throw py::value_error("idx_a/idx_b must be int64");
        }
        if (raw.dim() != 3) {
            throw py::value_error("raw must be [B, H*T, P]");
        }
        if (idx_a.dim() != 2 || idx_b.dim() != 2) {
            throw py::value_error("idx_a/idx_b must be 2D [H, T*P]");
        }
        if (soft_mode < 0 || soft_mode > 2) {
            throw py::value_error("soft_mode must be 0=sigmoid, 1=rational, 2=ste");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in [1, 1024]");
        }

        int64_t B = raw.size(0);
        int64_t HT = raw.size(1);
        int64_t P = raw.size(2);
        int64_t H = idx_a.size(0);
        int64_t TP = idx_a.size(1);
        if (HT % H != 0) {
            throw py::value_error("raw.size(1) must be divisible by H = idx_a.size(0)");
        }
        int64_t T = HT / H;
        if (TP != T * P) {
            throw py::value_error("idx_a.size(1) must equal (raw.size(1)/H) * raw.size(2)");
        }
        if (idx_b.size(0) != H || idx_b.size(1) != TP) {
            throw py::value_error("idx_a and idx_b shapes must match");
        }

        auto raw_c = raw.contiguous();
        auto idx_a_c = idx_a.contiguous();
        auto idx_b_c = idx_b.contiguous();

        auto opts = torch::TensorOptions().dtype(raw.dtype()).device(raw.device());
        torch::Tensor out = torch::zeros({B, H, n_outputs}, opts);

        c10::cuda::CUDAGuard guard(raw.device().index());
        int64_t total = B * H * T * P;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(raw.scalar_type(), "perm_lut_forward", [&] {
            scalar_t T_val = static_cast<scalar_t>(temperature);
            scalar_t inv_T = static_cast<scalar_t>(1.0 / temperature);
            perm_lut_fwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                B, H, T, P, n_outputs,
                static_cast<int>(soft_mode),
                T_val, inv_T,
                reinterpret_cast<const scalar_t*>(raw_c.data_ptr()),
                reinterpret_cast<const int64_t*>(idx_a_c.data_ptr()),
                reinterpret_cast<const int64_t*>(idx_b_c.data_ptr()),
                reinterpret_cast<scalar_t*>(out.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        return out;
    }

    // -----------------------------------------------------------------
    // PermutationalLut fused backward
    // grad_out: [B, H, N]
    // raw:      [B, H*T, P]      (saved from forward)
    // idx_a/b:  [H, T*P]
    // returns grad_raw: [B, H*T, P]
    // -----------------------------------------------------------------
    torch::Tensor perm_lut_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& raw,
        const torch::Tensor& idx_a,
        const torch::Tensor& idx_b,
        int64_t soft_mode,
        double temperature,
        int64_t threads_per_block = 256
    ) {
        if (!grad_out.is_cuda() || !raw.is_cuda() || !idx_a.is_cuda() || !idx_b.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!grad_out.is_floating_point() || !raw.is_floating_point()) {
            throw py::value_error("grad_out and raw must be floating point");
        }
        if (grad_out.dtype() != raw.dtype()) {
            throw py::value_error("grad_out and raw must have same dtype");
        }
        if (idx_a.dtype() != torch::kInt64 || idx_b.dtype() != torch::kInt64) {
            throw py::value_error("idx_a/idx_b must be int64");
        }
        if (grad_out.dim() != 3 || raw.dim() != 3) {
            throw py::value_error("grad_out must be [B,H,N], raw must be [B,H*T,P]");
        }

        int64_t B = raw.size(0);
        int64_t HT = raw.size(1);
        int64_t P = raw.size(2);
        int64_t H = idx_a.size(0);
        int64_t TP = idx_a.size(1);
        if (HT % H != 0) {
            throw py::value_error("raw.size(1) must be divisible by H");
        }
        int64_t T = HT / H;
        if (TP != T * P) {
            throw py::value_error("idx_a shape must equal T*P");
        }
        if (grad_out.size(0) != B || grad_out.size(1) != H) {
            throw py::value_error("grad_out shape mismatch with raw/idx");
        }
        int64_t N = grad_out.size(2);

        auto grad_out_c = grad_out.contiguous();
        auto raw_c = raw.contiguous();
        auto idx_a_c = idx_a.contiguous();
        auto idx_b_c = idx_b.contiguous();

        torch::Tensor grad_raw = torch::empty_like(raw_c);

        c10::cuda::CUDAGuard guard(raw.device().index());
        int64_t total = B * H * T * P;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(raw.scalar_type(), "perm_lut_backward", [&] {
            scalar_t T_val = static_cast<scalar_t>(temperature);
            scalar_t inv_T = static_cast<scalar_t>(1.0 / temperature);
            perm_lut_bwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                B, H, T, P, N,
                static_cast<int>(soft_mode),
                T_val, inv_T,
                reinterpret_cast<const scalar_t*>(grad_out_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(raw_c.data_ptr()),
                reinterpret_cast<const int64_t*>(idx_a_c.data_ptr()),
                reinterpret_cast<const int64_t*>(idx_b_c.data_ptr()),
                reinterpret_cast<scalar_t*>(grad_raw.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        return grad_raw;
    }

    // -----------------------------------------------------------------
    // Dominance-path forward: halves atomics vs remap trick.
    // raw:      [B, H*T, P_slots] float  (P_slots = output_nap)
    // pair_idx: [H, T*P_slots] int64   (canonical pair index in [0, P_out))
    // sign:     [H, T*P_slots] float   (±1; same dtype as raw)
    // returns:  [B, H, P_out]
    // -----------------------------------------------------------------
    torch::Tensor perm_lut_dom_forward(
        const torch::Tensor& raw,
        const torch::Tensor& pair_idx,
        const torch::Tensor& sign,
        int64_t n_pairs,
        int64_t soft_mode,
        double temperature,
        int64_t threads_per_block = 256
    ) {
        if (!raw.is_cuda() || !pair_idx.is_cuda() || !sign.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!raw.is_floating_point()) {
            throw py::value_error("raw must be floating point");
        }
        if (pair_idx.dtype() != torch::kInt64) {
            throw py::value_error("pair_idx must be int64");
        }
        if (sign.dtype() != raw.dtype()) {
            throw py::value_error("sign must match raw dtype");
        }
        if (raw.dim() != 3) {
            throw py::value_error("raw must be [B, H*T, P_slots]");
        }
        if (pair_idx.dim() != 2 || sign.dim() != 2) {
            throw py::value_error("pair_idx/sign must be 2D [H, T*P_slots]");
        }
        if (soft_mode < 0 || soft_mode > 2) {
            throw py::value_error("soft_mode must be 0=sigmoid, 1=rational, 2=ste");
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            throw py::value_error("threads_per_block must be in [1, 1024]");
        }

        int64_t B = raw.size(0);
        int64_t HT = raw.size(1);
        int64_t P_slots = raw.size(2);
        int64_t H = pair_idx.size(0);
        int64_t TP = pair_idx.size(1);
        if (HT % H != 0) {
            throw py::value_error("raw.size(1) must be divisible by H = pair_idx.size(0)");
        }
        int64_t T = HT / H;
        if (TP != T * P_slots) {
            throw py::value_error("pair_idx.size(1) must equal T*P_slots");
        }
        if (sign.size(0) != H || sign.size(1) != TP) {
            throw py::value_error("sign shape must match pair_idx");
        }

        auto raw_c = raw.contiguous();
        auto pair_idx_c = pair_idx.contiguous();
        auto sign_c = sign.contiguous();

        auto opts = torch::TensorOptions().dtype(raw.dtype()).device(raw.device());
        torch::Tensor out = torch::zeros({B, H, n_pairs}, opts);

        c10::cuda::CUDAGuard guard(raw.device().index());
        int64_t total = B * H * T * P_slots;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(raw.scalar_type(), "perm_lut_dom_forward", [&] {
            scalar_t T_val = static_cast<scalar_t>(temperature);
            scalar_t inv_T = static_cast<scalar_t>(1.0 / temperature);
            perm_lut_dom_fwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                B, H, T, P_slots, n_pairs,
                static_cast<int>(soft_mode),
                T_val, inv_T,
                reinterpret_cast<const scalar_t*>(raw_c.data_ptr()),
                reinterpret_cast<const int64_t*>(pair_idx_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(sign_c.data_ptr()),
                reinterpret_cast<scalar_t*>(out.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        return out;
    }

    // -----------------------------------------------------------------
    // Dominance-path forward (gather variant): no atomics, one thread per
    // output. Forward only — reuse perm_lut_dom_backward for backward.
    // raw:      [B, H*TP] float  (TP = tph * output_nap)
    // inv_idx:  [H, P, K] int64  (raw slot index or -1 padding)
    // inv_sign: [H, P, K] float  (±1 sign)
    // returns:  [B, H, P]
    // -----------------------------------------------------------------
    torch::Tensor perm_lut_dom_gather_forward(
        const torch::Tensor& raw,
        const torch::Tensor& inv_idx,
        const torch::Tensor& inv_sign,
        int64_t n_pairs,
        int64_t soft_mode,
        double temperature,
        int64_t threads_per_block = 256
    ) {
        if (!raw.is_cuda() || !inv_idx.is_cuda() || !inv_sign.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!raw.is_floating_point()) {
            throw py::value_error("raw must be floating point");
        }
        if (inv_idx.dtype() != torch::kInt64) {
            throw py::value_error("inv_idx must be int64");
        }
        if (inv_sign.dtype() != raw.dtype()) {
            throw py::value_error("inv_sign must match raw dtype");
        }
        if (inv_idx.dim() != 3 || inv_sign.dim() != 3) {
            throw py::value_error("inv_idx/inv_sign must be [H, P, K]");
        }

        int64_t B = raw.size(0);
        int64_t HTP = raw.size(1) * raw.size(2);  // flat over H*TP
        int64_t H = inv_idx.size(0);
        int64_t P = inv_idx.size(1);
        int64_t K = inv_idx.size(2);
        if (HTP % H != 0) {
            throw py::value_error("raw total must be divisible by H");
        }
        int64_t TP = HTP / H;
        if (P != n_pairs) {
            throw py::value_error("inv_idx.size(1) must equal n_pairs");
        }

        auto raw_c = raw.contiguous().view({B, HTP});
        auto inv_idx_c = inv_idx.contiguous();
        auto inv_sign_c = inv_sign.contiguous();

        auto opts = torch::TensorOptions().dtype(raw.dtype()).device(raw.device());
        torch::Tensor out = torch::empty({B, H, P}, opts);

        c10::cuda::CUDAGuard guard(raw.device().index());
        int64_t total = B * H * P;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(raw.scalar_type(), "perm_lut_dom_gather_forward", [&] {
            scalar_t T_val = static_cast<scalar_t>(temperature);
            scalar_t inv_T = static_cast<scalar_t>(1.0 / temperature);
            perm_lut_dom_gather_fwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                B, H, TP, P, K,
                static_cast<int>(soft_mode),
                T_val, inv_T,
                reinterpret_cast<const scalar_t*>(raw_c.data_ptr()),
                reinterpret_cast<const int64_t*>(inv_idx_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(inv_sign_c.data_ptr()),
                reinterpret_cast<scalar_t*>(out.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        return out;
    }

    // -----------------------------------------------------------------
    // Dominance-path backward
    // -----------------------------------------------------------------
    torch::Tensor perm_lut_dom_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& raw,
        const torch::Tensor& pair_idx,
        const torch::Tensor& sign,
        int64_t soft_mode,
        double temperature,
        int64_t threads_per_block = 256
    ) {
        if (!grad_out.is_cuda() || !raw.is_cuda() || !pair_idx.is_cuda() || !sign.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!grad_out.is_floating_point() || !raw.is_floating_point()) {
            throw py::value_error("grad_out and raw must be floating point");
        }
        if (grad_out.dtype() != raw.dtype() || sign.dtype() != raw.dtype()) {
            throw py::value_error("grad_out/raw/sign must share dtype");
        }
        if (pair_idx.dtype() != torch::kInt64) {
            throw py::value_error("pair_idx must be int64");
        }

        int64_t B = raw.size(0);
        int64_t HT = raw.size(1);
        int64_t P_slots = raw.size(2);
        int64_t H = pair_idx.size(0);
        int64_t TP = pair_idx.size(1);
        int64_t T = HT / H;
        int64_t P_out = grad_out.size(2);

        auto grad_out_c = grad_out.contiguous();
        auto raw_c = raw.contiguous();
        auto pair_idx_c = pair_idx.contiguous();
        auto sign_c = sign.contiguous();

        torch::Tensor grad_raw = torch::empty_like(raw_c);

        c10::cuda::CUDAGuard guard(raw.device().index());
        int64_t total = B * H * T * P_slots;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(raw.scalar_type(), "perm_lut_dom_backward", [&] {
            scalar_t T_val = static_cast<scalar_t>(temperature);
            scalar_t inv_T = static_cast<scalar_t>(1.0 / temperature);
            perm_lut_dom_bwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                B, H, T, P_slots, P_out,
                static_cast<int>(soft_mode),
                T_val, inv_T,
                reinterpret_cast<const scalar_t*>(grad_out_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(raw_c.data_ptr()),
                reinterpret_cast<const int64_t*>(pair_idx_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(sign_c.data_ptr()),
                reinterpret_cast<scalar_t*>(grad_raw.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        return grad_raw;
    }

    // -----------------------------------------------------------------
    // TinyAnchorPairsLookup forward
    // x:        [B, input_dim] float (contiguous)
    // a/b:      [n_tables, n_anchor_pairs] int16
    // returns: (lookup_idx [B,n_tables] int16,
    //           alt_idx    [B,n_tables] int16,
    //           alt_delta  [B,n_tables] float,
    //           anchor1_ids[B,n_tables] int16,
    //           anchor2_ids[B,n_tables] int16)
    // -----------------------------------------------------------------
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    tiny_apl_forward(
        const torch::Tensor& x,
        const torch::Tensor& anchor_pairs_a,
        const torch::Tensor& anchor_pairs_b,
        int64_t threads_per_block = 256
    ) {
        if (!x.is_cuda() || !anchor_pairs_a.is_cuda() || !anchor_pairs_b.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (!x.is_floating_point()) {
            throw py::value_error("x must be floating point");
        }
        if (anchor_pairs_a.dtype() != torch::kInt16 || anchor_pairs_b.dtype() != torch::kInt16) {
            throw py::value_error("anchor_pairs_a/b must be int16");
        }
        if (x.dim() != 2) {
            throw py::value_error("x must be [B, input_dim]");
        }
        if (anchor_pairs_a.dim() != 2 || anchor_pairs_b.dim() != 2) {
            throw py::value_error("anchor_pairs_a/b must be 2D");
        }

        int64_t B = x.size(0);
        int64_t input_dim = x.size(1);
        int64_t n_tables = anchor_pairs_a.size(0);
        int64_t n_anchor_pairs = anchor_pairs_a.size(1);
        if (anchor_pairs_b.size(0) != n_tables || anchor_pairs_b.size(1) != n_anchor_pairs) {
            throw py::value_error("anchor_pairs_a and anchor_pairs_b shapes must match");
        }
        if (input_dim > 32767) {
            throw py::value_error("input_dim must be <= 32767 (int16 limit)");
        }
        if (n_anchor_pairs > 16) {
            throw py::value_error("n_anchor_pairs must be <= 16");
        }

        auto x_c = x.contiguous();
        auto a_c = anchor_pairs_a.contiguous();
        auto b_c = anchor_pairs_b.contiguous();

        auto opts_float = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        auto opts_int16 = torch::TensorOptions().dtype(torch::kInt16).device(x.device());

        torch::Tensor lookup_idx = torch::empty({B, n_tables}, opts_int16);
        torch::Tensor alt_idx    = torch::empty({B, n_tables}, opts_int16);
        torch::Tensor alt_delta  = torch::empty({B, n_tables}, opts_float);
        torch::Tensor anchor1_ids = torch::empty({B, n_tables}, opts_int16);
        torch::Tensor anchor2_ids = torch::empty({B, n_tables}, opts_int16);

        c10::cuda::CUDAGuard guard(x.device().index());
        int64_t total = B * n_tables;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tiny_apl_forward", [&] {
            tiny_apl_fwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t*>(x_c.data_ptr()),
                static_cast<int32_t>(B),
                static_cast<int32_t>(input_dim),
                reinterpret_cast<const int16_t*>(a_c.data_ptr()),
                reinterpret_cast<const int16_t*>(b_c.data_ptr()),
                static_cast<int32_t>(n_tables),
                static_cast<int32_t>(n_anchor_pairs),
                reinterpret_cast<int16_t*>(lookup_idx.data_ptr()),
                reinterpret_cast<int16_t*>(alt_idx.data_ptr()),
                reinterpret_cast<scalar_t*>(alt_delta.data_ptr()),
                reinterpret_cast<int16_t*>(anchor1_ids.data_ptr()),
                reinterpret_cast<int16_t*>(anchor2_ids.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        return std::make_tuple(lookup_idx, alt_idx, alt_delta, anchor1_ids, anchor2_ids);
    }

    // -----------------------------------------------------------------
    // TinyAnchorPairsLookup backward
    // -----------------------------------------------------------------
    torch::Tensor tiny_apl_backward(
        int64_t batch_size,
        int64_t input_dim,
        const torch::Tensor& anchor1_ids,                // int16 [B, n_tables]
        const torch::Tensor& anchor2_ids,                // int16 [B, n_tables]
        const torch::Tensor& lookup_alt_deltas,          // float [B, n_tables]
        const torch::Tensor& grad_main,                  // float [B, n_tables]
        const torch::Tensor& grad_alt,                   // float [B, n_tables]
        c10::optional<torch::Tensor> grad_direct,        // float [B, n_tables] or None
        int64_t threads_per_block = 256
    ) {
        if (!anchor1_ids.is_cuda() || !lookup_alt_deltas.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (anchor1_ids.dtype() != torch::kInt16 || anchor2_ids.dtype() != torch::kInt16) {
            throw py::value_error("anchor1/2_ids must be int16");
        }
        if (!lookup_alt_deltas.is_floating_point()) {
            throw py::value_error("lookup_alt_deltas must be floating point");
        }

        int64_t n_tables = lookup_alt_deltas.size(-1) / 1;  // B * n_tables = total
        if (anchor1_ids.dim() == 2) n_tables = anchor1_ids.size(1);

        auto a1_c = anchor1_ids.contiguous();
        auto a2_c = anchor2_ids.contiguous();
        auto d_c = lookup_alt_deltas.contiguous();
        auto gm_c = grad_main.contiguous();
        auto ga_c = grad_alt.contiguous();

        auto opts_float = torch::TensorOptions().dtype(lookup_alt_deltas.dtype()).device(lookup_alt_deltas.device());
        torch::Tensor x_grad_flat = torch::zeros({batch_size * input_dim}, opts_float);

        c10::cuda::CUDAGuard guard(lookup_alt_deltas.device().index());
        int64_t total = batch_size * n_tables;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        const bool has_direct = grad_direct.has_value();
        torch::Tensor gd_c;
        if (has_direct) gd_c = grad_direct.value().contiguous();

        AT_DISPATCH_FLOATING_TYPES(lookup_alt_deltas.scalar_type(), "tiny_apl_backward", [&] {
            const scalar_t* gd_ptr = has_direct ? reinterpret_cast<const scalar_t*>(gd_c.data_ptr()) : nullptr;
            tiny_apl_bwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<int32_t>(total),
                static_cast<int32_t>(input_dim),
                static_cast<int32_t>(n_tables),
                reinterpret_cast<const int16_t*>(a1_c.data_ptr()),
                reinterpret_cast<const int16_t*>(a2_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(d_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(gm_c.data_ptr()),
                reinterpret_cast<const scalar_t*>(ga_c.data_ptr()),
                gd_ptr,
                reinterpret_cast<scalar_t*>(x_grad_flat.data_ptr())
            );
        });
        CU_CHECK(cudaGetLastError());
        return x_grad_flat;
    }

    // -----------------------------------------------------------------
    // BitPermutationLUT forward (int32 output, no float math in kernel).
    // Assumes CANONICAL_DISTINCT output pair sampling (no inv_sign needed).
    // lookup_indices: [B, n_heads*tph] int16 (from TinyAnchorPairsLookup)
    // bit_weights:    [n_heads*tph, table_dim, n_blocks] int32 (bitset)
    // inv_idx:        [n_heads, P, K] int32 (-1 padding; values in [0, tph*output_nap))
    // returns:        [B, n_heads, P] int32
    // -----------------------------------------------------------------
    torch::Tensor bit_perm_lut_dom_gather_forward(
        const torch::Tensor& lookup_indices,
        const torch::Tensor& bit_weights,
        const torch::Tensor& inv_idx,
        int64_t n_heads,
        int64_t tph,
        int64_t output_nap,
        int64_t n_pairs,
        int64_t threads_per_block = 256
    ) {
        if (!lookup_indices.is_cuda() || !bit_weights.is_cuda() || !inv_idx.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (lookup_indices.dtype() != torch::kInt16) throw py::value_error("lookup_indices must be int16");
        if (bit_weights.dtype()    != torch::kInt32) throw py::value_error("bit_weights must be int32");
        if (inv_idx.dtype()        != torch::kInt32) throw py::value_error("inv_idx must be int32");

        if (lookup_indices.dim() != 2) throw py::value_error("lookup_indices must be [B, n_heads*tph]");
        if (bit_weights.dim() != 3) throw py::value_error("bit_weights must be [n_heads*tph, table_dim, n_blocks]");
        if (inv_idx.dim() != 3) throw py::value_error("inv_idx must be [n_heads, P, K]");

        int64_t B = lookup_indices.size(0);
        if (lookup_indices.size(1) != n_heads * tph) throw py::value_error("lookup_indices.size(1) must equal n_heads*tph");
        if (bit_weights.size(0) != n_heads * tph) throw py::value_error("bit_weights.size(0) must equal n_heads*tph");

        int64_t table_dim = bit_weights.size(1);
        int64_t n_blocks = bit_weights.size(2);
        int64_t expected_n_blocks = (output_nap + 31) / 32;
        if (n_blocks != expected_n_blocks) throw py::value_error("bit_weights.size(2) must equal ceil(output_nap / 32)");
        if (inv_idx.size(0) != n_heads || inv_idx.size(1) != n_pairs) throw py::value_error("inv_idx shape mismatch");
        int64_t K = inv_idx.size(2);

        auto li_c = lookup_indices.contiguous();
        auto bw_c = bit_weights.contiguous();
        auto ii_c = inv_idx.contiguous();

        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(lookup_indices.device());
        torch::Tensor out = torch::empty({B, n_heads, n_pairs}, opts_i32);

        c10::cuda::CUDAGuard guard(lookup_indices.device().index());
        int64_t total = B * n_heads * n_pairs;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);

        bit_perm_lut_dom_gather_fwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int32_t>(B),
            static_cast<int32_t>(n_heads),
            static_cast<int32_t>(tph),
            static_cast<int32_t>(n_blocks),
            static_cast<int32_t>(n_pairs),
            static_cast<int32_t>(K),
            static_cast<int32_t>(table_dim),
            static_cast<int32_t>(output_nap),
            reinterpret_cast<const int16_t*>(li_c.data_ptr()),
            reinterpret_cast<const int32_t*>(bw_c.data_ptr()),
            reinterpret_cast<const int32_t*>(ii_c.data_ptr()),
            reinterpret_cast<int32_t*>(out.data_ptr())
        );
        CU_CHECK(cudaGetLastError());
        return out;
    }

    // -----------------------------------------------------------------
    // BitPermutationLUT backward:
    //   grad_out            [B, n_heads, P]    float32
    //   lookup_indices      [B, n_heads*tph]    int16
    //   lookup_alt_indices  [B, n_heads*tph, 1] int16
    //   bit_weights         [n_heads*tph, table_dim, n_blocks] int32
    //   pair_idx_per_slot   [n_heads, tph, output_nap] int32
    //   scale                scalar (0.5 / sqrt(n_votes_per_pair))
    // Returns (grad_main [B, n_heads*tph],
    //          grad_alt  [B, n_heads*tph, 1])  float32.
    // No weight gradient — bits are discrete.
    // -----------------------------------------------------------------
    std::tuple<torch::Tensor, torch::Tensor> bit_perm_lut_dom_gather_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& lookup_indices,
        const torch::Tensor& lookup_alt_indices,
        const torch::Tensor& bit_weights,
        const torch::Tensor& pair_idx_per_slot,
        int64_t n_heads,
        int64_t tph,
        int64_t output_nap,
        int64_t n_pairs,
        double scale,
        int64_t threads_per_block = 256
    ) {
        if (!grad_out.is_cuda() || !lookup_indices.is_cuda() || !lookup_alt_indices.is_cuda()
            || !bit_weights.is_cuda() || !pair_idx_per_slot.is_cuda()) {
            throw py::value_error("all tensors must be CUDA");
        }
        if (grad_out.dtype() != torch::kFloat32) throw py::value_error("grad_out must be float32");
        if (lookup_indices.dtype()     != torch::kInt16) throw py::value_error("lookup_indices must be int16");
        if (lookup_alt_indices.dtype() != torch::kInt16) throw py::value_error("lookup_alt_indices must be int16");
        if (bit_weights.dtype()        != torch::kInt32) throw py::value_error("bit_weights must be int32");
        if (pair_idx_per_slot.dtype()  != torch::kInt32) throw py::value_error("pair_idx_per_slot must be int32");

        int64_t B = grad_out.size(0);
        if (grad_out.dim() != 3 || grad_out.size(1) != n_heads || grad_out.size(2) != n_pairs)
            throw py::value_error("grad_out must be [B, n_heads, n_pairs]");
        if (lookup_indices.dim() != 2 || lookup_indices.size(0) != B || lookup_indices.size(1) != n_heads * tph)
            throw py::value_error("lookup_indices must be [B, n_heads*tph]");
        if (lookup_alt_indices.dim() != 3 || lookup_alt_indices.size(0) != B
            || lookup_alt_indices.size(1) != n_heads * tph || lookup_alt_indices.size(2) != 1)
            throw py::value_error("lookup_alt_indices must be [B, n_heads*tph, 1]");
        if (bit_weights.dim() != 3 || bit_weights.size(0) != n_heads * tph)
            throw py::value_error("bit_weights.size(0) must equal n_heads*tph");
        int64_t table_dim = bit_weights.size(1);
        int64_t n_blocks  = bit_weights.size(2);
        if (n_blocks != (output_nap + 31) / 32) throw py::value_error("bit_weights.size(2) must equal ceil(output_nap/32)");
        if (pair_idx_per_slot.dim() != 3
            || pair_idx_per_slot.size(0) != n_heads
            || pair_idx_per_slot.size(1) != tph
            || pair_idx_per_slot.size(2) != output_nap)
            throw py::value_error("pair_idx_per_slot must be [n_heads, tph, output_nap]");

        auto go_c  = grad_out.contiguous();
        auto li_c  = lookup_indices.contiguous();
        auto lai_c = lookup_alt_indices.contiguous();
        auto bw_c  = bit_weights.contiguous();
        auto pi_c  = pair_idx_per_slot.contiguous();

        auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(grad_out.device());
        torch::Tensor grad_main = torch::empty({B, n_heads * tph}, opts_f);
        torch::Tensor grad_alt  = torch::empty({B, n_heads * tph, 1}, opts_f);

        c10::cuda::CUDAGuard guard(grad_out.device().index());
        int64_t total = B * n_heads * tph;
        int threads = static_cast<int>(threads_per_block);
        int blocks  = static_cast<int>((total + threads - 1) / threads);

        bit_perm_lut_dom_gather_bwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int32_t>(B),
            static_cast<int32_t>(n_heads),
            static_cast<int32_t>(tph),
            static_cast<int32_t>(n_blocks),
            static_cast<int32_t>(n_pairs),
            static_cast<int32_t>(table_dim),
            static_cast<int32_t>(output_nap),
            static_cast<float>(scale),
            reinterpret_cast<const int16_t*>(li_c.data_ptr()),
            reinterpret_cast<const int16_t*>(lai_c.data_ptr()),
            reinterpret_cast<const int32_t*>(bw_c.data_ptr()),
            reinterpret_cast<const int32_t*>(pi_c.data_ptr()),
            reinterpret_cast<const float*>(go_c.data_ptr()),
            reinterpret_cast<float*>(grad_main.data_ptr()),
            reinterpret_cast<float*>(grad_alt.data_ptr())
        );
        CU_CHECK(cudaGetLastError());
        return std::make_tuple(grad_main, grad_alt);
    }

    // -----------------------------------------------------------------
    // Pack ±1 signs into int32 bit_weights blocks.
    //   signs:       [N, table_dim, output_nap] float32
    //   bit_weights: [N, table_dim, n_blocks]   int32 (output, pre-allocated
    //                                                  with n_blocks = ceil(output_nap/32))
    // -----------------------------------------------------------------
    void bit_pack_signs(
        const torch::Tensor& signs,
        torch::Tensor& bit_weights,
        int64_t output_nap,
        int64_t threads_per_block = 256
    ) {
        if (!signs.is_cuda() || !bit_weights.is_cuda()) throw py::value_error("tensors must be CUDA");
        if (signs.dtype() != torch::kFloat32) throw py::value_error("signs must be float32");
        if (bit_weights.dtype() != torch::kInt32) throw py::value_error("bit_weights must be int32");
        if (signs.dim() != 3) throw py::value_error("signs must be [N, table_dim, output_nap]");
        if (bit_weights.dim() != 3) throw py::value_error("bit_weights must be [N, table_dim, n_blocks]");
        int64_t N = signs.size(0);
        int64_t table_dim = signs.size(1);
        if (signs.size(2) != output_nap) throw py::value_error("signs.size(2) must equal output_nap");
        int64_t n_blocks = bit_weights.size(2);
        if (bit_weights.size(0) != N || bit_weights.size(1) != table_dim) throw py::value_error("bit_weights shape mismatch");
        if (n_blocks != (output_nap + 31) / 32) throw py::value_error("bit_weights.size(2) must equal ceil(output_nap/32)");

        auto sc = signs.contiguous();
        auto bc = bit_weights.contiguous();
        c10::cuda::CUDAGuard guard(signs.device().index());
        int64_t total = N * table_dim * n_blocks;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);
        bit_pack_signs_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int32_t>(N),
            static_cast<int32_t>(table_dim),
            static_cast<int32_t>(output_nap),
            static_cast<int32_t>(n_blocks),
            reinterpret_cast<const float*>(sc.data_ptr()),
            reinterpret_cast<int32_t*>(bc.data_ptr())
        );
        CU_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------
    // Fused fp8-latent Adam step (single kernel over all elements):
    //   Dequantize latent / m / v from fp8 (latent has fixed scale 1/448;
    //   m and v use per-table float32 scales). Update m, v with grad.
    //   Update latent, clamp to [-1, 1], requantize to fp8 (fixed scale).
    //   Emit new m, v as float32 scratch tensors (per-table requantization
    //   happens in a separate pass).
    // -----------------------------------------------------------------
    std::tuple<torch::Tensor, torch::Tensor> fused_fp8_adam(
        torch::Tensor& latent_fp8,              // [N, td, ona] fp8 (inplace)
        const torch::Tensor& m_fp8,             // [N, td, ona] fp8
        const torch::Tensor& m_scale,           // [N, 1, 1] float32
        const torch::Tensor& v_fp8,             // [N, td, ona] fp8
        const torch::Tensor& v_scale,           // [N, 1, 1] float32
        const torch::Tensor& weight_grad,       // [N, td, ona] float32
        double beta1, double beta2,
        double eps, double bias1, double bias2,
        double lr,
        int64_t threads_per_block = 256
    ) {
        if (!latent_fp8.is_cuda() || !m_fp8.is_cuda() || !v_fp8.is_cuda()
            || !m_scale.is_cuda() || !v_scale.is_cuda() || !weight_grad.is_cuda())
            throw py::value_error("tensors must be CUDA");
        if (latent_fp8.dtype() != torch::kFloat8_e4m3fn) throw py::value_error("latent_fp8 must be float8_e4m3fn");
        if (m_fp8.dtype()      != torch::kFloat8_e4m3fn) throw py::value_error("m_fp8 must be float8_e4m3fn");
        if (v_fp8.dtype()      != torch::kFloat8_e4m3fn) throw py::value_error("v_fp8 must be float8_e4m3fn");
        if (m_scale.dtype()    != torch::kFloat32) throw py::value_error("m_scale must be float32");
        if (v_scale.dtype()    != torch::kFloat32) throw py::value_error("v_scale must be float32");
        if (weight_grad.dtype()!= torch::kFloat32) throw py::value_error("weight_grad must be float32");
        if (latent_fp8.sizes() != m_fp8.sizes() || latent_fp8.sizes() != v_fp8.sizes()
            || latent_fp8.sizes() != weight_grad.sizes())
            throw py::value_error("latent/m/v/weight_grad shape mismatch");
        if (latent_fp8.dim() != 3) throw py::value_error("tensors must be [N, table_dim, output_nap]");

        int64_t N = latent_fp8.size(0);
        int64_t per_table = latent_fp8.size(1) * latent_fp8.size(2);
        int64_t total = N * per_table;

        auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(latent_fp8.device());
        auto m_out = torch::empty_like(weight_grad);
        auto v_out = torch::empty_like(weight_grad);

        auto lc = latent_fp8.contiguous();
        auto mc = m_fp8.contiguous();
        auto vc = v_fp8.contiguous();
        auto msc = m_scale.contiguous();
        auto vsc = v_scale.contiguous();
        auto gc = weight_grad.contiguous();

        // 1/bias1 could overflow if bias1 -> 0 (very early step with beta1^1 ~ 0.1,
        // so bias1 = 0.9 — not actually tiny). Caller supplies bias1/bias2 already
        // computed from the step counter; we just fuse the coefficients here.
        float bias2_sqrt = std::sqrt(std::max(bias2, 1e-30));
        float eps_times_b2sqrt = static_cast<float>(eps * bias2_sqrt);
        float lr_step_coef = static_cast<float>(-lr * bias2_sqrt / bias1);

        c10::cuda::CUDAGuard guard(latent_fp8.device().index());
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fused_fp8_adam_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int32_t>(total),
            static_cast<int32_t>(per_table),
            static_cast<float>(beta1),
            static_cast<float>(beta2),
            static_cast<float>(1.0 - beta1),
            static_cast<float>(1.0 - beta2),
            eps_times_b2sqrt,
            lr_step_coef,
            reinterpret_cast<__nv_fp8_e4m3*>(lc.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(mc.data_ptr()),
            reinterpret_cast<const float*>(msc.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(vc.data_ptr()),
            reinterpret_cast<const float*>(vsc.data_ptr()),
            reinterpret_cast<const float*>(gc.data_ptr()),
            reinterpret_cast<float*>(m_out.data_ptr()),
            reinterpret_cast<float*>(v_out.data_ptr())
        );
        CU_CHECK(cudaGetLastError());
        return std::make_tuple(m_out, v_out);
    }

    // -----------------------------------------------------------------
    // Project grad_out [B, n_heads, P] through (lookup_indices, pair_idx)
    // into weight_grad [n_heads*tph, table_dim, output_nap]. Caller owns
    // weight_grad (typically pre-zeroed); kernel atomicAdds into touched
    // (n, entry_main(b, n), k) positions only.
    // -----------------------------------------------------------------
    void bit_perm_lut_weight_grad(
        const torch::Tensor& grad_out,         // [B, H, P] float32
        const torch::Tensor& lookup_indices,   // int16 [B, H*tph]
        const torch::Tensor& pair_idx,          // int32 [H, tph, output_nap]
        torch::Tensor& weight_grad,             // float32 [N, table_dim, output_nap] (pre-zeroed)
        int64_t n_heads,
        int64_t tph,
        int64_t output_nap,
        int64_t table_dim,
        int64_t n_pairs,
        double scale,
        int64_t threads_per_block = 256
    ) {
        if (!grad_out.is_cuda() || !lookup_indices.is_cuda() || !pair_idx.is_cuda() || !weight_grad.is_cuda())
            throw py::value_error("tensors must be CUDA");
        if (grad_out.dtype() != torch::kFloat32) throw py::value_error("grad_out must be float32");
        if (lookup_indices.dtype() != torch::kInt16) throw py::value_error("lookup_indices must be int16");
        if (pair_idx.dtype() != torch::kInt32) throw py::value_error("pair_idx must be int32");
        if (weight_grad.dtype() != torch::kFloat32) throw py::value_error("weight_grad must be float32");
        int64_t B = grad_out.size(0);
        int64_t N = n_heads * tph;
        if (grad_out.size(1) != n_heads || grad_out.size(2) != n_pairs) throw py::value_error("grad_out shape mismatch");
        if (lookup_indices.size(0) != B || lookup_indices.size(1) != N) throw py::value_error("lookup_indices shape mismatch");
        if (pair_idx.size(0) != n_heads || pair_idx.size(1) != tph || pair_idx.size(2) != output_nap)
            throw py::value_error("pair_idx shape mismatch");
        if (weight_grad.size(0) != N || weight_grad.size(1) != table_dim || weight_grad.size(2) != output_nap)
            throw py::value_error("weight_grad shape mismatch");

        auto go = grad_out.contiguous();
        auto li = lookup_indices.contiguous();
        auto pi = pair_idx.contiguous();
        auto wg = weight_grad.contiguous();
        c10::cuda::CUDAGuard guard(grad_out.device().index());
        int64_t total = B * N;
        int threads = static_cast<int>(threads_per_block);
        int blocks = static_cast<int>((total + threads - 1) / threads);
        bit_perm_lut_weight_grad_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int32_t>(B),
            static_cast<int32_t>(n_heads),
            static_cast<int32_t>(tph),
            static_cast<int32_t>(table_dim),
            static_cast<int32_t>(output_nap),
            static_cast<int32_t>(n_pairs),
            static_cast<float>(scale),
            reinterpret_cast<const int16_t*>(li.data_ptr()),
            reinterpret_cast<const int32_t*>(pi.data_ptr()),
            reinterpret_cast<const float*>(go.data_ptr()),
            reinterpret_cast<float*>(wg.data_ptr())
        );
        CU_CHECK(cudaGetLastError());
    }
    #endif

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
            "anchor_pairs_lookup_forward_na2",
            &LUTorchManager::anchor_pairs_lookup_forward_na2,
            py::arg("x"),
            py::arg("anchor_pairs_a"),
            py::arg("anchor_pairs_b"),
            py::arg("cmp_eps"),
            py::arg("save_anchor_ids") = true,
            py::arg("threads_per_block") = 256
        )
        .def(
            "anchor_pairs_lookup_forward_na3",
            &LUTorchManager::anchor_pairs_lookup_forward_na3,
            py::arg("x"),
            py::arg("anchor_pairs_a"),
            py::arg("anchor_pairs_b"),
            py::arg("cmp_eps"),
            py::arg("save_anchor_ids") = true,
            py::arg("threads_per_block") = 256
        )
        .def(
            "anchor_pairs_lookup_forward_all",
            &LUTorchManager::anchor_pairs_lookup_forward_all,
            py::arg("x"),
            py::arg("anchor_pairs_a"),
            py::arg("anchor_pairs_b"),
            py::arg("cmp_eps"),
            py::arg("save_anchor_ids") = true,
            py::arg("threads_per_block") = 256
        )
        .def(
            "anchor_pairs_lookup_eval_forward",
            &LUTorchManager::anchor_pairs_lookup_eval_forward,
            py::arg("x"),
            py::arg("anchor_pairs_a"),
            py::arg("anchor_pairs_b"),
            py::arg("cmp_eps"),
            py::arg("threads_per_block") = 256
        )
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
            py::arg("uncertainty_bias") = 0.5,
            py::arg("threads_per_block") = 256
        )
        .def(
            "wta_lookup_forward_na1",
            &LUTorchManager::wta_lookup_forward_na1,
            py::arg("x"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "wta_lookup_forward_na2",
            &LUTorchManager::wta_lookup_forward_na2,
            py::arg("x"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "wta_lookup_forward_na3",
            &LUTorchManager::wta_lookup_forward_na3,
            py::arg("x"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "wta_lookup_backward",
            &LUTorchManager::wta_lookup_backward,
            py::arg("x"),
            py::arg("winner_ids"),
            py::arg("alt_ids"),
            py::arg("alt_deltas"),
            py::arg("batch_offset"),
            py::arg("grad_main"),
            py::arg("grad_alt"),
            py::arg("n_alternatives"),
            py::arg("inv_l1"),
            py::arg("uncertainty_bias") = 0.5,
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
            "lprojection_backward_nonsmooth",
            &LUTorchManager::lprojection_backward_nonsmooth,
            py::arg("grad_output"),
            py::arg("weights"),
            py::arg("lookup_indices"),
            py::arg("lookup_alt_indices"),
            py::arg("table_indices_flat"),
            py::arg("table_indices_alt_flat"),
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
            py::arg("uncertainty_bias") = 0.5,
            py::arg("threads_per_block") = 256
        )
        .def(
            "lprojection_backward_smooth",
            &LUTorchManager::lprojection_backward_smooth,
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
            "lut_attn_fwd_na1",
            &LUTorchManager::lut_attn_fwd_na1,
            py::arg("x"),
            py::arg("weights"),
            py::arg("anchor_a"),
            py::arg("anchor_b"),
            py::arg("pair_rows"),
            py::arg("pair_cols"),
            py::arg("rel_pe"),
            py::arg("H"),
            py::arg("tables_per_head"),
            py::arg("causal"),
            py::arg("self_excitement"),
            py::arg("cmp_eps"),
            py::arg("se_mode")
        )
        .def(
            "lut_attn_bwd_na1",
            &LUTorchManager::lut_attn_bwd_na1,
            py::arg("x"),
            py::arg("weights"),
            py::arg("anchor_a"),
            py::arg("anchor_b"),
            py::arg("pair_rows"),
            py::arg("pair_cols"),
            py::arg("rel_pe"),
            py::arg("pair_out_buf"),
            py::arg("result_grad"),
            py::arg("H"),
            py::arg("tables_per_head"),
            py::arg("causal"),
            py::arg("self_excitement"),
            py::arg("cmp_eps"),
            py::arg("uncertainty_bias"),
            py::arg("se_mode")
        )
        .def(
            "perm_lut_forward",
            &LUTorchManager::perm_lut_forward,
            py::arg("raw"),
            py::arg("idx_a"),
            py::arg("idx_b"),
            py::arg("n_outputs"),
            py::arg("soft_mode"),
            py::arg("temperature"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "perm_lut_backward",
            &LUTorchManager::perm_lut_backward,
            py::arg("grad_out"),
            py::arg("raw"),
            py::arg("idx_a"),
            py::arg("idx_b"),
            py::arg("soft_mode"),
            py::arg("temperature"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "perm_lut_dom_forward",
            &LUTorchManager::perm_lut_dom_forward,
            py::arg("raw"),
            py::arg("pair_idx"),
            py::arg("sign"),
            py::arg("n_pairs"),
            py::arg("soft_mode"),
            py::arg("temperature"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "perm_lut_dom_gather_forward",
            &LUTorchManager::perm_lut_dom_gather_forward,
            py::arg("raw"),
            py::arg("inv_idx"),
            py::arg("inv_sign"),
            py::arg("n_pairs"),
            py::arg("soft_mode"),
            py::arg("temperature"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "perm_lut_dom_backward",
            &LUTorchManager::perm_lut_dom_backward,
            py::arg("grad_out"),
            py::arg("raw"),
            py::arg("pair_idx"),
            py::arg("sign"),
            py::arg("soft_mode"),
            py::arg("temperature"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "tiny_apl_forward",
            &LUTorchManager::tiny_apl_forward,
            py::arg("x"),
            py::arg("anchor_pairs_a"),
            py::arg("anchor_pairs_b"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "tiny_apl_backward",
            &LUTorchManager::tiny_apl_backward,
            py::arg("batch_size"),
            py::arg("input_dim"),
            py::arg("anchor1_ids"),
            py::arg("anchor2_ids"),
            py::arg("lookup_alt_deltas"),
            py::arg("grad_main"),
            py::arg("grad_alt"),
            py::arg("grad_direct") = py::none(),
            py::arg("threads_per_block") = 256
        )
        .def(
            "bit_perm_lut_dom_gather_forward",
            &LUTorchManager::bit_perm_lut_dom_gather_forward,
            py::arg("lookup_indices"),
            py::arg("bit_weights"),
            py::arg("inv_idx"),
            py::arg("n_heads"),
            py::arg("tph"),
            py::arg("output_nap"),
            py::arg("n_pairs"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "bit_perm_lut_dom_gather_backward",
            &LUTorchManager::bit_perm_lut_dom_gather_backward,
            py::arg("grad_out"),
            py::arg("lookup_indices"),
            py::arg("lookup_alt_indices"),
            py::arg("bit_weights"),
            py::arg("pair_idx_per_slot"),
            py::arg("n_heads"),
            py::arg("tph"),
            py::arg("output_nap"),
            py::arg("n_pairs"),
            py::arg("scale"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "bit_pack_signs",
            &LUTorchManager::bit_pack_signs,
            py::arg("signs"),
            py::arg("bit_weights"),
            py::arg("output_nap"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "fused_fp8_adam",
            &LUTorchManager::fused_fp8_adam,
            py::arg("latent_fp8"),
            py::arg("m_fp8"),
            py::arg("m_scale"),
            py::arg("v_fp8"),
            py::arg("v_scale"),
            py::arg("weight_grad"),
            py::arg("beta1"),
            py::arg("beta2"),
            py::arg("eps"),
            py::arg("bias1"),
            py::arg("bias2"),
            py::arg("lr"),
            py::arg("threads_per_block") = 256
        )
        .def(
            "bit_perm_lut_weight_grad",
            &LUTorchManager::bit_perm_lut_weight_grad,
            py::arg("grad_out"),
            py::arg("lookup_indices"),
            py::arg("pair_idx"),
            py::arg("weight_grad"),
            py::arg("n_heads"),
            py::arg("tph"),
            py::arg("output_nap"),
            py::arg("table_dim"),
            py::arg("n_pairs"),
            py::arg("scale"),
            py::arg("threads_per_block") = 256
        )
        #endif
        .def("get_profiling_stats", &LUTorchManager::get_profiling_stats)
        .def("reset_profiling_stats", &LUTorchManager::reset_profiling_stats);

    // Singleton: one manager for all lutorch ops so profiler sees both lookup and lprojection.
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
