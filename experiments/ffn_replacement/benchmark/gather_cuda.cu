// Hand-written CUDA gather+sum for the FastMultiHeadLut hard-eval path.
//
// Same contract as the Triton kernel in gather.py:
//     out[n, h, :] = sum over the tph tables of head h of W[table, index[n, table], :]
// output-stationary, table offset folded into the addressing, per-head sum
// accumulated in registers across the table loop.
//
// Two things this has that the Triton kernel does not:
//
// 1. SOFTWARE PIPELINING. The row load's address comes from an in-loop indirect load
//    of the index, so the two loads have different dependency depths: the index must
//    run further ahead than the row. Triton's `num_stages` drives one pipeline depth
//    for the whole loop body and cannot express that -- sweeping it 1..5 moved these
//    shapes by 0.0-0.3%. Here the index runs 2 ahead and the row 1 ahead, which is
//    worth 1.19-1.30x over Triton on the 5090.
//    A deeper pipeline (index 3 / row 2) was tried and is REJECTED: 128 registers at
//    512 threads is the whole 64K register file, i.e. one block per SM, and it runs
//    0.72-0.90x. Latency hiding bought with occupancy is a bad trade here.
//
// 2. A bf16 TABLE (gather_sum_bf16), which the Triton load path could not exploit.
//    Rows are read as 16-byte vector loads and converted on arrival; accumulation
//    stays fp32. This is an APPROXIMATE path -- see the numerics note below.
//
// MECHANISM, since it is easy to state wrongly: the win is in 32-byte SECTORS, not
// 128-byte cache lines. Every row pitch here is 32 B aligned, so an fp32 48-wide row
// is exactly 6 sectors and a bf16 row exactly 3 -- packed or padded. That predicts
// both measurements: padding an fp32 row from 192 B to 256 B changes nothing (1.00x,
// measured), and padding a bf16 row from 96 B to 128 B also changes nothing much
// (1.03-1.06x, measured) even though it takes the row from straddling 1.5 lines to 1.
// A line-count model predicts 1.5x for that second one and is wrong. The bf16 win is
// bounded by the 2x sector reduction and lands at 1.36-1.57x.
//
// NUMERICS: the bf16 path rounds table values to bf16 (8-bit mantissa) and is NOT
// bit-exact. Error enters once per gathered row and does not compound, because the
// sum accumulates in fp32. Measured gather relative error 9.3e-4 to 1.7e-3, and real
// val_bpb cost +0.00007 to +0.00014 -- but it must never be run through a bit-exact
// assertion. gather_cuda.py gates it on a tolerance check instead.
//
// The row pitch is a runtime parameter so the table can be padded (last dim 64 rather
// than 48) without a separate kernel; the pad is what made the sector story testable,
// and it is kept because it is free.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

constexpr int D = 48;     // outputs per head
constexpr int D4 = 12;    // fp32 row: 48 floats = 12 x float4
constexpr int U16 = 6;    // bf16 row: 48 bf16 = 96 B = 6 x 16 B

// ------------------------------------------------------------------ fp32 table
// Bit-exact against the Triton kernel: same rows, same order, no reassociation.
template <int BLOCK_N, int NT>
__global__ __launch_bounds__(NT) void gather_fp32_kernel(
    const float4* __restrict__ W, const int64_t* __restrict__ IDX,
    float4* __restrict__ OUT, int N, int n_tables, int table_dim, int H, int tph,
    int pitch) {
  constexpr int UNITS = BLOCK_N * D4;
  constexpr int UPT = (UNITS + NT - 1) / NT;   // float4 units per thread

  const int h = blockIdx.y;
  const int n0 = blockIdx.x * BLOCK_N;
  const int t0 = h * tph;

  int tok[UPT], quad[UPT];
  bool live[UPT];
#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    const int u = threadIdx.x + k * NT;
    const int tt = n0 + u / D4;
    tok[k] = tt;
    quad[k] = u % D4;
    live[k] = (u < UNITS) && (tt < N);
  }

  float4 acc[UPT], v_cur[UPT], v_nxt[UPT];
  int64_t i_nxt[UPT], i_nxt2[UPT];
#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    acc[k] = v_cur[k] = v_nxt[k] = make_float4(0.f, 0.f, 0.f, 0.f);
    i_nxt[k] = i_nxt2[k] = 0;
  }

  // prologue: row for table 0, index for table 1
#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    if (!live[k]) continue;
    const int64_t i0 = IDX[(size_t)tok[k] * n_tables + t0];
    v_cur[k] = W[((size_t)t0 * table_dim + (size_t)i0) * pitch + quad[k]];
    if (tph > 1) i_nxt[k] = IDX[(size_t)tok[k] * n_tables + t0 + 1];
  }

  for (int t = 0; t < tph; ++t) {
    const size_t tnext = (size_t)(t0 + t + 1) * table_dim;
    // issue the next row and the index after it before consuming the current row
#pragma unroll
    for (int k = 0; k < UPT; ++k) {
      if (!live[k]) continue;
      if (t + 1 < tph) v_nxt[k] = W[(tnext + (size_t)i_nxt[k]) * pitch + quad[k]];
      if (t + 2 < tph) i_nxt2[k] = IDX[(size_t)tok[k] * n_tables + t0 + t + 2];
    }
#pragma unroll
    for (int k = 0; k < UPT; ++k) {
      if (!live[k]) continue;
      acc[k].x += v_cur[k].x;
      acc[k].y += v_cur[k].y;
      acc[k].z += v_cur[k].z;
      acc[k].w += v_cur[k].w;
      v_cur[k] = v_nxt[k];
      i_nxt[k] = i_nxt2[k];
    }
  }

#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    if (!live[k]) continue;
    OUT[((size_t)tok[k] * H + h) * D4 + quad[k]] = acc[k];
  }
}

// ------------------------------------------------------------------ bf16 table
// APPROXIMATE. Half the sectors per row; fp32 accumulator.
template <int BLOCK_N, int NT>
__global__ __launch_bounds__(NT) void gather_bf16_kernel(
    const uint4* __restrict__ Wb, const int64_t* __restrict__ IDX,
    float4* __restrict__ OUT, int N, int n_tables, int table_dim, int H, int tph,
    int pitch) {
  constexpr int UNITS = BLOCK_N * U16;
  constexpr int UPT = (UNITS + NT - 1) / NT;   // 16-byte units per thread

  const int h = blockIdx.y;
  const int n0 = blockIdx.x * BLOCK_N;
  const int t0 = h * tph;

  int tok[UPT], u16[UPT];
  bool live[UPT];
#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    const int u = threadIdx.x + k * NT;
    const int tt = n0 + u / U16;
    tok[k] = tt;
    u16[k] = u % U16;
    live[k] = (u < UNITS) && (tt < N);
  }

  float acc[UPT][8];                            // 8 bf16 per 16-byte unit
  uint4 v_cur[UPT], v_nxt[UPT];
  int64_t i_nxt[UPT], i_nxt2[UPT];
#pragma unroll
  for (int k = 0; k < UPT; ++k) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[k][j] = 0.f;
    v_cur[k] = v_nxt[k] = make_uint4(0, 0, 0, 0);
    i_nxt[k] = i_nxt2[k] = 0;
  }

#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    if (!live[k]) continue;
    const int64_t i0 = IDX[(size_t)tok[k] * n_tables + t0];
    v_cur[k] = Wb[((size_t)t0 * table_dim + (size_t)i0) * pitch + u16[k]];
    if (tph > 1) i_nxt[k] = IDX[(size_t)tok[k] * n_tables + t0 + 1];
  }

  for (int t = 0; t < tph; ++t) {
    const size_t tnext = (size_t)(t0 + t + 1) * table_dim;
#pragma unroll
    for (int k = 0; k < UPT; ++k) {
      if (!live[k]) continue;
      if (t + 1 < tph) v_nxt[k] = Wb[(tnext + (size_t)i_nxt[k]) * pitch + u16[k]];
      if (t + 2 < tph) i_nxt2[k] = IDX[(size_t)tok[k] * n_tables + t0 + t + 2];
    }
#pragma unroll
    for (int k = 0; k < UPT; ++k) {
      if (!live[k]) continue;
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v_cur[k]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(b2[j]);
        acc[k][2 * j] += f.x;
        acc[k][2 * j + 1] += f.y;
      }
      v_cur[k] = v_nxt[k];
      i_nxt[k] = i_nxt2[k];
    }
  }

#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    if (!live[k]) continue;
    const size_t base = ((size_t)tok[k] * H + h) * D4 + (size_t)u16[k] * 2;
    OUT[base] = make_float4(acc[k][0], acc[k][1], acc[k][2], acc[k][3]);
    OUT[base + 1] = make_float4(acc[k][4], acc[k][5], acc[k][6], acc[k][7]);
  }
}

}  // namespace

#define DISPATCH(KER, ...)                                                             \
  do {                                                                                 \
    const int key = (int)block_n * 1000 + (int)nthreads;                               \
    switch (key) {                                                                     \
      case 64 * 1000 + 256:                                                            \
        KER<64, 256><<<grid64, 256, 0, stream>>>(__VA_ARGS__); break;                  \
      case 128 * 1000 + 256:                                                           \
        KER<128, 256><<<grid128, 256, 0, stream>>>(__VA_ARGS__); break;                \
      case 128 * 1000 + 512:                                                           \
        KER<128, 512><<<grid128, 512, 0, stream>>>(__VA_ARGS__); break;                \
      case 256 * 1000 + 512:                                                           \
        KER<256, 512><<<grid256, 512, 0, stream>>>(__VA_ARGS__); break;                \
      default:                                                                         \
        TORCH_CHECK(false, "unsupported (block_n, nthreads) = (", block_n, ", ",       \
                    nthreads, "); see gather_cuda.CFGS");                              \
    }                                                                                  \
  } while (0)

// last dim 48 = packed row, 64 = padded row (only the first 48 values are read)
#define CHECK_TABLE(T)                                                                 \
  TORCH_CHECK((T).dim() == 3 && ((T).size(2) == 48 || (T).size(2) == 64) &&            \
                  (T).is_contiguous(),                                                 \
              "table must be a contiguous [n_tables, table_dim, 48 or 64]");           \
  TORCH_CHECK(IDX.scalar_type() == torch::kLong && IDX.is_cuda(),                      \
              "index must be int64 CUDA (as the native bit-pack kernel emits it)")

torch::Tensor gather_sum_fp32(const torch::Tensor& W, const torch::Tensor& IDX,
                              int64_t n_heads, int64_t tph, int64_t block_n,
                              int64_t nthreads) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat32, "table must be fp32 CUDA");
  CHECK_TABLE(W);
  const int pitch = (int)W.size(2) / 4;
  const int n_tables = (int)W.size(0), table_dim = (int)W.size(1), N = (int)IDX.size(0);
  auto out = torch::empty({N, n_heads, D}, W.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* wp = reinterpret_cast<const float4*>(W.data_ptr<float>());
  const auto* ip = IDX.data_ptr<int64_t>();
  auto* op = reinterpret_cast<float4*>(out.data_ptr<float>());
  const dim3 grid64((N + 63) / 64, (unsigned)n_heads);
  const dim3 grid128((N + 127) / 128, (unsigned)n_heads);
  const dim3 grid256((N + 255) / 256, (unsigned)n_heads);
  DISPATCH(gather_fp32_kernel, wp, ip, op, N, n_tables, table_dim, (int)n_heads,
           (int)tph, pitch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gather_sum_bf16(const torch::Tensor& Wb, const torch::Tensor& IDX,
                              int64_t n_heads, int64_t tph, int64_t block_n,
                              int64_t nthreads) {
  TORCH_CHECK(Wb.is_cuda() && Wb.scalar_type() == torch::kBFloat16,
              "table must be bf16 CUDA");
  CHECK_TABLE(Wb);
  const int pitch = (int)Wb.size(2) / 8;
  const int n_tables = (int)Wb.size(0), table_dim = (int)Wb.size(1), N = (int)IDX.size(0);
  auto out = torch::empty({N, n_heads, D}, Wb.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* wp = reinterpret_cast<const uint4*>(Wb.data_ptr());
  const auto* ip = IDX.data_ptr<int64_t>();
  auto* op = reinterpret_cast<float4*>(out.data_ptr<float>());
  const dim3 grid64((N + 63) / 64, (unsigned)n_heads);
  const dim3 grid128((N + 127) / 128, (unsigned)n_heads);
  const dim3 grid256((N + 255) / 256, (unsigned)n_heads);
  DISPATCH(gather_bf16_kernel, wp, ip, op, N, n_tables, table_dim, (int)n_heads,
           (int)tph, pitch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_sum_fp32", &gather_sum_fp32,
        "pipelined gather+sum, fp32 table (bit-exact)");
  m.def("gather_sum_bf16", &gather_sum_bf16,
        "pipelined gather+sum, bf16 table (approximate)");
}
