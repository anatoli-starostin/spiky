// SCRATCH: FUSED v2-routing + gather. Not committed. Target nap8/0128.
// route_v2 made routing L2-bound on the 50 MB int64 index WRITE to HBM. This keeps the
// index in SHARED (never materialized to HBM) and feeds it straight to the gather:
//   phase 1: v2 routing (col-major z in shared, token-inner warp, conflict-free) ->
//            indices into shared idxsh (col-major, conflict-free write);
//   phase 2: output-stationary gather+sum reading idxsh from shared -> OUT.
// Same index math + same rows + same fp32 sum order => bit-exact vs native+Triton.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
constexpr int DIN = 48, D4 = 12;

template <int BLOCK_N, int NT>
__global__ __launch_bounds__(NT) void fused_v2_kernel(
    const float* __restrict__ Z, const int64_t* __restrict__ Aa,
    const int64_t* __restrict__ Ab, const float4* __restrict__ W,
    float4* __restrict__ OUT, int N, int n_tables, int table_dim, int H, int tph,
    int nap, int Zwidth, int pitch) {
  extern __shared__ char smem[];
  float* zsh = reinterpret_cast<float*>(smem);            // [DIN*BLOCK_N] col-major
  int* ash = reinterpret_cast<int*>(zsh + DIN * BLOCK_N); // [tph*nap]
  int* bsh = ash + tph * nap;                             // [tph*nap]
  int* idxsh = bsh + tph * nap;                           // [tph*BLOCK_N] col-major

  const int h = blockIdx.y, n0 = blockIdx.x * BLOCK_N, t0 = h * tph, hcol = h * DIN;

  for (int u = threadIdx.x; u < BLOCK_N * DIN; u += NT) {
    const int lt = u / DIN, c = u % DIN, gt = n0 + lt;
    zsh[c * BLOCK_N + lt] = (gt < N) ? Z[(size_t)gt * Zwidth + hcol + c] : 0.f;
  }
  for (int u = threadIdx.x; u < tph * nap; u += NT) {
    ash[u] = (int)Aa[(size_t)t0 * nap + u] - hcol;
    bsh[u] = (int)Ab[(size_t)t0 * nap + u] - hcol;
  }
  __syncthreads();

  // phase 1: v2 routing (token-inner) -> idxsh[t*BLOCK_N + lt]  (conflict-free write)
  {
    constexpr int TPP = NT / BLOCK_N;
    const int lt = threadIdx.x % BLOCK_N, tgrp = threadIdx.x / BLOCK_N;
    const bool live = (n0 + lt) < N;
    for (int t = tgrp; t < tph; t += TPP) {
      int idx = 0;
      if (live) {
        const int* ap = ash + t * nap;
        const int* bp = bsh + t * nap;
        for (int p = 0; p < nap; ++p) {
          const float d = zsh[ap[p] * BLOCK_N + lt] - zsh[bp[p] * BLOCK_N + lt];
          if (d > 0.f) idx |= (1 << (nap - 1 - p));
        }
      }
      idxsh[t * BLOCK_N + lt] = idx;
    }
  }
  __syncthreads();

  // phase 2: output-stationary gather from shared idxsh (NT == BLOCK_N * D4)
  const int ltok = threadIdx.x / D4, quad = threadIdx.x % D4;
  const int tok = n0 + ltok;
  if (tok >= N) return;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int t = 0; t < tph; ++t) {
    const int idx = idxsh[t * BLOCK_N + ltok];
    const float4 row = W[((size_t)(t0 + t) * table_dim + (size_t)idx) * pitch + quad];
    acc.x += row.x; acc.y += row.y; acc.z += row.z; acc.w += row.w;
  }
  OUT[((size_t)tok * H + h) * D4 + quad] = acc;
}
}  // namespace

torch::Tensor fused_v2(const torch::Tensor& Z, const torch::Tensor& Aa,
                       const torch::Tensor& Ab, const torch::Tensor& W,
                       int64_t n_heads, int64_t tph, int64_t nap, int64_t block_n) {
  TORCH_CHECK(Z.is_cuda() && Z.scalar_type() == torch::kFloat32 && Z.is_contiguous(), "Z fp32");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat32 && W.is_contiguous() &&
                  W.dim() == 3 && (W.size(2) == 48 || W.size(2) == 64), "W fp32 [nt,td,48|64]");
  const int N = (int)Z.size(0), Zwidth = (int)Z.size(1);
  const int n_tables = (int)W.size(0), table_dim = (int)W.size(1), pitch = (int)W.size(2) / 4;
  auto out = torch::empty({N, n_heads, DIN}, W.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* zp = Z.data_ptr<float>();
  const auto* aap = Aa.data_ptr<int64_t>();
  const auto* abp = Ab.data_ptr<int64_t>();
  const auto* wp = reinterpret_cast<const float4*>(W.data_ptr<float>());
  auto* op = reinterpret_cast<float4*>(out.data_ptr<float>());
  const size_t smem = (size_t)DIN * block_n * sizeof(float) + (size_t)2 * tph * nap * sizeof(int)
                    + (size_t)tph * block_n * sizeof(int);

#define LF2(BN)                                                                        \
  do {                                                                                 \
    constexpr int NTH = BN * D4;                                                       \
    cudaFuncSetAttribute(fused_v2_kernel<BN, NTH>,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
    dim3 grid((N + BN - 1) / BN, (unsigned)n_heads);                                   \
    fused_v2_kernel<BN, NTH><<<grid, NTH, smem, stream>>>(                             \
        zp, aap, abp, wp, op, N, n_tables, table_dim, (int)n_heads, (int)tph,          \
        (int)nap, Zwidth, pitch);                                                      \
  } while (0)

  switch ((int)block_n) {
    case 32: LF2(32); break;
    case 64: LF2(64); break;
    case 85: LF2(85); break;
    default: TORCH_CHECK(false, "block_n must be 32/64/85");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_v2", &fused_v2, "fused v2-routing + gather, index kept in shared (bit-exact)");
}
