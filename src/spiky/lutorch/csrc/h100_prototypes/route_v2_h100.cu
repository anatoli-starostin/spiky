// SCRATCH route_shared v2: kill shared bank-conflicts (ncu: route_shared still L1/TEX
// bound ~96%). Two changes vs v1:
//   (a) z-slice stored COLUMN-MAJOR in shared: zsh[c*BLOCK_N + lt].
//   (b) thread->work mapping is TOKEN-INNER: a warp = 32 consecutive tokens of the SAME
//       table. Then each (table,pair) read across the warp is zsh[col*BLOCK_N + lt] with
//       lt = 0..31 consecutive -> 32 distinct banks -> CONFLICT-FREE; and the anchor
//       reads ash[t*nap+p] are identical across the warp -> a broadcast (1 transaction).
// Same MSB index math => bit-exact vs native. Output IDX [N, n_tables] int64.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
constexpr int DIN = 48;

template <int BLOCK_N, int NT>
__global__ __launch_bounds__(NT) void route_v2_kernel(
    const float* __restrict__ Z, const int64_t* __restrict__ Aa,
    const int64_t* __restrict__ Ab, int64_t* __restrict__ IDX,
    int N, int n_tables, int H, int tph, int nap, int Zwidth) {
  extern __shared__ char smem[];
  float* zsh = reinterpret_cast<float*>(smem);          // [DIN * BLOCK_N] column-major
  int* ash = reinterpret_cast<int*>(zsh + DIN * BLOCK_N);   // [tph * nap]
  int* bsh = ash + tph * nap;

  const int h = blockIdx.y;
  const int n0 = blockIdx.x * BLOCK_N;
  const int t0 = h * tph;
  const int hcol = h * DIN;

  // stage z transposed: global read coalesced over c (consecutive), shared write column-major
  for (int u = threadIdx.x; u < BLOCK_N * DIN; u += NT) {
    const int lt = u / DIN, c = u % DIN, gt = n0 + lt;
    zsh[c * BLOCK_N + lt] = (gt < N) ? Z[(size_t)gt * Zwidth + hcol + c] : 0.f;
  }
  for (int u = threadIdx.x; u < tph * nap; u += NT) {
    ash[u] = (int)Aa[(size_t)t0 * nap + u] - hcol;
    bsh[u] = (int)Ab[(size_t)t0 * nap + u] - hcol;
  }
  __syncthreads();

  // token-inner mapping: lt = tid % BLOCK_N (warp-consecutive tokens), table group = tid / BLOCK_N
  constexpr int TABLES_PER_PASS = NT / BLOCK_N;
  const int lt = threadIdx.x % BLOCK_N;
  const int tgrp = threadIdx.x / BLOCK_N;
  const int gt = n0 + lt;
  const bool tok_live = gt < N;
  for (int t = tgrp; t < tph; t += TABLES_PER_PASS) {
    if (!tok_live) break;
    const int* ap = ash + t * nap;
    const int* bp = bsh + t * nap;
    int64_t idx = 0;
    for (int p = 0; p < nap; ++p) {
      const float d = zsh[ap[p] * BLOCK_N + lt] - zsh[bp[p] * BLOCK_N + lt];
      if (d > 0.f) idx |= ((int64_t)1 << (nap - 1 - p));
    }
    IDX[(size_t)gt * n_tables + (t0 + t)] = idx;
  }
}
}  // namespace

torch::Tensor route_v2(const torch::Tensor& Z, const torch::Tensor& Aa,
                       const torch::Tensor& Ab, int64_t n_heads, int64_t tph,
                       int64_t nap, int64_t block_n, int64_t nthreads) {
  TORCH_CHECK(Z.is_cuda() && Z.scalar_type() == torch::kFloat32 && Z.is_contiguous(),
              "Z contiguous fp32 CUDA");
  const int N = (int)Z.size(0), Zwidth = (int)Z.size(1);
  const int n_tables = (int)(n_heads * tph);
  auto idx = torch::empty({N, n_tables}, Z.options().dtype(torch::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* zp = Z.data_ptr<float>();
  const auto* aap = Aa.data_ptr<int64_t>();
  const auto* abp = Ab.data_ptr<int64_t>();
  auto* ip = idx.data_ptr<int64_t>();
  const size_t smem = (size_t)DIN * block_n * sizeof(float) + (size_t)2 * tph * nap * sizeof(int);

#define LR2(BN, NTH)                                                                   \
  do {                                                                                 \
    static_assert(NTH % BN == 0, "NT must be a multiple of BLOCK_N");                  \
    cudaFuncSetAttribute(route_v2_kernel<BN, NTH>,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
    dim3 grid((N + BN - 1) / BN, (unsigned)n_heads);                                   \
    route_v2_kernel<BN, NTH><<<grid, NTH, smem, stream>>>(                             \
        zp, aap, abp, ip, N, n_tables, (int)n_heads, (int)tph, (int)nap, Zwidth);      \
  } while (0)

  const int key = (int)block_n * 1000 + (int)nthreads;
  switch (key) {
    case 64 * 1000 + 256: LR2(64, 256); break;
    case 64 * 1000 + 512: LR2(64, 512); break;
    case 128 * 1000 + 256: LR2(128, 256); break;
    case 128 * 1000 + 512: LR2(128, 512); break;
    case 32 * 1000 + 256: LR2(32, 256); break;
    default: TORCH_CHECK(false, "unsupported (block_n, nthreads)");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route_v2", &route_v2, "routing, column-major shared z + token-inner map (bit-exact)");
}
