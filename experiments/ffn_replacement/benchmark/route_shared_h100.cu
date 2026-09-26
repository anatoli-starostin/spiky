// SCRATCH Priority-A routing fix: standalone anchor-compare that stages each token's
// z-slice in SHARED once and reuses it across all tph tables, moving the redundant z
// reads off the L1/TEX path (ncu showed the native kernel is L1-throughput-bound at
// nap8: 99% L1, 27% compute). Same output as the native MSB kernel: IDX [N, n_tables]
// int64. Same index math (delta>0, MSB pack) => bit-exact vs native.
//
// Per-head: head h's tables read only z cols [h*48, h*48+48), so stage 48 cols/token.
// Also stages this head's anchor columns (a-h*48, b-h*48) in shared as int32 to keep
// them off L1 too.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
constexpr int DIN = 48;

template <int BLOCK_N, int NT>
__global__ __launch_bounds__(NT) void route_shared_kernel(
    const float* __restrict__ Z, const int64_t* __restrict__ Aa,
    const int64_t* __restrict__ Ab, int64_t* __restrict__ IDX,
    int N, int n_tables, int H, int tph, int nap, int Zwidth) {
  extern __shared__ char smem[];
  float* zsh = reinterpret_cast<float*>(smem);          // [BLOCK_N * DIN]
  int* ash = reinterpret_cast<int*>(zsh + BLOCK_N * DIN);   // [tph * nap] local a-cols
  int* bsh = ash + tph * nap;                              // [tph * nap] local b-cols

  const int h = blockIdx.y;
  const int n0 = blockIdx.x * BLOCK_N;
  const int t0 = h * tph;
  const int hcol = h * DIN;

  for (int u = threadIdx.x; u < BLOCK_N * DIN; u += NT) {
    const int lt = u / DIN, c = u % DIN, gt = n0 + lt;
    zsh[u] = (gt < N) ? Z[(size_t)gt * Zwidth + hcol + c] : 0.f;
  }
  for (int u = threadIdx.x; u < tph * nap; u += NT) {
    ash[u] = (int)Aa[(size_t)t0 * nap + u] - hcol;
    bsh[u] = (int)Ab[(size_t)t0 * nap + u] - hcol;
  }
  __syncthreads();

  for (int u = threadIdx.x; u < BLOCK_N * tph; u += NT) {
    const int lt = u / tph, t = u % tph, gt = n0 + lt;
    if (gt >= N) continue;
    const float* zr = zsh + lt * DIN;
    const int* ap = ash + t * nap;
    const int* bp = bsh + t * nap;
    int64_t idx = 0;
    for (int p = 0; p < nap; ++p) {
      const float d = zr[ap[p]] - zr[bp[p]];
      if (d > 0.f) idx |= ((int64_t)1 << (nap - 1 - p));
    }
    IDX[(size_t)gt * n_tables + (t0 + t)] = idx;
  }
}
}  // namespace

torch::Tensor route_shared(const torch::Tensor& Z, const torch::Tensor& Aa,
                           const torch::Tensor& Ab, int64_t n_heads, int64_t tph,
                           int64_t nap, int64_t block_n, int64_t nthreads) {
  TORCH_CHECK(Z.is_cuda() && Z.scalar_type() == torch::kFloat32 && Z.is_contiguous(),
              "Z contiguous fp32 CUDA [N, n_heads*48]");
  const int N = (int)Z.size(0), Zwidth = (int)Z.size(1);
  const int n_tables = (int)(n_heads * tph);
  auto idx = torch::empty({N, n_tables}, Z.options().dtype(torch::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* zp = Z.data_ptr<float>();
  const auto* aap = Aa.data_ptr<int64_t>();
  const auto* abp = Ab.data_ptr<int64_t>();
  auto* ip = idx.data_ptr<int64_t>();
  const size_t smem = (size_t)block_n * DIN * sizeof(float) + (size_t)2 * tph * nap * sizeof(int);

#define LAUNCHR(BN, NTH)                                                               \
  do {                                                                                 \
    cudaFuncSetAttribute(route_shared_kernel<BN, NTH>,                                 \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
    dim3 grid((N + BN - 1) / BN, (unsigned)n_heads);                                   \
    route_shared_kernel<BN, NTH><<<grid, NTH, smem, stream>>>(                         \
        zp, aap, abp, ip, N, n_tables, (int)n_heads, (int)tph, (int)nap, Zwidth);      \
  } while (0)

  const int key = (int)block_n * 1000 + (int)nthreads;
  switch (key) {
    case 64 * 1000 + 256: LAUNCHR(64, 256); break;
    case 128 * 1000 + 256: LAUNCHR(128, 256); break;
    case 128 * 1000 + 512: LAUNCHR(128, 512); break;
    case 256 * 1000 + 256: LAUNCHR(256, 256); break;
    case 64 * 1000 + 128: LAUNCHR(64, 128); break;
    default: TORCH_CHECK(false, "unsupported (block_n, nthreads)");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route_shared", &route_shared, "anchor-compare with z staged in shared (bit-exact)");
}
