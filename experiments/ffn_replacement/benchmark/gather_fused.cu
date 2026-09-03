// FUSED routing + gather. RTX 5090 adaptation of gather_fused_v2_h100.cu (nebius/H100).
//
// Same core idea, which is the whole point of the port: fuse routing and gather into one
// kernel and keep the index in SHARED, so the 48-96 MB int64 index array is never written
// to HBM and never read back. On the 5090 that matters most for 0127, whose index is
// 96 MB = exactly this GPU's L2.
//
// ncu is unavailable on the 5090 (ERR_NVGPUCTRPERM), so "the HBM index write is gone" was
// measured by control instead: `spill=true` runs this identical kernel and additionally
// writes the index out, isolating exactly the write the fusion removes. Measured
// 0126 +0.0414 ms (48 MB), 0127 +0.0870 ms (96 MB), 0128 +0.0387 ms (48 MB) -- 0127's
// penalty is 2.1-2.25x the others against an index exactly 2x the size, at a consistent
// 1.1-1.2 TB/s, i.e. HBM write bandwidth. `spill` exists ONLY for that experiment.
//
// Three 5090-specific changes vs the H100 version:
//
// 1. bf16 TABLE. On the H100 bf16 lost, so nebius fused against the fp32/Triton gather.
//    Here bf16 won 1.36-1.57x (committed da6a3f5d), so the fused kernel must gather from
//    a bf16 table with fp32 accumulation. Templated on UPR (units per row): 12 float4 for
//    fp32, 6 uint4 for bf16.
//
// 2. BOTH routing regimes, selectable. nebius found routing is regime-dependent and the
//    5090 reproduces it: v2 (col-major z + token-inner warps) wins at nap8 (0128: 1.78x
//    native) but REGRESSES at nap7 (0126: 0.70x), where v1 (row-major z + table-inner)
//    wins (1.30x). The H100 fused kernel hardcodes v2; here it is a template parameter so
//    each model can take its own winner.
//
// 3. uint8 index in shared instead of int32. The index is a bit-pack of nap <= 8 pairs,
//    so it fits a byte exactly (nap7 -> 0..127, nap8 -> 0..255). Cuts the idxsh footprint
//    4x, which is what makes tph=128 (0127) fit at a useful BLOCK_N.
//
// Also adds the row-1-ahead prefetch that was worth 1.19-1.30x in the standalone kernel.
// The index now comes from shared, so its latency is gone, but the ROW load latency is
// not -- it still pays to have the next row in flight while summing the current one.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {
constexpr int DIN = 48, D4 = 12;

template <int BLOCK_N, int UPR, bool V2>
__global__ __launch_bounds__(BLOCK_N * UPR) void fused_kernel(
    const float* __restrict__ Z, const int64_t* __restrict__ Aa,
    const int64_t* __restrict__ Ab, const void* __restrict__ Wv,
    float4* __restrict__ OUT, int64_t* __restrict__ IDXOUT,
    int N, int n_tables, int table_dim, int H, int tph,
    int nap, int Zwidth, int pitch) {
  constexpr int NT = BLOCK_N * UPR;
  constexpr int ACC = (UPR == 6) ? 8 : 4;      // floats accumulated per thread

  extern __shared__ char smem[];
  float* zsh = reinterpret_cast<float*>(smem);
  int* ash = reinterpret_cast<int*>(zsh + DIN * BLOCK_N);
  int* bsh = ash + tph * nap;
  uint8_t* idxsh = reinterpret_cast<uint8_t*>(bsh + tph * nap);

  const int h = blockIdx.y, n0 = blockIdx.x * BLOCK_N, t0 = h * tph, hcol = h * DIN;

  // stage z: global read coalesced over c either way; shared layout differs by regime
  for (int u = threadIdx.x; u < BLOCK_N * DIN; u += NT) {
    const int lt = u / DIN, c = u % DIN, gt = n0 + lt;
    const float v = (gt < N) ? Z[(size_t)gt * Zwidth + hcol + c] : 0.f;
    if constexpr (V2) zsh[c * BLOCK_N + lt] = v;   // column-major
    else zsh[lt * DIN + c] = v;                    // row-major
  }
  for (int u = threadIdx.x; u < tph * nap; u += NT) {
    ash[u] = (int)Aa[(size_t)t0 * nap + u] - hcol;
    bsh[u] = (int)Ab[(size_t)t0 * nap + u] - hcol;
  }
  __syncthreads();

  // ---- phase 1: routing -> idxsh[t * BLOCK_N + lt] (col-major both ways, for phase 2)
  if constexpr (V2) {
    constexpr int TPP = NT / BLOCK_N;
    const int lt = threadIdx.x % BLOCK_N, tgrp = threadIdx.x / BLOCK_N;
    const bool live = (n0 + lt) < N;
    for (int t = tgrp; t < tph; t += TPP) {
      int idx = 0;
      if (live) {
        const int* ap = ash + t * nap;
        const int* bp = bsh + t * nap;
        for (int p = 0; p < nap; ++p)
          if (zsh[ap[p] * BLOCK_N + lt] - zsh[bp[p] * BLOCK_N + lt] > 0.f)
            idx |= (1 << (nap - 1 - p));
      }
      idxsh[t * BLOCK_N + lt] = (uint8_t)idx;
    }
  } else {
    for (int u = threadIdx.x; u < BLOCK_N * tph; u += NT) {
      const int lt = u / tph, t = u % tph, gt = n0 + lt;
      int idx = 0;
      if (gt < N) {
        const float* zr = zsh + lt * DIN;
        const int* ap = ash + t * nap;
        const int* bp = bsh + t * nap;
        for (int p = 0; p < nap; ++p)
          if (zr[ap[p]] - zr[bp[p]] > 0.f) idx |= (1 << (nap - 1 - p));
      }
      idxsh[t * BLOCK_N + lt] = (uint8_t)idx;
    }
  }
  __syncthreads();

  // CONTROL PATH ONLY (never used in the fast path): additionally write the index out to
  // HBM, exactly as the non-fused pipeline must. ncu is blocked on this box, so this is
  // how the "the index write is gone" claim gets measured -- the fused/spill difference
  // is that write, and it should scale with the index size.
  if (IDXOUT != nullptr) {
    for (int u = threadIdx.x; u < BLOCK_N * tph; u += NT) {
      const int t = u / BLOCK_N, lt = u % BLOCK_N, gt = n0 + lt;
      if (gt < N) IDXOUT[(size_t)gt * n_tables + t0 + t] = (int64_t)idxsh[t * BLOCK_N + lt];
    }
  }

  // ---- phase 2: output-stationary gather, index read from shared, row 1 ahead
  const int ltok = threadIdx.x / UPR, unit = threadIdx.x % UPR;
  const int tok = n0 + ltok;
  if (tok >= N) return;

  float acc[ACC];
#pragma unroll
  for (int j = 0; j < ACC; ++j) acc[j] = 0.f;

  if constexpr (UPR == 6) {                       // ---- bf16 table
    const uint4* W = reinterpret_cast<const uint4*>(Wv);
    uint4 cur = W[((size_t)t0 * table_dim + (size_t)idxsh[ltok]) * pitch + unit];
    for (int t = 0; t < tph; ++t) {
      uint4 nxt = make_uint4(0, 0, 0, 0);
      if (t + 1 < tph)
        nxt = W[((size_t)(t0 + t + 1) * table_dim +
                 (size_t)idxsh[(t + 1) * BLOCK_N + ltok]) * pitch + unit];
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&cur);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(b2[j]);
        acc[2 * j] += f.x;
        acc[2 * j + 1] += f.y;
      }
      cur = nxt;
    }
    const size_t base = ((size_t)tok * H + h) * D4 + (size_t)unit * 2;
    OUT[base] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    OUT[base + 1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
  } else {                                        // ---- fp32 table (bit-exact)
    const float4* W = reinterpret_cast<const float4*>(Wv);
    float4 cur = W[((size_t)t0 * table_dim + (size_t)idxsh[ltok]) * pitch + unit];
    for (int t = 0; t < tph; ++t) {
      float4 nxt = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t + 1 < tph)
        nxt = W[((size_t)(t0 + t + 1) * table_dim +
                 (size_t)idxsh[(t + 1) * BLOCK_N + ltok]) * pitch + unit];
      acc[0] += cur.x;
      acc[1] += cur.y;
      acc[2] += cur.z;
      acc[3] += cur.w;
      cur = nxt;
    }
    OUT[((size_t)tok * H + h) * D4 + unit] =
        make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
}
}  // namespace

torch::Tensor fused(const torch::Tensor& Z, const torch::Tensor& Aa,
                    const torch::Tensor& Ab, const torch::Tensor& W,
                    int64_t n_heads, int64_t tph, int64_t nap, int64_t block_n,
                    bool v2, bool spill) {
  TORCH_CHECK(Z.is_cuda() && Z.scalar_type() == torch::kFloat32 && Z.is_contiguous(),
              "Z contiguous fp32 CUDA");
  TORCH_CHECK(W.is_cuda() && W.is_contiguous() && W.dim() == 3 &&
                  (W.size(2) == 48 || W.size(2) == 64), "W [nt, td, 48|64]");
  const bool bf16 = (W.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(bf16 || W.scalar_type() == torch::kFloat, "W must be fp32 or bf16");
  TORCH_CHECK(nap <= 8, "index is packed into a byte in shared: nap must be <= 8");
  const int N = (int)Z.size(0), Zwidth = (int)Z.size(1);
  const int n_tables = (int)W.size(0), table_dim = (int)W.size(1);
  TORCH_CHECK(table_dim <= 256, "table_dim must be <= 256 for a uint8 shared index");
  const int pitch = (int)W.size(2) / (bf16 ? 8 : 4);
  auto out = torch::empty({N, n_heads, DIN},
                          W.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* zp = Z.data_ptr<float>();
  const auto* aap = Aa.data_ptr<int64_t>();
  const auto* abp = Ab.data_ptr<int64_t>();
  const void* wp = W.data_ptr();
  auto* op = reinterpret_cast<float4*>(out.data_ptr<float>());
  auto idxout = spill ? torch::empty({N, n_tables}, Z.options().dtype(torch::kLong))
                      : torch::Tensor();
  int64_t* iop = spill ? idxout.data_ptr<int64_t>() : nullptr;
  const size_t smem = (size_t)DIN * block_n * sizeof(float)
                    + (size_t)2 * tph * nap * sizeof(int)
                    + (size_t)tph * block_n;                  // uint8 index

#define LF(BN, UPR, V2)                                                              \
  do {                                                                               \
    constexpr int NTH = (BN) * (UPR);                                                \
    static_assert(NTH <= 1024, "too many threads");                                  \
    cudaFuncSetAttribute(fused_kernel<BN, UPR, V2>,                                  \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);    \
    dim3 grid((N + (BN) - 1) / (BN), (unsigned)n_heads);                             \
    fused_kernel<BN, UPR, V2><<<grid, NTH, smem, stream>>>(                          \
        zp, aap, abp, wp, op, iop, N, n_tables, table_dim, (int)n_heads, (int)tph,   \
        (int)nap, Zwidth, pitch);                                                    \
  } while (0)

// fp32 needs 12 threads per token, so BLOCK_N=128 would want 1536 threads -- over the
// 1024 limit. bf16 needs only 6, so it reaches BLOCK_N=128 at 768.
#define PICK(BN)                                                                     \
  do {                                                                               \
    if (bf16 && v2) LF(BN, 6, true);                                                 \
    else if (bf16) LF(BN, 6, false);                                                 \
    else if (v2) LF(BN, 12, true);                                                   \
    else LF(BN, 12, false);                                                          \
  } while (0)

  switch ((int)block_n) {
    case 32: PICK(32); break;
    case 64: PICK(64); break;
    case 128:
      TORCH_CHECK(bf16, "block_n=128 requires the bf16 table (fp32 would need 1536 "
                        "threads); use block_n 32 or 64 for fp32");
      if (v2) LF(128, 6, true);
      else LF(128, 6, false);
      break;
    default: TORCH_CHECK(false, "block_n must be 32/64/128");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused", &fused,
        "fused routing+gather, index in shared; v2/v1 routing, fp32/bf16 table");
}
