// Fused int8 power-of-two LUT read (eval only) for QuantisedLightFFN -- RTX 5090 (sm_120).
//
// One launch per call computes, for every (token, head) and every one of that head's T tables, the note's integer read
// (doc/research/lut_ablation/quantisation_simple.tex, Section 6):
//
//     acc_h[c] = sum_t  (W[c1_t][c] << (k'_t + 6))  +  (W[c2_t][c] << (k'_t + 6 - q_t))      int32, units of 2^-6
//
// skipping every cell whose shift is the discard code 15 (table skipped: k' below the window; second cell dropped:
// q > Q). Every row byte is sign-extended to int32 in registers and added into int32 accumulators that stay in registers
// for the whole table loop; the accumulators are converted to float once, when the unit is written. No float row, and no
// widened row, is ever written to memory. |acc| <= 2T * 128 << 10 = 2^25 for T = 128 (int32 headroom asserted in tests).
//
// Two regimes, one template (PROLOGUE):
//
//   PROLOGUE = false  ("cells")  -- the per-table integers come from torch (pow2_read.blend_exponents, compiled), packed
//        as CELLS[N, H, T, 3] = (c1, c2, shifts) with shifts = sh1 | sh2 << 4. The kernel stages them into shared and
//        accumulates. The integers are torch's own, so the output is bit-identical to pow2_read.int8_blend_read.
//
//   PROLOGUE = true   ("fused")  -- the kernel also computes the per-table integers itself from the compressed code Z:
//        anchor margins, address c1, the least-certain bit and c2, log2 s in the log domain, q, c_q, k', window, drop.
//        One launch for the whole read. NOT BIT-EXACT against the torch path: it follows pow2_read.blend_exponents op
//        for op (sequential sums over the anchor pairs, first-index argmin, std::fma exactly where inductor emits a fused
//        multiply-add, the file built with --fmad=false), but torch.compile's own float32 reduction order over the 8
//        anchor pairs matches no fixed order (it is not eager torch's tree order either), so a table sitting within an
//        ulp of a k' rounding boundary can take the other k'. Measured on abl_45: 11 of 75,497,472 tables (k' only; c1,
//        c2 and q always equal), bpb 1.1591491 vs 1.1591503. Opt-in only; the default regime is "cells".
//
// Cell width D is a runtime parameter. Table rows are stored with a STRIDE padded up to a multiple of 16 bytes (the
// vector load width): at D = 48 the stride is 48 (no padding); otherwise at most 15 bytes per row. Each output UNIT is 16
// lanes = one 16-byte (int4) load per row; a row takes ceil(D / 16) units. The last unit's lanes at or beyond D are
// predicated off at accumulation, so padding bytes never contribute (pack_tables also zeroes them; the tests fill them
// with garbage to prove the predicate alone suffices). Work assignment, not storage, is what is padded to the warp: the
// threads per block are BLOCK_N * UPR (a multiple of 32 for BLOCK_N in {32, 64, 128}); UPR = units per row is a template
// parameter covering D in (16 (UPR - 1), 16 UPR].
//
// Memory placement (per block of BLOCK_N tokens and one head):
//   shared, phase 0 : the block's cells (cells), or its z rows and the head's anchors (fused)
//   shared, phase 1 : cells[t * BLOCK_N + lt] = (c1, c2, shifts)                           (fused computes them here)
//   registers        : the float scalars (inv, g, beta, gamma, ln 2, c_q table; passed by value); in phase 2 the 16 int32
//                     accumulators of the thread's unit and the current 16-byte row load
//   global           : int8 rows (one 16-byte int4 load per unit per cell; LOAD16=false: four char4 loads), float output

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {
constexpr int NAP_MAX = 8;      // c1 / c2 fit a byte
constexpr int UPR_MAX = 8;      // D <= 128
constexpr uint8_t DISCARD = 15;

template <int BLOCK_N, int UPR, bool PROLOGUE, bool LOAD16>
__global__ __launch_bounds__(BLOCK_N * UPR) void p2_int8_kernel(
    const float* __restrict__ Z, const int* __restrict__ Aa, const int* __restrict__ Ab,
    const uint8_t* __restrict__ CELLS, uint8_t* __restrict__ CELLS_OUT, const int8_t* __restrict__ W,
    float* __restrict__ OUT,
    int N, int H, int T, int nap, int K, int din, int D, int stride, int lo, int hi, int Q,
    float inv, float g, float beta, float gamma, float ln2, float4 ctab_lo, float4 ctab_hi) {
  constexpr int NT = BLOCK_N * UPR;
  extern __shared__ char smem[];
  uint8_t* csh = reinterpret_cast<uint8_t*>(smem);                     // cells [T * BLOCK_N * 3]
  const size_t cells_bytes = (size_t)T * BLOCK_N * 3;
  float* zsh = reinterpret_cast<float*>(csh + cells_bytes + (4 - cells_bytes % 4) % 4);
  int* ash = reinterpret_cast<int*>(zsh + BLOCK_N * din);
  int* bsh = ash + T * nap;

  const int h = blockIdx.y, n0 = blockIdx.x * BLOCK_N, t0 = h * T;

  // ---- phase 0: stage the block's inputs into shared
  if constexpr (PROLOGUE) {
    for (int u = threadIdx.x; u < BLOCK_N * din; u += NT) {
      const int lt = u / din, c = u % din, gt = n0 + lt;
      zsh[u] = (gt < N) ? Z[((size_t)gt * H + h) * din + c] : 0.f;
    }
    for (int u = threadIdx.x; u < T * nap; u += NT) {
      ash[u] = Aa[(size_t)t0 * nap + u];
      bsh[u] = Ab[(size_t)t0 * nap + u];
    }
  } else {
    for (int u = threadIdx.x; u < BLOCK_N * T; u += NT) {
      const int lt = u / T, t = u % T, gt = n0 + lt;
      const size_t dst = ((size_t)t * BLOCK_N + lt) * 3;
      if (gt < N) {
        const size_t src = (((size_t)gt * H + h) * T + t) * 3;
        csh[dst] = CELLS[src];
        csh[dst + 1] = CELLS[src + 1];
        csh[dst + 2] = CELLS[src + 2];
      } else {
        csh[dst + 2] = DISCARD | (DISCARD << 4);
      }
    }
  }
  __syncthreads();

  // ---- phase 1 (fused regime): the per-table integers, table-inner over the block's tokens
  if constexpr (PROLOGUE) {
    const float ctab[8] = {ctab_lo.x, ctab_lo.y, ctab_lo.z, ctab_lo.w, ctab_hi.x, ctab_hi.y, ctab_hi.z, ctab_hi.w};
    for (int u = threadIdx.x; u < BLOCK_N * T; u += NT) {
      const int lt = u / T, t = u % T, gt = n0 + lt;
      const size_t dst = ((size_t)t * BLOCK_N + lt) * 3;
      if (gt >= N) {
        csh[dst + 2] = DISCARD | (DISCARD << 4);
        continue;
      }
      const float* zr = zsh + lt * din;
      const int* ap = ash + t * nap;
      const int* bp = bsh + t * nap;
      int c1 = 0, mj = 0;
      float S = 0.f, ls = 0.f, mv = 0.f;
      for (int p = 0; p < nap; ++p) {
        const float d = zr[ap[p]] - zr[bp[p]];
        if (d > 0.f) c1 |= (1 << (nap - 1 - p));
        const float m = std::fabs(d);
        S += m;
        const float x = beta * m;                                   // logsigmoid(x) = min(0, x) - log1p(exp(-|x|))
        ls += std::fmin(0.f, x) - std::log1p(std::exp(-std::fabs(x)));
        if (p == 0 || m < mv) { mv = m; mj = p; }
      }
      const int c2 = c1 ^ (1 << (nap - 1 - mj));
      // inductor emits these two a*b+c as fused multiply-adds (Triton fp-fusion), so they are std::fma here; nothing
      // else in this file is contracted (built with --fmad=false). See the header for why this is still not bit-exact.
      const float qf = std::fmin(std::fmax(std::floor(std::fma(mv, inv, 0.5f)), 0.f), 64.f);
      const float cq = (qf < 8.f) ? ctab[(int)qf] : 0.f;
      const float l2 = std::log2(S) + std::fma(gamma, ls, g) / ln2;  // divide by ln 2, as torch does
      const float kr = std::floor(l2 - cq + 0.5f);
      const bool skip = !(kr >= (float)lo);                          // also true for NaN / -inf (S == 0)
      const int k = skip ? lo : (kr > (float)hi ? hi : (int)kr);
      const int q = (int)qf;
      const bool drop = q > Q;
      const uint8_t sh1 = skip ? DISCARD : (uint8_t)(k + 6);
      const uint8_t sh2 = (skip || drop) ? DISCARD : (uint8_t)(k + 6 - q);
      csh[dst] = (uint8_t)c1;
      csh[dst + 1] = (uint8_t)c2;
      csh[dst + 2] = sh1 | (sh2 << 4);
      if (CELLS_OUT != nullptr) {                                    // debug / test: expose the integers
        const size_t o = (((size_t)gt * H + h) * T + t) * 3;
        CELLS_OUT[o] = (uint8_t)c1;
        CELLS_OUT[o + 1] = (uint8_t)c2;
        CELLS_OUT[o + 2] = sh1 | (sh2 << 4);
      }
    }
    __syncthreads();
  }

  // ---- phase 2: output-stationary integer accumulation, one 16-lane unit per thread
  const int ltok = threadIdx.x / UPR, unit = threadIdx.x % UPR;
  const int tok = n0 + ltok;
  if (tok >= N) return;
  const int lane0 = unit * 16;
  const int nl = (D - lane0 < 16) ? D - lane0 : 16;                  // real lanes in this unit (tail predicate)

  int acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = 0;

  const size_t pitch = (size_t)K * stride;
  const size_t base = (size_t)t0 * pitch;
  for (int t = 0; t < T; ++t) {
    const uint8_t* cell = csh + ((size_t)t * BLOCK_N + ltok) * 3;
    const int sh1 = cell[2] & 15, sh2 = cell[2] >> 4;
    const size_t tb = base + (size_t)t * pitch;
    for (int r = 0; r < 2; ++r) {
      const int sh = (r == 0) ? sh1 : sh2;
      if (sh == DISCARD) continue;
      const int8_t* row = W + tb + (size_t)cell[r] * stride + (size_t)lane0;
      if constexpr (LOAD16) {
        const int4 v = *reinterpret_cast<const int4*>(row);          // one 16-byte load into a register
        const unsigned w4[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
#pragma unroll
        for (int j = 0; j < 16; ++j) {                               // sign-extend in registers, shift, add
          const int lane = (j < nl) ? (int)(int8_t)(uint8_t)(w4[j >> 2] >> (8 * (j & 3))) : 0;
          acc[j] += lane << sh;
        }
      } else {
        const char4* c4 = reinterpret_cast<const char4*>(row); // four 4-byte loads
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const char4 v = c4[j];
          const int b = 4 * j;
          acc[b] += (b < nl ? (int)v.x : 0) << sh;
          acc[b + 1] += (b + 1 < nl ? (int)v.y : 0) << sh;
          acc[b + 2] += (b + 2 < nl ? (int)v.z : 0) << sh;
          acc[b + 3] += (b + 3 < nl ? (int)v.w : 0) << sh;
        }
      }
    }
  }
  const size_t ob = ((size_t)tok * H + h) * D + (size_t)lane0;
  if (nl == 16 && D % 16 == 0) {                                     // output row 16-byte aligned: vector writes
    float4* const o4 = reinterpret_cast<float4*>(OUT + ob);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o4[j] = make_float4((float)acc[4 * j], (float)acc[4 * j + 1], (float)acc[4 * j + 2], (float)acc[4 * j + 3]);
  } else {
    for (int j = 0; j < nl; ++j) OUT[ob + j] = (float)acc[j];
  }
}
}  // namespace

// Z [N, H, din] fp32 (fused) or empty; Aa, Ab [H, T, nap] int32 local column indices (fused) or empty; CELLS
// [N, H, T, 3] uint8 (cells) or empty; W [H * T * K, stride] int8 with stride = ceil(D / 16) * 16.
// Returns float32 [N, H, D] = the int32 accumulators, converted once.
torch::Tensor p2_read(const torch::Tensor& Z, const torch::Tensor& Aa, const torch::Tensor& Ab, const torch::Tensor& CELLS,
                      const torch::Tensor& W, int64_t N, int64_t H, int64_t T, int64_t nap, int64_t K, int64_t din,
                      int64_t D, int64_t lo, int64_t hi, int64_t Q, double inv, double g, double beta, double gamma,
                      const std::vector<double>& ctab, int64_t block_n, bool fused, bool load16) {
  const int64_t stride = (D + 15) / 16 * 16;
  const int64_t upr = stride / 16;
  TORCH_CHECK(D >= 1 && upr <= UPR_MAX, "D must be in [1, ", 16 * UPR_MAX, "]");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kInt8 && W.is_contiguous() && W.dim() == 2 &&
                  W.size(1) == stride && W.size(0) == H * T * K,
              "W must be contiguous int8 CUDA [H*T*K, ceil(D/16)*16]");
  TORCH_CHECK(nap >= 1 && nap <= NAP_MAX && K == (1 << nap), "nap must be in [1, 8] with K = 2^nap");
  TORCH_CHECK(lo >= -3 && hi <= 4 && lo <= hi && Q >= 0 && Q <= 3, "window must lie in [-3, 4] and Q in [0, 3]");
  TORCH_CHECK(ctab.size() == 8, "ctab must hold c_q for q = 0..7");
  if (fused) {
    TORCH_CHECK(Z.is_cuda() && Z.scalar_type() == torch::kFloat32 && Z.is_contiguous() && Z.dim() == 3 &&
                    Z.size(0) == N && Z.size(1) == H && Z.size(2) == din, "Z must be contiguous fp32 CUDA [N, H, din]");
    TORCH_CHECK(Aa.is_cuda() && Ab.is_cuda() && Aa.scalar_type() == torch::kInt32 && Ab.scalar_type() == torch::kInt32 &&
                    Aa.is_contiguous() && Ab.is_contiguous() && Aa.numel() == H * T * nap && Ab.numel() == H * T * nap,
                "anchors must be contiguous int32 CUDA [H, T, nap]");
  }
  if (!fused || (CELLS.defined() && CELLS.numel() > 0)) {
    TORCH_CHECK(CELLS.is_cuda() && CELLS.scalar_type() == torch::kUInt8 && CELLS.is_contiguous() &&
                    CELLS.numel() == N * H * T * 3, "CELLS must be contiguous uint8 CUDA [N, H, T, 3]");
  }
  auto out = torch::empty({N, H, D}, W.options().dtype(torch::kFloat32));
  if (N == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* zp = fused ? Z.data_ptr<float>() : nullptr;
  const int* ap = fused ? Aa.data_ptr<int>() : nullptr;
  const int* bp = fused ? Ab.data_ptr<int>() : nullptr;
  const uint8_t* cp = fused ? nullptr : CELLS.data_ptr<uint8_t>();
  // fused regime: a CELLS tensor, when given, is an OUTPUT receiving the kernel's own per-table integers (tests)
  uint8_t* cop = (fused && CELLS.defined() && CELLS.numel() > 0) ? CELLS.data_ptr<uint8_t>() : nullptr;
  const int8_t* wp = W.data_ptr<int8_t>();
  float* op = out.data_ptr<float>();
  const float4 clo = make_float4((float)ctab[0], (float)ctab[1], (float)ctab[2], (float)ctab[3]);
  const float4 chi = make_float4((float)ctab[4], (float)ctab[5], (float)ctab[6], (float)ctab[7]);
  const float ln2 = (float)std::log(2.0);
  const size_t cells_bytes = (size_t)T * block_n * 3;
  const size_t smem = cells_bytes + (4 - cells_bytes % 4) % 4 +
                      (fused ? (size_t)block_n * din * sizeof(float) + (size_t)2 * T * nap * sizeof(int) : 0);

#define LF(BN, U, PRO, L16)                                                                                        \
  do {                                                                                                             \
    constexpr int NTH = (BN) * (U);                                                                                \
    if (NTH > 1024) TORCH_CHECK(false, "block_n * units per row exceeds 1024 threads; use a smaller block_n");     \
    cudaFuncSetAttribute(p2_int8_kernel<BN, U, PRO, L16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
    dim3 grid((unsigned)((N + (BN) - 1) / (BN)), (unsigned)H);                                                     \
    p2_int8_kernel<BN, U, PRO, L16><<<grid, NTH, smem, stream>>>(                                                  \
        zp, ap, bp, cp, cop, wp, op, (int)N, (int)H, (int)T, (int)nap, (int)K, (int)din, (int)D, (int)stride, (int)lo,  \
        (int)hi, (int)Q, (float)inv, (float)g, (float)beta, (float)gamma, ln2, clo, chi);                          \
  } while (0)
#define PICKU(BN, U)                                                     \
  do {                                                                   \
    if (fused && load16) LF(BN, U, true, true);                          \
    else if (fused) LF(BN, U, true, false);                              \
    else if (load16) LF(BN, U, false, true);                             \
    else LF(BN, U, false, false);                                        \
  } while (0)
#define PICK(BN)                                                         \
  do {                                                                   \
    switch ((int)upr) {                                                  \
      case 1: PICKU(BN, 1); break;                                       \
      case 2: PICKU(BN, 2); break;                                       \
      case 3: PICKU(BN, 3); break;                                       \
      case 4: PICKU(BN, 4); break;                                       \
      case 5: PICKU(BN, 5); break;                                       \
      case 6: PICKU(BN, 6); break;                                       \
      case 7: PICKU(BN, 7); break;                                       \
      default: PICKU(BN, 8); break;                                      \
    }                                                                    \
  } while (0)

  switch ((int)block_n) {
    case 32: PICK(32); break;
    case 64: PICK(64); break;
    case 128: PICK(128); break;
    default: TORCH_CHECK(false, "block_n must be 32, 64 or 128");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::map<std::string, int64_t> shared_bytes(int64_t T, int64_t nap, int64_t din, int64_t block_n, bool fused) {
  const size_t cells_bytes = (size_t)T * block_n * 3;
  const size_t smem = cells_bytes + (4 - cells_bytes % 4) % 4 +
                      (fused ? (size_t)block_n * din * sizeof(float) + (size_t)2 * T * nap * sizeof(int) : 0);
  int dev = 0, maxsm = 0, reserved = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&maxsm, cudaDevAttrMaxSharedMemoryPerBlock, dev);
  cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
  return {{"smem", (int64_t)smem}, {"max_per_block", maxsm}, {"reserved_per_block", reserved}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read", &p2_read, "fused int8 power-of-two LUT read (cells or fused-prologue regime), generic cell width D");
  m.def("shared_bytes", &shared_bytes, "dynamic shared memory per block for a configuration");
}
