// int8 power-of-two LUT read for QuantisedLightFFN, and the forward of the p2_scalars custom op -- RTX 5090 (sm_120).
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
// One template, two instantiations (PROLOGUE):
//
//   PROLOGUE = true   (read_fused, the inference read) -- the kernel computes the per-table integers itself from the
//        compressed code Z, in the same launch, by calling p2::table_scalars (csrc/pow2_scalars.cuh): the same function
//        the spiky_lutorch::p2_scalars custom op runs (p2_scalar_kernel below) for the training forward, so training and
//        inference take identical integers by construction.
//
//   PROLOGUE = false  (read_cells, the test reference) -- the per-table integers are supplied by the caller, packed as
//        CELLS[N, H, T, 3] = (c1, c2, shifts) with shifts = sh1 | sh2 << 4; the kernel stages them into shared and
//        accumulates: bit-identical to pow2_read.int8_blend_read on those integers.
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
//   shared, phase 0 : the block's z rows and the head's anchors (read_fused), or its cells (read_cells)
//   shared, phase 1 : cells[t * BLOCK_N + lt] = (c1, c2, shifts)                      (read_fused computes them here)
//   registers        : the float scalars (tau, g, beta, gamma, read once from their one-element tensors); in phase 2 the 16 int32
//                     accumulators of the thread's unit and the current 16-byte row load
//   global           : int8 rows (one 16-byte int4 load per unit per cell; LOAD16=false: four char4 loads), float output

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cmath>

#include "pow2_scalars.cuh"

namespace {
constexpr int NAP_MAX = 8;      // c1 / c2 fit a byte
constexpr int UPR_MAX = 8;      // D <= 128
using p2::DISCARD;

template <int BLOCK_N, int UPR, bool PROLOGUE, bool LOAD16>
__global__ __launch_bounds__(BLOCK_N * UPR) void p2_int8_kernel(
    const float* __restrict__ Z, const int* __restrict__ Aa, const int* __restrict__ Ab,
    const uint8_t* __restrict__ CELLS, uint8_t* __restrict__ CELLS_OUT, const int8_t* __restrict__ W,
    float* __restrict__ OUT,
    int N, int H, int T, int nap, int K, int din, int D, int stride, int lo, int hi, int Q,
    const float* __restrict__ TAU, const float* __restrict__ G, const float* __restrict__ BETA,
    const float* __restrict__ GAMMA) {
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

  // ---- phase 1 (read_fused): the per-table integers, table-inner over the block's tokens
  if constexpr (PROLOGUE) {
    const float tau = *TAU, g = *G, beta = *BETA, gamma = *GAMMA;
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
      float dd[8];
      for (int p = 0; p < nap; ++p) dd[p] = zr[ap[p]] - zr[bp[p]];
      const p2::TableScalars r = p2::table_scalars(dd, nap, tau, g, beta, gamma, lo, hi, Q);
      csh[dst] = r.c1;
      csh[dst + 1] = r.c2;
      csh[dst + 2] = r.sh;
      if (CELLS_OUT != nullptr) {                                    // debug / test: expose the integers
        const size_t o = (((size_t)gt * H + h) * T + t) * 3;
        CELLS_OUT[o] = r.c1;
        CELLS_OUT[o + 1] = r.c2;
        CELLS_OUT[o + 2] = r.sh;
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

// Z [N, H, din] fp32 (read_fused) or empty; Aa, Ab [H, T, nap] int32 local column indices (read_fused) or empty; CELLS
// [N, H, T, 3] uint8 (read_cells input; read_fused: optional output) or empty; W [H * T * K, stride] int8 with stride = ceil(D / 16) * 16.
// Returns float32 [N, H, D] = the int32 accumulators, converted once.
torch::Tensor p2_read(const torch::Tensor& Z, const torch::Tensor& Aa, const torch::Tensor& Ab, const torch::Tensor& CELLS,
                      const torch::Tensor& W, int64_t N, int64_t H, int64_t T, int64_t nap, int64_t K, int64_t din,
                      int64_t D, int64_t lo, int64_t hi, int64_t Q, const torch::Tensor& TAU, const torch::Tensor& G,
                      const torch::Tensor& BETA, const torch::Tensor& GAMMA, int64_t block_n, bool fused, bool load16) {
  const int64_t stride = (D + 15) / 16 * 16;
  const int64_t upr = stride / 16;
  TORCH_CHECK(D >= 1 && upr <= UPR_MAX, "D must be in [1, ", 16 * UPR_MAX, "]");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kInt8 && W.is_contiguous() && W.dim() == 2 &&
                  W.size(1) == stride && W.size(0) == H * T * K,
              "W must be contiguous int8 CUDA [H*T*K, ceil(D/16)*16]");
  TORCH_CHECK(nap >= 1 && nap <= NAP_MAX && K == (1 << nap), "nap must be in [1, 8] with K = 2^nap");
  TORCH_CHECK(lo >= -3 && hi <= 4 && lo <= hi && Q >= 0 && Q <= 3, "window must lie in [-3, 4] and Q in [0, 3]");
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
  // read_fused: a CELLS tensor, when given, is an OUTPUT receiving the kernel's own per-table integers (tests)
  uint8_t* cop = (fused && CELLS.defined() && CELLS.numel() > 0) ? CELLS.data_ptr<uint8_t>() : nullptr;
  const int8_t* wp = W.data_ptr<int8_t>();
  float* op = out.data_ptr<float>();
  const float* taup = nullptr;
  const float* gp = nullptr;
  const float* betap = nullptr;
  const float* gammap = nullptr;
  if (fused) {
    for (const auto* t : {&TAU, &G, &BETA, &GAMMA})
      TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->numel() == 1 && t->is_contiguous(),
                  "tau, g, beta, gamma must be one-element contiguous fp32 CUDA tensors");
    taup = TAU.data_ptr<float>();
    gp = G.data_ptr<float>();
    betap = BETA.data_ptr<float>();
    gammap = GAMMA.data_ptr<float>();
  }
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
        (int)hi, (int)Q, taup, gp, betap, gammap);                          \
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


// ============================================================================================================================
// Forward of the spiky_lutorch::p2_scalars custom op: p2::table_scalars for every table, from margins D [B, T, nap] (B =
// tokens x heads, flattened). One launch, ONE TABLE PER THREAD (1024 per block), so the per-table math runs across the
// warp's lanes. Outputs (all [B, T, ...], global):
//   PSW   float [.., 2]  forward cell weights: skip ? 0 : 2^kc,  (skip || drop) ? 0 : 2^(kc - q)
//   CELLS uint8 [.., 3]  c1, c2, sh1 | sh2 << 4   (the int8 accumulation's input)
//   KQ    int8  [.., 2]  kc, q                     (the backward's STE ratios)
namespace {
constexpr int SCALAR_TB = 1024;

__global__ __launch_bounds__(SCALAR_TB) void p2_scalar_kernel(
    const float* __restrict__ Dm, float* __restrict__ PSW, uint8_t* __restrict__ CELLS, int8_t* __restrict__ KQ, int64_t BT,
    int nap, int lo, int hi, int Q, const float* __restrict__ TAU, const float* __restrict__ G, const float* __restrict__ BETA,
    const float* __restrict__ GAMMA) {
  const float tau = *TAU, g = *G, beta = *BETA, gamma = *GAMMA;
  const int64_t bt = (int64_t)blockIdx.x * SCALAR_TB + threadIdx.x;  // one table (token, head, table) per thread
  if (bt >= BT) return;
  const p2::TableScalars r = p2::table_scalars(Dm + bt * nap, nap, tau, g, beta, gamma, lo, hi, Q);
  PSW[bt * 2] = r.skip ? 0.f : std::ldexp(1.0f, r.kc);
  PSW[bt * 2 + 1] = (r.skip || r.drop) ? 0.f : std::ldexp(1.0f, (int)r.kc - (int)r.q);
  CELLS[bt * 3] = r.c1;
  CELLS[bt * 3 + 1] = r.c2;
  CELLS[bt * 3 + 2] = r.sh;
  KQ[bt * 2] = r.kc;
  KQ[bt * 2 + 1] = r.q;
}
}  // namespace

std::vector<torch::Tensor> p2_scalars(const torch::Tensor& Dm, const torch::Tensor& TAU, const torch::Tensor& G,
                                      const torch::Tensor& BETA, const torch::Tensor& GAMMA, int64_t lo, int64_t hi,
                                      int64_t Q) {
  TORCH_CHECK(Dm.is_cuda() && Dm.scalar_type() == torch::kFloat32 && Dm.dim() >= 2, "margins must be fp32 CUDA [..., T, nap]");
  const int64_t nap = Dm.size(-1);
  TORCH_CHECK(nap >= 1 && nap <= NAP_MAX, "nap must be in [1, 8]");
  TORCH_CHECK(lo >= -3 && hi <= 4 && lo <= hi && Q >= 0 && Q <= 3, "window must lie in [-3, 4] and Q in [0, 3]");
  for (const auto* t : {&TAU, &G, &BETA, &GAMMA})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->numel() == 1,
                "tau, g, beta, gamma must be one-element fp32 CUDA tensors");
  const auto D = Dm.contiguous();
  auto lead = D.sizes().vec();
  lead.pop_back();                                                   // [..., T]
  const int64_t BT = D.numel() / nap;
  auto fo = D.options();
  auto psw_shape = lead; psw_shape.push_back(2);
  auto cells_shape = lead; cells_shape.push_back(3);
  auto psw = torch::empty(psw_shape, fo);
  auto cells = torch::empty(cells_shape, fo.dtype(torch::kUInt8));
  auto kq = torch::empty(psw_shape, fo.dtype(torch::kInt8));
  if (BT > 0) {
    const auto tau = TAU.contiguous(), g = G.contiguous(), beta = BETA.contiguous(), gamma = GAMMA.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)((BT + SCALAR_TB - 1) / SCALAR_TB));
    p2_scalar_kernel<<<grid, SCALAR_TB, 0, stream>>>(
        D.data_ptr<float>(), psw.data_ptr<float>(), cells.data_ptr<uint8_t>(), kq.data_ptr<int8_t>(), BT, (int)nap, (int)lo,
        (int)hi, (int)Q, tau.data_ptr<float>(), g.data_ptr<float>(), beta.data_ptr<float>(), gamma.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {psw, cells, kq};
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
  m.def("read", &p2_read, "int8 power-of-two LUT read, generic cell width D: fused (integers in-kernel) or on supplied cells (reference)");
  m.def("shared_bytes", &shared_bytes, "dynamic shared memory per block for a configuration");
  m.def("scalars", &p2_scalars, "p2::table_scalars for every table: forward of the spiky_lutorch::p2_scalars op");
}
