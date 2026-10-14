// Forward of the lutorch_ex::p2_scalars custom op: p2::table_scalars (csrc/pow2_scalars.cuh) for every table of a batch of
// margins. The op supplies the per-table integers (c1, c2, shift codes, k', q) and the straight-through cell weights of the
// power-of-two read; lutorch_ex accumulates the int8 rows with its torch read (pow2_read.int_blend_read).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "pow2_scalars.cuh"

namespace {
constexpr int NAP_MAX = 8;      // c1 / c2 fit a byte
}  // namespace

// ============================================================================================================================
// Forward of the lutorch_ex::p2_scalars custom op: p2::table_scalars for every table, from margins D [B, T, nap] (B =
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

// ============================================================================================================================
// FusedQuantisedConfidenceLUT's forward: three launches per call (the exponents take two, see te_partial_kernel).
//
// 1. table_exponents: e[g, c] = ceil(log2(max_{t, cell} |W[g, t, cell, c]|)) - (bits - 1) + offset, the per-(group,
//    channel) power-of-two scale of the fake-quant table (_pow2.head_chan_exponents). Input-independent; a max, so its
//    result does not depend on the reduction order: a row-split partial max, then the max over the splits.
// 2. quant_fwd: the whole read, one CTA per (group g, QF_ROWS consecutive samples), structured as FusedConfidenceLUT's
//    forward (csrc/fused_confidence.cu). The read width N (1 or 2, read_top_n) is a template parameter, as in
//    deploy_fwd_kernel: the N = 1 instantiation has no second cell, no blend and no tau anywhere (if constexpr).
//    phase A  thread per table: the margins d_j = z[a_j] - z[b_j] from the z row in shared memory, then, N = 2,
//             p2::table_scalars (the same code as p2_scalar_kernel: c1, c2, k', q, skip, drop) and the cell weights 2^k' /
//             2^(k'-q), or, N = 1, the address c1 (bits d_j > 0) and p2::single_scalars (k', skip) and the one weight
//             2^k' -- times the table-dropout factor, all in registers. The margins never reach global memory.
//    phase B  stripe x chunk: nstripe = QF threads / (d_out / VEC) stripes; stripe s accumulates tables t = s, s +
//             nstripe, ... (increasing), each table adding w1 * fq(W[c1]) then (N = 2) w2 * fq(W[c2]), where fq is the fake-quant
//             value clamp(floor(W / 2^e + 1/2), qlo, qhi) * 2^e computed in registers from the float32 master row
// (__ldg,
//             VEC floats per load). Every add is a separate float32 add (built with --fmad=false: no contraction), and
// with
//             no dropout each product is exact (a power of two times a power-of-two-scaled integer).
//    phase C  y = ((0 + p_0) + p_1) + ... + p_{nstripe-1} over the stripe partials, in that order, stored once.
//    THE REDUCTION ORDER IS FIXED: QF_THREADS and the vector width (4, else 2, else 1: the widest dividing d_out) are
//    constants of this kernel, not launch knobs, so the order the eager reference has to mirror never changes.
namespace qf {
constexpr int NAP_MAX = 8;
constexpr int QF_THREADS = 64;   // raised to cover one table row when d_out / VEC > 64
constexpr int QF_ROWS = 4;       // samples per CTA

struct Params {
  const float* z;           // [B, G, d_in]
  const int16_t* anc_a;     // [G, tph, nap]
  const int16_t* anc_b;     // [G, tph, nap]
  const float* W;           // [G * tph * K, d_out] float32 master table
  const float* E;           // [G, d_out] the exponents e (integers, as float)
  const float* dmask;       // [B, G, tph] table-dropout factors, nullptr = none
  const float* log_beta;
  const float* log_gamma;
  const float* log_tau;     // the blend temperature: N = 2 only (nullptr at N = 1, which has no blend)
  int32_t* tabs;            // [B, G, tph] each table's packed integers (pack_tab): written by the forward when the
                            // backward will run, read by the backward; nullptr in a forward without autograd
  int B, G, tph, nap, d_in, d_out, lo, hi, Q;
  float qlo, qhi;
};

// One table's integers for the backward, packed into 28 bits of an int32: c1 | c2 << 8 | (k' + 8) << 16 | q << 20 |
// skip << 27. c1, c2 < 2^nap <= 256 (8 bits each); the clamped k' lies in [lo, hi] within [-3, 4] (the host checks the
// window), so k' + 8 is in [5, 12] (4 bits); q is clamped to [0, 64] (7 bits); skip is one bit; drop is q > Q, recomputed.
// Exactly the integers p2::table_scalars returned in the forward, so the backward reads what it used to recompute.
// N = 1 uses the same layout with c2 = q = 0 (p2::single_scalars has neither): c1 < 2^nap, k' + 8 in [5, 12], skip.
__device__ inline int32_t pack_tab(const p2::TableScalars& s) {
  return (int32_t)s.c1 | ((int32_t)s.c2 << 8) | (((int32_t)s.kc + 8) << 16) | ((int32_t)s.q << 20) |
         ((int32_t)s.skip << 27);
}

__device__ inline int32_t pack_tab1(int c1, const p2::SingleScalars& s) {
  return (int32_t)c1 | (((int32_t)s.kc + 8) << 16) | ((int32_t)s.skip << 27);
}

__device__ inline p2::TableScalars unpack_tab(int32_t v, int Q) {
  p2::TableScalars s;
  s.c1 = (uint8_t)(v & 255);
  s.c2 = (uint8_t)((v >> 8) & 255);
  s.kc = (int8_t)(((v >> 16) & 15) - 8);
  s.q = (int8_t)((v >> 20) & 127);
  s.skip = ((v >> 27) & 1) != 0;
  s.drop = s.q > Q;
  s.sh = 0;
  s.mj = 0;
  return s;
}

__host__ __device__ inline size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

struct Smem {
  size_t zs, w1, w2, cs, cas, part, sc, si, sa, sb, total;
  __host__ __device__ Smem(int tph, int nap, int d_in, int d_out, int nstripe, int n) {
    size_t o = 0;
    zs = o;   o = align16(o + sizeof(float) * d_in);
    w1 = o;   o = align16(o + sizeof(float) * tph);
    w2 = o;   o = align16(o + sizeof(float) * tph * (n - 1));     // second cell: N = 2 only
    cs = o;   o = align16(o + sizeof(int) * tph);
    cas = o;  o = align16(o + sizeof(int) * tph * (n - 1));
    part = o; o = align16(o + sizeof(float) * nstripe * d_out);
    sc = o;   o = align16(o + sizeof(float) * d_out);
    si = o;   o = align16(o + sizeof(float) * d_out);
    sa = o;   o = align16(o + sizeof(int16_t) * tph * nap);
    sb = o;   o = align16(o + sizeof(int16_t) * tph * nap);
    total = o;
  }
};

template <int VEC> struct VecT;
template <> struct VecT<1> { using T = float; };
template <> struct VecT<2> { using T = float2; };
template <> struct VecT<4> { using T = float4; };

template <int VEC>
__device__ inline void load_vec(const float* p, float (&v)[VEC]) {
  typename VecT<VEC>::T x = __ldg(reinterpret_cast<const typename VecT<VEC>::T*>(p));
  const float* f = reinterpret_cast<const float*>(&x);
#pragma unroll
  for (int k = 0; k < VEC; ++k) v[k] = f[k];
}

// table_exponents in two launches (a max over all of a group's rows, per channel; the 50 MB champion table is read once):
//   te_partial_kernel  one CTA per (group, row split): TE_THREADS-ish threads as TX columns x TY rows, so consecutive
//                      threads read consecutive floats of consecutive rows (a fully coalesced sweep of the split's rows);
//                      the per-channel max of the split goes to P[g, split, c]. Splits are sized so the grid has
//                      ~G x rows x d_out / TE_ELEMS_PER_CTA CTAs (3,072 at the champion geometry), not G x d_out / 32.
//   te_final_kernel    one thread per (group, channel): the max over the splits, then e.
// A max is exact and order-independent, so the result equals any other decomposition bit for bit; no atomics.
constexpr int TE_THREADS = 256;                // threads per CTA (rounded down to a multiple of TX)
constexpr int TE_ELEMS_PER_CTA = 4096;         // floats each CTA reads (16 KB): sets the number of row splits

__global__ void te_partial_kernel(const float* __restrict__ W, float* __restrict__ P, int rows_per_group, int d_out,
                                  int nsplit, int rows_per_split, int TX, int TY) {
  extern __shared__ float red[];               // [TY][TX]
  const int b = blockIdx.x;
  const int g = b / nsplit, split = b % nsplit;
  const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  const int r0 = split * rows_per_split;
  const int r1 = min(rows_per_group, r0 + rows_per_split);
  const float* base = W + (size_t)g * rows_per_group * d_out;
  for (int c0 = 0; c0 < d_out; c0 += TX) {
    const int c = c0 + tx;
    float m = 0.f;
    if (c < d_out)
      for (int r = r0 + ty; r < r1; r += TY) m = fmaxf(m, fabsf(__ldg(base + (size_t)r * d_out + c)));
    red[ty * TX + tx] = m;
    __syncthreads();
    if (ty == 0 && c < d_out) {
      float a = red[tx];
      for (int y = 1; y < TY; ++y) a = fmaxf(a, red[y * TX + tx]);
      P[((size_t)g * nsplit + split) * d_out + c] = a;
    }
    __syncthreads();
  }
}

__global__ void te_final_kernel(const float* __restrict__ P, float* __restrict__ E, int G, int d_out, int nsplit,
                                float shift) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;   // (g, c)
  if (i >= (int64_t)G * d_out) return;
  const int64_t g = i / d_out, c = i % d_out;
  float a = 0.f;
  for (int s = 0; s < nsplit; ++s) a = fmaxf(a, P[(g * nsplit + s) * d_out + c]);
  E[i] = ceilf(log2f(fmaxf(a, 1e-30f))) - shift;   // _pow2.head_chan_exponents
}

template <int VEC, int N>
__global__ void quant_fwd_kernel(Params p, float* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int nchunk = p.d_out / VEC;
  const int nstripe = blockDim.x / nchunk;
  const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, N);
  float* zs = reinterpret_cast<float*>(smem_raw + L.zs);
  float* w1 = reinterpret_cast<float*>(smem_raw + L.w1);
  float* w2 = reinterpret_cast<float*>(smem_raw + L.w2);
  int* cs = reinterpret_cast<int*>(smem_raw + L.cs);
  int* cas = reinterpret_cast<int*>(smem_raw + L.cas);
  float* part = reinterpret_cast<float*>(smem_raw + L.part);
  float* sc = reinterpret_cast<float*>(smem_raw + L.sc);
  float* si = reinterpret_cast<float*>(smem_raw + L.si);
  int16_t* sa = reinterpret_cast<int16_t*>(smem_raw + L.sa);
  int16_t* sb = reinterpret_cast<int16_t*>(smem_raw + L.sb);

  const int tid = threadIdx.x, bd = blockDim.x;
  const int g = blockIdx.x % p.G;
  const int b0 = static_cast<int>((static_cast<int64_t>(blockIdx.x) / p.G) * QF_ROWS);
  const int an = p.tph * p.nap;
  for (int i = tid; i < an; i += bd) {
    sa[i] = p.anc_a[(size_t)g * an + i];
    sb[i] = p.anc_b[(size_t)g * an + i];
  }
  for (int i = tid; i < p.d_out; i += bd) {                // this group's fake-quant scale 2^e and its inverse
    const int e = (int)p.E[(size_t)g * p.d_out + i];
    sc[i] = ldexpf(1.0f, e);
    si[i] = ldexpf(1.0f, -e);
  }
  const float beta = expf(*p.log_beta), gamma = expf(*p.log_gamma);
  const float tau = N == 2 ? expf(*p.log_tau) : 1.f;        // no blend at N = 1: tau is never read
  const int stripe = tid / nchunk, chunk = tid % nchunk;

  for (int r = 0; r < QF_ROWS; ++r) {
    const int b = b0 + r;
    if (b >= p.B) break;                                    // uniform across the CTA
    const size_t row = (size_t)b * p.G + g;
    for (int i = tid; i < p.d_in; i += bd) zs[i] = p.z[row * p.d_in + i];
    __syncthreads();
    for (int t = tid; t < p.tph; t += bd) {                 // phase A
      float d[NAP_MAX];
#pragma unroll
      for (int j = 0; j < NAP_MAX; ++j)
        if (j < p.nap) d[j] = zs[sa[t * p.nap + j]] - zs[sb[t * p.nap + j]];
      const int base = (g * p.tph + t) << p.nap;
      const float k = p.dmask ? p.dmask[row * p.tph + t] : 1.f;
      if constexpr (N == 2) {
        const p2::TableScalars s = p2::table_scalars(d, p.nap, tau, 0.f, beta, gamma, p.lo, p.hi, p.Q);
        if (p.tabs) p.tabs[row * p.tph + t] = pack_tab(s);
        float a1 = s.skip ? 0.f : ldexpf(1.0f, s.kc);
        float a2 = (s.skip || s.drop) ? 0.f : ldexpf(1.0f, (int)s.kc - (int)s.q);
        if (p.dmask) {
          a1 = a1 * k;
          a2 = a2 * k;
        }
        cs[t] = base + s.c1;
        cas[t] = base + s.c2;
        w1[t] = a1;
        w2[t] = a2;
      } else {
        int c1 = 0;
#pragma unroll
        for (int j = 0; j < NAP_MAX; ++j)
          if (j < p.nap && d[j] > 0.f) c1 |= 1 << (p.nap - 1 - j);
        const p2::SingleScalars s = p2::single_scalars(d, p.nap, 0.f, beta, gamma, p.lo, p.hi);
        if (p.tabs) p.tabs[row * p.tph + t] = pack_tab1(c1, s);
        float a1 = s.skip ? 0.f : ldexpf(1.0f, s.kc);
        if (p.dmask) a1 = a1 * k;
        cs[t] = base + c1;
        w1[t] = a1;
      }
    }
    __syncthreads();
    if (stripe < nstripe) {                                 // phase B
      float acc[VEC];
#pragma unroll
      for (int k = 0; k < VEC; ++k) acc[k] = 0.f;
      float scl[VEC], inv[VEC];
#pragma unroll
      for (int k = 0; k < VEC; ++k) { scl[k] = sc[chunk * VEC + k]; inv[k] = si[chunk * VEC + k]; }
      for (int t = stripe; t < p.tph; t += nstripe) {
        float v[VEC];
        load_vec<VEC>(p.W + (size_t)cs[t] * p.d_out + chunk * VEC, v);
        const float a = w1[t];
#pragma unroll
        for (int k = 0; k < VEC; ++k)
          acc[k] = acc[k] + a * (fminf(fmaxf(floorf(v[k] * inv[k] + 0.5f), p.qlo), p.qhi) * scl[k]);
        if constexpr (N == 2) {
          load_vec<VEC>(p.W + (size_t)cas[t] * p.d_out + chunk * VEC, v);
          const float a2 = w2[t];
#pragma unroll
          for (int k = 0; k < VEC; ++k)
            acc[k] = acc[k] + a2 * (fminf(fmaxf(floorf(v[k] * inv[k] + 0.5f), p.qlo), p.qhi) * scl[k]);
        }
      }
#pragma unroll
      for (int k = 0; k < VEC; ++k) part[stripe * p.d_out + chunk * VEC + k] = acc[k];
    }
    __syncthreads();
    for (int i = tid; i < p.d_out; i += bd) {               // phase C
      float y = 0.f;
      for (int s = 0; s < nstripe; ++s) y = y + part[s * p.d_out + i];
      out[row * p.d_out + i] = y;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// quant_bwd: the backward of quant_fwd, structured as FusedConfidenceLUT's backward, templated on the read width N as
// quant_fwd is (N = 1: one cell, no blend, no tau, no argmin term; its smem holds no second-cell arrays). Saved from the
// forward: z, the parameters, the exponents E, the dropout factors and each table's integers (tabs, pack_tab /
// pack_tab1): per (group, row) the CTA re-stages z for the margins and reads the integers back. At N = 1 the
// straight-through weight is psw = val + s (b / s)|detached (QuantisedConfidenceLUT._ste_score_weight): G = r * mask *
// b / max(s, 1e-30), g_s = G, and only the score's terms (S, ls) reach the margins and beta / gamma.
//   phase A  thread per table: margins (kept in shared memory, tph x nap per CTA, for phase C), the forward's integers
//            read back from tabs (one packed int32 per table, pack_tab; not recomputed), the forward cell weights w1 / w2
//            (x dropout) and the straight-through numerators b1 = 2^k', b2 = 2^(k'-q) (clamped k', also for skipped /
//            dropped cells, as _pow2.ste_blend_weights).
//   phase B  stripe x chunk: gW[c] += w go (float32 atomics, vector atomics on sm_90+: the ONLY atomics; gW feeds no
//            exponent) and the partial dot products go . fq(W[c]) of this chunk, each written to its own slot
//            rp[chunk][t] (no atomics).
//   phase C  thread per table: r = sum over chunks 0..nchunk-1 of rp[chunk][t] (in that order); then the gradient of the
//            straight-through weights psw_c = val_c + e_c (b_c / e_c)|detached with e_1 = s sigmoid(x), e_2 = s sigmoid(-x),
//            x = 2 mv / tau, s = S exp(gamma ls): G_c = r_c * mask * b_c / max(e_c, 1e-30); g_s = G_1 sig(x) + G_2 sig(-x);
//            g_x = s (G_1 - G_2) sig(x) sig(-x); then the margins' gradients (S, ls, the argmin's mv) into gd[t][j], and
//            this thread's share of the beta / gamma / tau gradients (tables in increasing t).
//   phase D  thread per input dim i: gz[i] = sum over the anchors that read i, in a fixed (anchor-sorted) order, of +gd
//            (an a-anchor) / -gd (a b-anchor) -- no atomics.
//   scalars  per-CTA partials: warp xor-shuffle tree, then the warps in order; summed over CTAs by the caller.
__device__ inline float qf_block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float s = 0.f;
  if (threadIdx.x == 0) for (int w = 0; w < nw; ++w) s += red[w];
  return s;                                                 // valid in thread 0
}

template <int VEC>
__device__ inline void atomic_add_vec(float* dst, const float (&v)[VEC], bool vec_atomics) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (vec_atomics) {
    if constexpr (VEC == 4) { atomicAdd(reinterpret_cast<float4*>(dst), make_float4(v[0], v[1], v[2], v[3])); return; }
    if constexpr (VEC == 2) { atomicAdd(reinterpret_cast<float2*>(dst), make_float2(v[0], v[1])); return; }
  }
#endif
#pragma unroll
  for (int k = 0; k < VEC; ++k) atomicAdd(dst + k, v[k]);
}

struct SmemB {
  size_t zs, gos, w1, w2, b1, b2, cs, cas, us, gd, rp1, rp2, sc, si, sa, sb, total;
  __host__ __device__ SmemB(int tph, int nap, int d_in, int d_out, int nchunk, int n) {
    size_t o = 0;
    zs = o;   o = align16(o + sizeof(float) * d_in);
    gos = o;  o = align16(o + sizeof(float) * d_out);
    w1 = o;   o = align16(o + sizeof(float) * tph);
    w2 = o;   o = align16(o + sizeof(float) * tph * (n - 1));     // second cell: N = 2 only
    b1 = o;   o = align16(o + sizeof(float) * tph);
    b2 = o;   o = align16(o + sizeof(float) * tph * (n - 1));
    cs = o;   o = align16(o + sizeof(int) * tph);
    cas = o;  o = align16(o + sizeof(int) * tph * (n - 1));
    us = o;   o = align16(o + sizeof(float) * tph * nap);
    gd = o;   o = align16(o + sizeof(float) * tph * nap);
    rp1 = o;  o = align16(o + sizeof(float) * nchunk * tph);
    rp2 = o;  o = align16(o + sizeof(float) * nchunk * tph * (n - 1));
    sc = o;   o = align16(o + sizeof(float) * d_out);
    si = o;   o = align16(o + sizeof(float) * d_out);
    sa = o;   o = align16(o + sizeof(int16_t) * tph * nap);
    sb = o;   o = align16(o + sizeof(int16_t) * tph * nap);
    total = o;
  }
};

template <int VEC, int N>
__global__ void quant_bwd_kernel(Params p, const float* __restrict__ go, const int* __restrict__ zoff,
                                 const int* __restrict__ zent, float* __restrict__ gW, float* __restrict__ gz,
                                 float* __restrict__ gscal, bool vec_atomics) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ float red[32];
  const int nchunk = p.d_out / VEC;
  const int nstripe = blockDim.x / nchunk;
  const SmemB L(p.tph, p.nap, p.d_in, p.d_out, nchunk, N);
  float* zs = reinterpret_cast<float*>(smem_raw + L.zs);
  float* gos = reinterpret_cast<float*>(smem_raw + L.gos);
  float* w1 = reinterpret_cast<float*>(smem_raw + L.w1);
  float* w2 = reinterpret_cast<float*>(smem_raw + L.w2);
  float* b1 = reinterpret_cast<float*>(smem_raw + L.b1);
  float* b2 = reinterpret_cast<float*>(smem_raw + L.b2);
  int* cs = reinterpret_cast<int*>(smem_raw + L.cs);
  int* cas = reinterpret_cast<int*>(smem_raw + L.cas);
  float* us = reinterpret_cast<float*>(smem_raw + L.us);
  float* gd = reinterpret_cast<float*>(smem_raw + L.gd);
  float* rp1 = reinterpret_cast<float*>(smem_raw + L.rp1);
  float* rp2 = reinterpret_cast<float*>(smem_raw + L.rp2);
  float* sc = reinterpret_cast<float*>(smem_raw + L.sc);
  float* si = reinterpret_cast<float*>(smem_raw + L.si);
  int16_t* sa = reinterpret_cast<int16_t*>(smem_raw + L.sa);
  int16_t* sb = reinterpret_cast<int16_t*>(smem_raw + L.sb);

  const int tid = threadIdx.x, bd = blockDim.x;
  const int g = blockIdx.x % p.G;
  const int b0 = static_cast<int>((static_cast<int64_t>(blockIdx.x) / p.G) * QF_ROWS);
  const int nap = p.nap, an = p.tph * nap;
  for (int i = tid; i < an; i += bd) {
    sa[i] = p.anc_a[(size_t)g * an + i];
    sb[i] = p.anc_b[(size_t)g * an + i];
  }
  for (int i = tid; i < p.d_out; i += bd) {
    const int e = (int)p.E[(size_t)g * p.d_out + i];
    sc[i] = ldexpf(1.0f, e);
    si[i] = ldexpf(1.0f, -e);
  }
  const float beta = expf(*p.log_beta), gamma = expf(*p.log_gamma);
  const float tau = N == 2 ? expf(*p.log_tau) : 1.f;        // no blend at N = 1: tau is never read
  const int stripe = tid / nchunk, chunk = tid % nchunk;
  const int* off = zoff + (size_t)g * (p.d_in + 1);
  const int* ent = zent + (size_t)g * 2 * an;
  float g_lbeta = 0.f, g_lgamma = 0.f, g_ltau = 0.f;

  for (int r = 0; r < QF_ROWS; ++r) {
    const int b = b0 + r;
    if (b >= p.B) break;
    const size_t row = (size_t)b * p.G + g;
    for (int i = tid; i < p.d_in; i += bd) zs[i] = p.z[row * p.d_in + i];
    for (int i = tid; i < p.d_out; i += bd) gos[i] = go[row * p.d_out + i];
    __syncthreads();
    for (int t = tid; t < p.tph; t += bd) {                 // phase A
      for (int j = 0; j < nap; ++j) us[t * nap + j] = zs[sa[t * nap + j]] - zs[sb[t * nap + j]];
      const p2::TableScalars s = unpack_tab(p.tabs[row * p.tph + t], p.Q);   // the forward's integers
      const float k = p.dmask ? p.dmask[row * p.tph + t] : 1.f;
      const int base = (g * p.tph + t) << nap;
      w1[t] = (s.skip ? 0.f : ldexpf(1.0f, s.kc)) * k;
      b1[t] = ldexpf(1.0f, s.kc);
      cs[t] = base + s.c1;
      if constexpr (N == 2) {
        w2[t] = ((s.skip || s.drop) ? 0.f : ldexpf(1.0f, (int)s.kc - (int)s.q)) * k;
        b2[t] = ldexpf(1.0f, (int)s.kc - (int)s.q);
        cas[t] = base + s.c2;
      }
    }
    __syncthreads();
    if (stripe < nstripe) {                                 // phase B
      float gv[VEC], scl[VEC], inv[VEC];
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        gv[k] = gos[chunk * VEC + k];
        scl[k] = sc[chunk * VEC + k];
        inv[k] = si[chunk * VEC + k];
      }
      for (int t = stripe; t < p.tph; t += nstripe) {
        float v[VEC], dw[VEC];
        const size_t o1 = (size_t)cs[t] * p.d_out + chunk * VEC;
        load_vec<VEC>(p.W + o1, v);
        float dot = 0.f;
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          dot = dot + (fminf(fmaxf(floorf(v[k] * inv[k] + 0.5f), p.qlo), p.qhi) * scl[k]) * gv[k];
          dw[k] = w1[t] * gv[k];
        }
        if (w1[t] != 0.f) atomic_add_vec<VEC>(gW + o1, dw, vec_atomics);
        rp1[chunk * p.tph + t] = dot;
        if constexpr (N == 2) {
          const size_t o2 = (size_t)cas[t] * p.d_out + chunk * VEC;
          load_vec<VEC>(p.W + o2, v);
          float dot2 = 0.f;
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            dot2 = dot2 + (fminf(fmaxf(floorf(v[k] * inv[k] + 0.5f), p.qlo), p.qhi) * scl[k]) * gv[k];
            dw[k] = w2[t] * gv[k];
          }
          if (w2[t] != 0.f) atomic_add_vec<VEC>(gW + o2, dw, vec_atomics);
          rp2[chunk * p.tph + t] = dot2;
        }
      }
    }
    __syncthreads();
    for (int t = tid; t < p.tph; t += bd) {                 // phase C
      float r1 = 0.f, r2 = 0.f;
      for (int c = 0; c < nchunk; ++c) {
        r1 = r1 + rp1[c * p.tph + t];
        if constexpr (N == 2) r2 = r2 + rp2[c * p.tph + t];
      }
      float M = 0.f, Ls = 0.f, mstar = 0.f, sig_m = 0.f;
      int jstar = 0;
#pragma unroll
      for (int j = 0; j < NAP_MAX; ++j)
        if (j < nap) {
          const float m = fabsf(us[t * nap + j]);
          M = M + m;
          const float x = beta * m;
          Ls = Ls + (fminf(0.f, x) - log1pf(expf(-fabsf(x))));
          sig_m = sig_m + m / (1.f + expf(x));               // m sigmoid(-beta m)
          if constexpr (N == 2)
            if (j == 0 || m < mstar) { mstar = m; jstar = j; } // first argmin, as torch.min
        }
      const float E = expf(gamma * Ls);
      const float s = M * E;
      const float k = p.dmask ? p.dmask[row * p.tph + t] : 1.f;
      float g_s, g_mv = 0.f;
      if constexpr (N == 2) {
        const float x = 2.0f * mstar / tau;
        const float sp = 1.f / (1.f + expf(-x)), sn = 1.f / (1.f + expf(x));
        const float G1 = r1 * k * (b1[t] / fmaxf(s * sp, 1e-30f));
        const float G2 = r2 * k * (b2[t] / fmaxf(s * sn, 1e-30f));
        g_s = G1 * sp + G2 * sn;
        const float g_x = s * (G1 - G2) * sp * sn;
        g_ltau += -g_x * x;
        g_mv = g_x * 2.0f / tau;
      } else {
        g_s = r1 * k * (b1[t] / fmaxf(s, 1e-30f));           // the one cell: e = s, no blend
      }
      const float g_ls = g_s * s * gamma;
      g_lbeta += g_ls * beta * sig_m;
      g_lgamma += g_ls * Ls;
#pragma unroll
      for (int j = 0; j < NAP_MAX; ++j)
        if (j < nap) {
          const float u = us[t * nap + j];
          const float m = fabsf(u);
          float gm = g_s * E + g_ls * beta / (1.f + expf(beta * m));
          if constexpr (N == 2)
            if (j == jstar) gm += g_mv;
          gd[t * nap + j] = gm * float((u > 0.f) - (u < 0.f));   // d|u|/du = sign(u), 0 at 0
        }
    }
    __syncthreads();
    for (int i = tid; i < p.d_in; i += bd) {                // phase D
      float acc = 0.f;
      for (int e = off[i]; e < off[i + 1]; ++e) {
        const int idx = ent[e];
        acc = idx < an ? acc + gd[idx] : acc - gd[idx - an];
      }
      gz[row * p.d_in + i] = acc;
    }
    __syncthreads();
  }
  const float s0 = qf_block_sum(g_lbeta, red);
  const float s1 = qf_block_sum(g_lgamma, red);
  const float s2 = qf_block_sum(g_ltau, red);
  if (tid == 0) {
    gscal[static_cast<int64_t>(blockIdx.x) * 3 + 0] = s0;
    gscal[static_cast<int64_t>(blockIdx.x) * 3 + 1] = s1;
    gscal[static_cast<int64_t>(blockIdx.x) * 3 + 2] = s2;
  }
}

int vec_for(int d_out) { return d_out % 4 == 0 ? 4 : (d_out % 2 == 0 ? 2 : 1); }

int threads_for(int d_out) {
  const int need = ((d_out / vec_for(d_out) + 31) / 32) * 32;
  return need > QF_THREADS ? need : QF_THREADS;
}
}  // namespace qf

// ============================================================================================================================
// DeployedQuantisedConfidenceLUT's forward: deploy_fwd, the Stage 1 template specialised for the stored int8 rows and for
// the read width N (1 or 2, a template parameter: the export's read_top_n). One CTA per (group g, QF_ROWS samples).
//   phase A  thread per table: the margins (z[a] - z[b], or z[a] for single anchors), the address c1 (bits d_p > 0),
//            then p2::table_scalars (N = 2: k', q, skip, drop, the argmin mj, c2 = c1 ^ 2^(nap-1-mj)) or p2::single_scalars
//            (N = 1: k', skip), giving each cell's left shift k' + 6 (and k' + 6 - q), or -1 for a cell that is not read.
//   phase B  stripe x chunk: VEC int8 bytes per load (8, 4, 2 or 1: the widest dividing d_out; 8 measured faster than 16
//            at the champion, which leaves too few chunks per row and too many stripes), widened to int32 in
//            registers and shifted; a cell that is not read is not loaded. int32 accumulation: exact, so the value does not
//            depend on the order (bounded by 2 tph 127 2^10 < 2^31).
//   phase C  the stripe partials in order, then ONE conversion int32 -> float32 (round to nearest, as torch's
// .to(float32)):
//            the RAW accumulator in units of 2^-6 (the scale 2^(e - 6) is folded into the decompress weight at export).
//   No backward (inference only). The margins never reach global memory.
namespace dq {
constexpr int DQ_THREADS = 64;

struct Params {
  const float* z;           // [B, G, d_in]
  const int16_t* anc_a;     // [G, tph, nap]
  const int16_t* anc_b;     // [G, tph, nap], nullptr = single anchors
  const int8_t* P;          // [G * tph * K, d_out] the packed int8 table
  const float* log_beta;
  const float* log_gamma;
  const float* log_tau;
  int B, G, tph, nap, d_in, d_out, lo, hi, Q;
};

template <int VEC> struct Bytes;
template <> struct Bytes<1> { using T = int8_t; };
template <> struct Bytes<2> { using T = int16_t; };
template <> struct Bytes<4> { using T = int32_t; };
template <> struct Bytes<8> { using T = int2; };

template <int VEC>
__device__ inline void load_bytes(const int8_t* p, int (&v)[VEC]) {
  typename Bytes<VEC>::T x = __ldg(reinterpret_cast<const typename Bytes<VEC>::T*>(p));
  const int8_t* b = reinterpret_cast<const int8_t*>(&x);
#pragma unroll
  for (int k = 0; k < VEC; ++k) v[k] = b[k];
}

struct Smem {
  size_t zs, sh, cs, part, sa, sb, total;
  __host__ __device__ Smem(int tph, int nap, int d_in, int d_out, int nstripe, int n) {
    size_t o = 0;
    zs = o;   o = qf::align16(o + sizeof(float) * d_in);
    sh = o;   o = qf::align16(o + sizeof(int) * tph * n);
    cs = o;   o = qf::align16(o + sizeof(int) * tph * n);
    part = o; o = qf::align16(o + sizeof(int) * nstripe * d_out);
    sa = o;   o = qf::align16(o + sizeof(int16_t) * tph * nap);
    sb = o;   o = qf::align16(o + sizeof(int16_t) * tph * nap);
    total = o;
  }
};

template <int VEC, int N>
__global__ void deploy_fwd_kernel(Params p, float* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int nchunk = p.d_out / VEC;
  const int nstripe = blockDim.x / nchunk;
  const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, N);
  float* zs = reinterpret_cast<float*>(smem_raw + L.zs);
  int* sh = reinterpret_cast<int*>(smem_raw + L.sh);        // [tph][N] shift, -1 = not read
  int* cs = reinterpret_cast<int*>(smem_raw + L.cs);        // [tph][N] cell (table offset included)
  int* part = reinterpret_cast<int*>(smem_raw + L.part);
  int16_t* sa = reinterpret_cast<int16_t*>(smem_raw + L.sa);
  int16_t* sb = reinterpret_cast<int16_t*>(smem_raw + L.sb);

  const int tid = threadIdx.x, bd = blockDim.x;
  const int g = blockIdx.x % p.G;
  const int b0 = static_cast<int>((static_cast<int64_t>(blockIdx.x) / p.G) * qf::QF_ROWS);
  const int nap = p.nap, an = p.tph * nap;
  const bool single = p.anc_b == nullptr;
  for (int i = tid; i < an; i += bd) {
    sa[i] = p.anc_a[(size_t)g * an + i];
    if (!single) sb[i] = p.anc_b[(size_t)g * an + i];
  }
  const float beta = expf(*p.log_beta), gamma = expf(*p.log_gamma);
  const float tau = N == 2 ? expf(*p.log_tau) : 1.f;
  const int stripe = tid / nchunk, chunk = tid % nchunk;

  for (int r = 0; r < qf::QF_ROWS; ++r) {
    const int b = b0 + r;
    if (b >= p.B) break;                                    // uniform across the CTA
    const size_t row = (size_t)b * p.G + g;
    for (int i = tid; i < p.d_in; i += bd) zs[i] = p.z[row * p.d_in + i];
    __syncthreads();
    for (int t = tid; t < p.tph; t += bd) {                 // phase A
      float d[qf::NAP_MAX];
      int c1 = 0;
#pragma unroll
      for (int j = 0; j < qf::NAP_MAX; ++j)
        if (j < nap) {
          d[j] = single ? zs[sa[t * nap + j]] : zs[sa[t * nap + j]] - zs[sb[t * nap + j]];
          if (d[j] > 0.f) c1 |= 1 << (nap - 1 - j);
        }
      const int base = (g * p.tph + t) << nap;
      if constexpr (N == 2) {
        const p2::TableScalars s = p2::table_scalars(d, nap, tau, 0.f, beta, gamma, p.lo, p.hi, p.Q);
        cs[t * 2] = base + c1;
        cs[t * 2 + 1] = base + (c1 ^ (1 << (nap - 1 - s.mj)));
        sh[t * 2] = s.skip ? -1 : (int)s.kc + 6;
        sh[t * 2 + 1] = (s.skip || s.drop) ? -1 : (int)s.kc + 6 - (int)s.q;
      } else {
        const p2::SingleScalars s = p2::single_scalars(d, nap, 0.f, beta, gamma, p.lo, p.hi);
        cs[t] = base + c1;
        sh[t] = s.skip ? -1 : (int)s.kc + 6;
      }
    }
    __syncthreads();
    if (stripe < nstripe) {                                 // phase B
      int acc[VEC];
#pragma unroll
      for (int k = 0; k < VEC; ++k) acc[k] = 0;
      for (int t = stripe; t < p.tph; t += nstripe) {
#pragma unroll
        for (int n = 0; n < N; ++n) {
          const int s = sh[t * N + n];
          if (s < 0) continue;
          int v[VEC];
          load_bytes<VEC>(p.P + (size_t)cs[t * N + n] * p.d_out + chunk * VEC, v);
#pragma unroll
          for (int k = 0; k < VEC; ++k) acc[k] += v[k] * (1 << s);
        }
      }
#pragma unroll
      for (int k = 0; k < VEC; ++k) part[stripe * p.d_out + chunk * VEC + k] = acc[k];
    }
    __syncthreads();
    for (int i = tid; i < p.d_out; i += bd) {               // phase C
      int y = 0;
      for (int s = 0; s < nstripe; ++s) y += part[s * p.d_out + i];
      out[row * p.d_out + i] = __int2float_rn(y);
    }
    __syncthreads();
  }
}

int vec_for(int d_out) {
  for (int v = 8; v > 1; v >>= 1)
    if (d_out % v == 0) return v;
  return 1;
}

int threads_for(int d_out) {
  const int need = ((d_out / vec_for(d_out) + 31) / 32) * 32;
  return need > DQ_THREADS ? need : DQ_THREADS;
}
}  // namespace dq

torch::Tensor table_exponents(const torch::Tensor& W, int64_t G, double shift) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat32 && W.is_contiguous() && W.dim() == 2,
              "W: contiguous fp32 CUDA [rows, d_out]");
  const c10::cuda::CUDAGuard guard(W.device());
  const int64_t rows = W.size(0), d_out = W.size(1);
  TORCH_CHECK(rows % G == 0, "W rows must be a multiple of G");
  auto E = torch::empty({G, d_out}, W.options());
  if (G == 0 || d_out == 0) return E;
  const int64_t rpg = rows / G;
  const int TX = (int)std::min<int64_t>(d_out, qf::TE_THREADS);
  const int TY = qf::TE_THREADS / TX;
  const int64_t want = (rpg * d_out + qf::TE_ELEMS_PER_CTA - 1) / qf::TE_ELEMS_PER_CTA;   // row splits per group
  const int nsplit = (int)std::max<int64_t>(1, std::min<int64_t>(want, std::max<int64_t>(rpg, 1)));
  const int rows_per_split = (int)((rpg + nsplit - 1) / nsplit);
  auto P = torch::empty({G, nsplit, d_out}, W.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  qf::te_partial_kernel<<<(unsigned)(G * nsplit), TX * TY, sizeof(float) * TX * TY, stream>>>(
      W.data_ptr<float>(), P.data_ptr<float>(), (int)rpg, (int)d_out, nsplit, rows_per_split, TX, TY);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t n = G * d_out;
  qf::te_final_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(P.data_ptr<float>(), E.data_ptr<float>(), (int)G,
                                                                        (int)d_out, nsplit, (float)shift);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return E;
}

namespace {
qf::Params read_params(const torch::Tensor& z, const torch::Tensor& anc_a, const torch::Tensor& anc_b, const torch::Tensor& W,
                       const torch::Tensor& E, const c10::optional<torch::Tensor>& dmask, const torch::Tensor& log_beta,
                       const torch::Tensor& log_gamma, const c10::optional<torch::Tensor>& log_tau, int64_t n, int64_t lo,
                       int64_t hi, int64_t Q, double qlo, double qhi) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.is_contiguous() && z.dim() == 3,
              "z: contiguous fp32 CUDA [B, G, d_in]");
  TORCH_CHECK(W.scalar_type() == torch::kFloat32 && W.is_contiguous() && W.dim() == 2, "W: contiguous fp32 [rows, d_out]");
  TORCH_CHECK(anc_a.scalar_type() == torch::kInt16 && anc_a.is_contiguous() && anc_a.dim() == 3 &&
              anc_b.scalar_type() == torch::kInt16 && anc_b.is_contiguous() && anc_b.sizes() == anc_a.sizes(),
              "anchors: contiguous int16 [G, tph, nap], pairs");
  qf::Params p{};
  p.z = z.data_ptr<float>();
  p.anc_a = anc_a.data_ptr<int16_t>();
  p.anc_b = anc_b.data_ptr<int16_t>();
  p.W = W.data_ptr<float>();
  p.E = E.data_ptr<float>();
  p.B = z.size(0); p.G = z.size(1); p.d_in = z.size(2);
  p.tph = anc_a.size(1); p.nap = anc_a.size(2); p.d_out = W.size(1);
  p.lo = lo; p.hi = hi; p.Q = Q; p.qlo = (float)qlo; p.qhi = (float)qhi;
  TORCH_CHECK(n == 1 || n == 2, "n (read_top_n) must be 1 or 2");
  // the packed per-table integers (pack_tab) budget 4 bits for k' + 8 and 7 for q: the window must lie in [-3, 4]
  TORCH_CHECK(lo >= -3 && hi <= 4 && lo <= hi && Q >= 0 && Q <= 3, "window must lie in [-3, 4] and Q in [0, 3]");
  TORCH_CHECK(p.nap >= 1 && p.nap <= qf::NAP_MAX, "nap must be in [1, 8]");
  TORCH_CHECK(anc_a.size(0) == p.G, "anchor groups != z groups");
  TORCH_CHECK(W.size(0) == ((int64_t)p.G * p.tph << p.nap), "W rows != G * tph * 2**nap");
  TORCH_CHECK(E.scalar_type() == torch::kFloat32 && E.is_contiguous() && E.size(0) == p.G && E.size(1) == p.d_out,
              "E: contiguous fp32 [G, d_out]");
  if (dmask.has_value())
    TORCH_CHECK(dmask->scalar_type() == torch::kFloat32 && dmask->is_contiguous() &&
                dmask->numel() == (int64_t)p.B * p.G * p.tph, "dmask: contiguous fp32 [B, G, tph]");
  p.dmask = dmask.has_value() ? dmask->data_ptr<float>() : nullptr;
  TORCH_CHECK(n == 1 || log_tau.has_value(), "the n = 2 read needs log_tau");
  for (const auto* t : {&log_beta, &log_gamma})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->numel() == 1, "log scalars: one-element fp32 CUDA");
  if (log_tau.has_value())
    TORCH_CHECK(log_tau->is_cuda() && log_tau->scalar_type() == torch::kFloat32 && log_tau->numel() == 1,
                "log_tau: one-element fp32 CUDA");
  p.log_beta = log_beta.data_ptr<float>();
  p.log_gamma = log_gamma.data_ptr<float>();
  p.log_tau = (n == 2 && log_tau.has_value()) ? log_tau->data_ptr<float>() : nullptr;   // read only by N = 2
  return p;
}
}  // namespace

// n = read_top_n (1 or 2; log_tau is required at 2 and unused at 1). Returns {out [B, G, d_out], tabs}: tabs is each
// table's packed integers, int32 [B, G, tph], when save_tables (the backward will run), else an empty tensor.
std::vector<torch::Tensor> quant_fwd(const torch::Tensor& z, const torch::Tensor& anc_a, const torch::Tensor& anc_b,
                                     const torch::Tensor& W, const torch::Tensor& E,
                                     const c10::optional<torch::Tensor>& dmask, const torch::Tensor& log_beta,
                                     const torch::Tensor& log_gamma, const c10::optional<torch::Tensor>& log_tau,
                                     int64_t n, int64_t lo, int64_t hi, int64_t Q, double qlo, double qhi,
                                     bool save_tables) {
  const c10::cuda::CUDAGuard guard(z.device());
  qf::Params p = read_params(z, anc_a, anc_b, W, E, dmask, log_beta, log_gamma, log_tau, n, lo, hi, Q, qlo, qhi);
  auto out = torch::empty({p.B, p.G, p.d_out}, z.options());
  auto tabs = save_tables ? torch::empty({p.B, p.G, p.tph}, z.options().dtype(torch::kInt32))
                          : torch::empty({0}, z.options().dtype(torch::kInt32));
  p.tabs = save_tables ? tabs.data_ptr<int32_t>() : nullptr;
  if (p.B == 0) return {out, tabs};
  const int vec = qf::vec_for(p.d_out), threads = qf::threads_for(p.d_out);
  TORCH_CHECK(threads <= 1024, "d_out too wide for one CTA");
  const int nstripe = threads / (p.d_out / vec);
  const qf::Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, (int)n);
  const int grid = p.G * ((p.B + qf::QF_ROWS - 1) / qf::QF_ROWS);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
    kernel<<<grid, threads, L.total, stream>>>(p, out.data_ptr<float>());
  };
  auto by_vec = [&](auto n_const) {
    constexpr int NN = decltype(n_const)::value;
    if (vec == 4) launch(qf::quant_fwd_kernel<4, NN>);
    else if (vec == 2) launch(qf::quant_fwd_kernel<2, NN>);
    else launch(qf::quant_fwd_kernel<1, NN>);
  };
  if (n == 2) by_vec(std::integral_constant<int, 2>{});
  else by_vec(std::integral_constant<int, 1>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, tabs};
}

// The backward of quant_fwd. go [B, G, d_out]; zoff [G, d_in + 1] / zent [G, 2 tph nap] (int32) the per-group map from
// each input dim to the anchor entries that read it (entry e < tph nap: anc_a[e], else anc_b[e - tph nap]), increasing.
// tabs [B, G, tph] int32: the forward's packed per-table integers (quant_fwd with save_tables).
// Returns gz [B, G, d_in], gW [rows, d_out] and the per-CTA scalar partials [grid, 3] (log_beta, log_gamma, log_tau;
// the tau column is 0 at n = 1, which has no tau).
std::vector<torch::Tensor> quant_bwd(const torch::Tensor& go, const torch::Tensor& z, const torch::Tensor& anc_a,
                                     const torch::Tensor& anc_b, const torch::Tensor& zoff, const torch::Tensor& zent,
                                     const torch::Tensor& tabs, const torch::Tensor& W, const torch::Tensor& E,
                                     const c10::optional<torch::Tensor>& dmask, const torch::Tensor& log_beta,
                                     const torch::Tensor& log_gamma, const c10::optional<torch::Tensor>& log_tau,
                                     int64_t n, int64_t lo, int64_t hi, int64_t Q, double qlo, double qhi) {
  const c10::cuda::CUDAGuard guard(z.device());
  qf::Params p = read_params(z, anc_a, anc_b, W, E, dmask, log_beta, log_gamma, log_tau, n, lo, hi, Q, qlo, qhi);
  TORCH_CHECK(tabs.is_cuda() && tabs.scalar_type() == torch::kInt32 && tabs.is_contiguous() &&
              tabs.numel() == (int64_t)p.B * p.G * p.tph, "tabs: contiguous int32 [B, G, tph] from quant_fwd");
  p.tabs = const_cast<int32_t*>(tabs.data_ptr<int32_t>());
  TORCH_CHECK(go.is_cuda() && go.scalar_type() == torch::kFloat32 && go.is_contiguous() && go.numel() == (int64_t)p.B * p.G * p.d_out,
              "go: contiguous fp32 [B, G, d_out]");
  TORCH_CHECK(zoff.scalar_type() == torch::kInt32 && zoff.is_contiguous() && zoff.numel() == (int64_t)p.G * (p.d_in + 1) &&
              zent.scalar_type() == torch::kInt32 && zent.is_contiguous() && zent.numel() == (int64_t)p.G * 2 * p.tph * p.nap,
              "zoff / zent: contiguous int32 [G, d_in + 1] / [G, 2 tph nap]");
  auto gz = torch::empty({p.B, p.G, p.d_in}, z.options());
  auto gW = torch::zeros_like(W);
  const int grid = p.G * ((p.B + qf::QF_ROWS - 1) / qf::QF_ROWS);
  auto gscal = torch::zeros({grid, 3}, z.options());
  if (p.B == 0) return {gz, gW, gscal};
  const int vec = qf::vec_for(p.d_out), threads = qf::threads_for(p.d_out);
  TORCH_CHECK(threads <= 1024, "d_out too wide for one CTA");
  const qf::SmemB L(p.tph, p.nap, p.d_in, p.d_out, p.d_out / vec, (int)n);
  const bool vec_atomics = at::cuda::getCurrentDeviceProperties()->major >= 9;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
    kernel<<<grid, threads, L.total, stream>>>(p, go.data_ptr<float>(), zoff.data_ptr<int>(), zent.data_ptr<int>(),
                                               gW.data_ptr<float>(), gz.data_ptr<float>(), gscal.data_ptr<float>(), vec_atomics);
  };
  auto by_vec = [&](auto n_const) {
    constexpr int NN = decltype(n_const)::value;
    if (vec == 4) launch(qf::quant_bwd_kernel<4, NN>);
    else if (vec == 2) launch(qf::quant_bwd_kernel<2, NN>);
    else launch(qf::quant_bwd_kernel<1, NN>);
  };
  if (n == 2) by_vec(std::integral_constant<int, 2>{});
  else by_vec(std::integral_constant<int, 1>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gz, gW, gscal};
}

// DeployedQuantisedConfidenceLUT's forward. z float32 [B, G, d_in]; anchors int16 [G, tph, nap] (anc_b None: single
// anchors); P int8 [G * tph * 2^nap, d_out]; n = 1 or 2 (the export's read_top_n). Returns the raw accumulator,
// float32.
torch::Tensor deploy_fwd(const torch::Tensor& z, const torch::Tensor& anc_a, const c10::optional<torch::Tensor>& anc_b,
                         const torch::Tensor& P, const torch::Tensor& log_beta, const torch::Tensor& log_gamma,
                         const c10::optional<torch::Tensor>& log_tau, int64_t n, int64_t lo, int64_t hi,
                         int64_t Q) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.is_contiguous() && z.dim() == 3,
              "z: contiguous fp32 CUDA [B, G, d_in]");
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kInt8 && P.is_contiguous() && P.dim() == 2,
              "P: contiguous int8 CUDA [rows, d_out]");
  TORCH_CHECK(anc_a.scalar_type() == torch::kInt16 && anc_a.is_contiguous() && anc_a.dim() == 3,
              "anchors: contiguous int16 [G, tph, nap]");
  if (anc_b.has_value())
    TORCH_CHECK(anc_b->scalar_type() == torch::kInt16 && anc_b->is_contiguous() && anc_b->sizes() == anc_a.sizes(),
                "anc_b: contiguous int16, the shape of anc_a");
  TORCH_CHECK(n == 1 || n == 2, "n must be 1 or 2");
  const c10::cuda::CUDAGuard guard(z.device());
  dq::Params p{};
  p.z = z.data_ptr<float>();
  p.anc_a = anc_a.data_ptr<int16_t>();
  p.anc_b = anc_b.has_value() ? anc_b->data_ptr<int16_t>() : nullptr;
  p.P = P.data_ptr<int8_t>();
  p.B = z.size(0); p.G = z.size(1); p.d_in = z.size(2);
  p.tph = anc_a.size(1); p.nap = anc_a.size(2); p.d_out = P.size(1);
  p.lo = lo; p.hi = hi; p.Q = Q;
  TORCH_CHECK(p.nap >= 1 && p.nap <= qf::NAP_MAX, "nap must be in [1, 8]");
  TORCH_CHECK(anc_a.size(0) == p.G, "anchor groups != z groups");
  TORCH_CHECK(P.size(0) == ((int64_t)p.G * p.tph << p.nap), "P rows != G * tph * 2**nap");
  TORCH_CHECK(n == 1 || log_tau.has_value(), "the n = 2 read needs log_tau");
  for (const auto* t : {&log_beta, &log_gamma})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->numel() == 1, "log scalars: one-element fp32 CUDA");
  if (log_tau.has_value())
    TORCH_CHECK(log_tau->is_cuda() && log_tau->scalar_type() == torch::kFloat32 && log_tau->numel() == 1,
                "log_tau: one-element fp32 CUDA");
  p.log_beta = log_beta.data_ptr<float>();
  p.log_gamma = log_gamma.data_ptr<float>();
  p.log_tau = log_tau.has_value() ? log_tau->data_ptr<float>() : nullptr;   // read only by the n = 2 kernel
  auto out = torch::empty({p.B, p.G, p.d_out}, z.options());
  if (p.B == 0) return out;
  const int vec = dq::vec_for(p.d_out), threads = dq::threads_for(p.d_out);
  TORCH_CHECK(threads <= 1024, "d_out too wide for one CTA");
  const int nstripe = threads / (p.d_out / vec);
  const dq::Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, (int)n);
  const int grid = p.G * ((p.B + qf::QF_ROWS - 1) / qf::QF_ROWS);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
    kernel<<<grid, threads, L.total, stream>>>(p, out.data_ptr<float>());
  };
  auto by_vec = [&](auto n_const) {
    constexpr int NN = decltype(n_const)::value;
    switch (vec) {   // dq::vec_for: 8, 4, 2 or 1
      case 8: launch(dq::deploy_fwd_kernel<8, NN>); break;
      case 4: launch(dq::deploy_fwd_kernel<4, NN>); break;
      case 2: launch(dq::deploy_fwd_kernel<2, NN>); break;
      default: launch(dq::deploy_fwd_kernel<1, NN>);
    }
  };
  if (n == 2) by_vec(std::integral_constant<int, 2>{});
  else by_vec(std::integral_constant<int, 1>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scalars", &p2_scalars, "p2::table_scalars for every table: forward of the lutorch_ex::p2_scalars op");
  m.def("table_exponents", &table_exponents, "per-(group, channel) fake-quant exponents e of the table");
  m.def("quant_fwd", &quant_fwd, "FusedQuantisedConfidenceLUT's fused forward (read_top_n 1 or 2): addressing, p2 integers, fake-quant read");
  m.def("quant_bwd", &quant_bwd, "FusedQuantisedConfidenceLUT's fused backward (the forward's integers, margins from z)");
  m.def("deploy_fwd", &deploy_fwd, "DeployedQuantisedConfidenceLUT's int8 read (read_top_n 1 or 2), the raw accumulator");
}
