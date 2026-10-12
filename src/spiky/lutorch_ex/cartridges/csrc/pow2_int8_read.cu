// Forward of the lutorch_ex::p2_scalars custom op: p2::table_scalars (csrc/pow2_scalars.cuh) for every table of a batch of
// margins. The op supplies the per-table integers (c1, c2, shift codes, k', q) and the straight-through cell weights of the
// power-of-two read; lutorch_ex accumulates the int8 rows with its torch read (pow2_read.int_blend_read).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cmath>

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
// FusedQuantisedConfidenceLUT's forward: two launches per call.
//
// 1. table_exponents: e[g, c] = ceil(log2(max_{t, cell} |W[g, t, cell, c]|)) - (bits - 1) + offset, the per-(group,
//    channel) power-of-two scale of the fake-quant table (_pow2.head_chan_exponents). Input-independent; a max, so its
//    result does not depend on the reduction order.
// 2. quant_fwd: the whole read, one CTA per (group g, QF_ROWS consecutive samples), structured as FusedConfidenceLUT's
//    forward (csrc/fused_confidence.cu):
//    phase A  thread per table: the margins d_j = z[a_j] - z[b_j] from the z row in shared memory, then p2::table_scalars
//             (the same code as p2_scalar_kernel: c1, c2, k', q, skip, drop) and the cell weights 2^k' / 2^(k'-q) (times
//             the table-dropout factor), all in registers. The margins never reach global memory.
//    phase B  stripe x chunk: nstripe = QF threads / (d_out / VEC) stripes; stripe s accumulates tables t = s, s +
//             nstripe, ... (increasing), each table adding w1 * fq(W[c1]) then w2 * fq(W[c2]), where fq is the fake-quant
//             value clamp(floor(W / 2^e + 1/2), qlo, qhi) * 2^e computed in registers from the fp32 master row (__ldg,
//             VEC floats per load). Every add is a separate fp32 add (built with --fmad=false: no contraction), and with
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
  const float* W;           // [G * tph * K, d_out] fp32 master table
  const float* E;           // [G, d_out] the exponents e (integers, as float)
  const float* dmask;       // [B, G, tph] table-dropout factors, nullptr = none
  const float* log_beta;
  const float* log_gamma;
  const float* log_tau;
  int B, G, tph, nap, d_in, d_out, lo, hi, Q;
  float qlo, qhi;
};

__host__ __device__ inline size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

struct Smem {
  size_t zs, w1, w2, cs, cas, part, sc, si, sa, sb, total;
  __host__ __device__ Smem(int tph, int nap, int d_in, int d_out, int nstripe) {
    size_t o = 0;
    zs = o;   o = align16(o + sizeof(float) * d_in);
    w1 = o;   o = align16(o + sizeof(float) * tph);
    w2 = o;   o = align16(o + sizeof(float) * tph);
    cs = o;   o = align16(o + sizeof(int) * tph);
    cas = o;  o = align16(o + sizeof(int) * tph);
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

__global__ void table_exponent_kernel(const float* __restrict__ W, float* __restrict__ E, int rows_per_group, int d_out,
                                      float shift) {
  __shared__ float red[32][33];
  const int g = blockIdx.x;                                // one block per group (no product: G is small)
  const int c = blockIdx.y * 32 + threadIdx.x;             // channel
  float m = 0.f;
  if (c < d_out) {
    const float* base = W + (size_t)g * rows_per_group * d_out + c;
    for (int r = threadIdx.y; r < rows_per_group; r += 32) m = fmaxf(m, fabsf(__ldg(base + (size_t)r * d_out)));
  }
  red[threadIdx.y][threadIdx.x] = m;
  __syncthreads();
  if (threadIdx.y == 0 && c < d_out) {
    float a = red[0][threadIdx.x];
    for (int y = 1; y < 32; ++y) a = fmaxf(a, red[y][threadIdx.x]);
    E[(size_t)g * d_out + c] = ceilf(log2f(fmaxf(a, 1e-30f))) - shift;   // _pow2.head_chan_exponents
  }
}

template <int VEC>
__global__ void quant_fwd_kernel(Params p, float* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int nchunk = p.d_out / VEC;
  const int nstripe = blockDim.x / nchunk;
  const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe);
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
  const float beta = expf(*p.log_beta), gamma = expf(*p.log_gamma), tau = expf(*p.log_tau);
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
      const p2::TableScalars s = p2::table_scalars(d, p.nap, tau, 0.f, beta, gamma, p.lo, p.hi, p.Q);
      float a1 = s.skip ? 0.f : ldexpf(1.0f, s.kc);
      float a2 = (s.skip || s.drop) ? 0.f : ldexpf(1.0f, (int)s.kc - (int)s.q);
      if (p.dmask) {
        const float k = p.dmask[row * p.tph + t];
        a1 = a1 * k;
        a2 = a2 * k;
      }
      const int base = (g * p.tph + t) << p.nap;
      cs[t] = base + s.c1;
      cas[t] = base + s.c2;
      w1[t] = a1;
      w2[t] = a2;
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
        load_vec<VEC>(p.W + (size_t)cas[t] * p.d_out + chunk * VEC, v);
        const float a2 = w2[t];
#pragma unroll
        for (int k = 0; k < VEC; ++k)
          acc[k] = acc[k] + a2 * (fminf(fmaxf(floorf(v[k] * inv[k] + 0.5f), p.qlo), p.qhi) * scl[k]);
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
// quant_bwd: the backward of quant_fwd, structured as FusedConfidenceLUT's backward. Nothing is saved from the forward but
// z, the parameters, the exponents E and the dropout factors: per (group, row) the CTA re-stages z and recomputes phase A.
//   phase A  thread per table: margins (kept in shared memory, tph x nap per CTA), p2::table_scalars, the forward cell
//            weights w1 / w2 (x dropout) and the straight-through numerators b1 = 2^k', b2 = 2^(k'-q) (clamped k', also
//            for skipped / dropped cells, as _pow2.ste_blend_weights).
//   phase B  stripe x chunk: gW[c] += w go (fp32 atomics, vector atomics on sm_90+: the ONLY atomics; gW feeds no
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
  __host__ __device__ SmemB(int tph, int nap, int d_in, int d_out, int nchunk) {
    size_t o = 0;
    zs = o;   o = align16(o + sizeof(float) * d_in);
    gos = o;  o = align16(o + sizeof(float) * d_out);
    w1 = o;   o = align16(o + sizeof(float) * tph);
    w2 = o;   o = align16(o + sizeof(float) * tph);
    b1 = o;   o = align16(o + sizeof(float) * tph);
    b2 = o;   o = align16(o + sizeof(float) * tph);
    cs = o;   o = align16(o + sizeof(int) * tph);
    cas = o;  o = align16(o + sizeof(int) * tph);
    us = o;   o = align16(o + sizeof(float) * tph * nap);
    gd = o;   o = align16(o + sizeof(float) * tph * nap);
    rp1 = o;  o = align16(o + sizeof(float) * nchunk * tph);
    rp2 = o;  o = align16(o + sizeof(float) * nchunk * tph);
    sc = o;   o = align16(o + sizeof(float) * d_out);
    si = o;   o = align16(o + sizeof(float) * d_out);
    sa = o;   o = align16(o + sizeof(int16_t) * tph * nap);
    sb = o;   o = align16(o + sizeof(int16_t) * tph * nap);
    total = o;
  }
};

template <int VEC>
__global__ void quant_bwd_kernel(Params p, const float* __restrict__ go, const int* __restrict__ zoff,
                                 const int* __restrict__ zent, float* __restrict__ gW, float* __restrict__ gz,
                                 float* __restrict__ gscal, bool vec_atomics) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ float red[32];
  const int nchunk = p.d_out / VEC;
  const int nstripe = blockDim.x / nchunk;
  const SmemB L(p.tph, p.nap, p.d_in, p.d_out, nchunk);
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
  const float beta = expf(*p.log_beta), gamma = expf(*p.log_gamma), tau = expf(*p.log_tau);
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
      float d[NAP_MAX];
#pragma unroll
      for (int j = 0; j < NAP_MAX; ++j)
        if (j < nap) {
          d[j] = zs[sa[t * nap + j]] - zs[sb[t * nap + j]];
          us[t * nap + j] = d[j];
        }
      const p2::TableScalars s = p2::table_scalars(d, nap, tau, 0.f, beta, gamma, p.lo, p.hi, p.Q);
      const float k = p.dmask ? p.dmask[row * p.tph + t] : 1.f;
      w1[t] = (s.skip ? 0.f : ldexpf(1.0f, s.kc)) * k;
      w2[t] = ((s.skip || s.drop) ? 0.f : ldexpf(1.0f, (int)s.kc - (int)s.q)) * k;
      b1[t] = ldexpf(1.0f, s.kc);
      b2[t] = ldexpf(1.0f, (int)s.kc - (int)s.q);
      const int base = (g * p.tph + t) << nap;
      cs[t] = base + s.c1;
      cas[t] = base + s.c2;
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
    __syncthreads();
    for (int t = tid; t < p.tph; t += bd) {                 // phase C
      float r1 = 0.f, r2 = 0.f;
      for (int c = 0; c < nchunk; ++c) { r1 = r1 + rp1[c * p.tph + t]; r2 = r2 + rp2[c * p.tph + t]; }
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
          if (j == 0 || m < mstar) { mstar = m; jstar = j; } // first argmin, as torch.min
        }
      const float E = expf(gamma * Ls);
      const float s = M * E;
      const float x = 2.0f * mstar / tau;
      const float sp = 1.f / (1.f + expf(-x)), sn = 1.f / (1.f + expf(x));
      const float k = p.dmask ? p.dmask[row * p.tph + t] : 1.f;
      const float G1 = r1 * k * (b1[t] / fmaxf(s * sp, 1e-30f));
      const float G2 = r2 * k * (b2[t] / fmaxf(s * sn, 1e-30f));
      const float g_s = G1 * sp + G2 * sn;
      const float g_x = s * (G1 - G2) * sp * sn;
      const float g_ls = g_s * s * gamma;
      g_lbeta += g_ls * beta * sig_m;
      g_lgamma += g_ls * Ls;
      g_ltau += -g_x * x;
      const float g_mv = g_x * 2.0f / tau;
#pragma unroll
      for (int j = 0; j < NAP_MAX; ++j)
        if (j < nap) {
          const float u = us[t * nap + j];
          const float m = fabsf(u);
          float gm = g_s * E + g_ls * beta / (1.f + expf(beta * m));
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

torch::Tensor table_exponents(const torch::Tensor& W, int64_t G, double shift) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat32 && W.is_contiguous() && W.dim() == 2,
              "W: contiguous fp32 CUDA [rows, d_out]");
  const c10::cuda::CUDAGuard guard(W.device());
  const int64_t rows = W.size(0), d_out = W.size(1);
  TORCH_CHECK(rows % G == 0, "W rows must be a multiple of G");
  auto E = torch::empty({G, d_out}, W.options());
  dim3 grid((unsigned)G, (unsigned)((d_out + 31) / 32)), block(32, 32);
  qf::table_exponent_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<float>(), E.data_ptr<float>(), (int)(rows / G), (int)d_out, (float)shift);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return E;
}

namespace {
qf::Params read_params(const torch::Tensor& z, const torch::Tensor& anc_a, const torch::Tensor& anc_b, const torch::Tensor& W,
                       const torch::Tensor& E, const c10::optional<torch::Tensor>& dmask, const torch::Tensor& log_beta,
                       const torch::Tensor& log_gamma, const torch::Tensor& log_tau, int64_t lo, int64_t hi, int64_t Q,
                       double qlo, double qhi) {
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
  TORCH_CHECK(p.nap >= 1 && p.nap <= qf::NAP_MAX, "nap must be in [1, 8]");
  TORCH_CHECK(anc_a.size(0) == p.G, "anchor groups != z groups");
  TORCH_CHECK(W.size(0) == ((int64_t)p.G * p.tph << p.nap), "W rows != G * tph * 2**nap");
  TORCH_CHECK(E.scalar_type() == torch::kFloat32 && E.is_contiguous() && E.size(0) == p.G && E.size(1) == p.d_out,
              "E: contiguous fp32 [G, d_out]");
  if (dmask.has_value())
    TORCH_CHECK(dmask->scalar_type() == torch::kFloat32 && dmask->is_contiguous() &&
                dmask->numel() == (int64_t)p.B * p.G * p.tph, "dmask: contiguous fp32 [B, G, tph]");
  p.dmask = dmask.has_value() ? dmask->data_ptr<float>() : nullptr;
  for (const auto* t : {&log_beta, &log_gamma, &log_tau})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kFloat32 && t->numel() == 1, "log scalars: one-element fp32 CUDA");
  p.log_beta = log_beta.data_ptr<float>();
  p.log_gamma = log_gamma.data_ptr<float>();
  p.log_tau = log_tau.data_ptr<float>();
  return p;
}
}  // namespace

torch::Tensor quant_fwd(const torch::Tensor& z, const torch::Tensor& anc_a, const torch::Tensor& anc_b,
                        const torch::Tensor& W, const torch::Tensor& E, const c10::optional<torch::Tensor>& dmask,
                        const torch::Tensor& log_beta, const torch::Tensor& log_gamma, const torch::Tensor& log_tau,
                        int64_t lo, int64_t hi, int64_t Q, double qlo, double qhi) {
  const c10::cuda::CUDAGuard guard(z.device());
  const qf::Params p = read_params(z, anc_a, anc_b, W, E, dmask, log_beta, log_gamma, log_tau, lo, hi, Q, qlo, qhi);
  auto out = torch::empty({p.B, p.G, p.d_out}, z.options());
  if (p.B == 0) return out;
  const int vec = qf::vec_for(p.d_out), threads = qf::threads_for(p.d_out);
  TORCH_CHECK(threads <= 1024, "d_out too wide for one CTA");
  const int nstripe = threads / (p.d_out / vec);
  const qf::Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe);
  const int grid = p.G * ((p.B + qf::QF_ROWS - 1) / qf::QF_ROWS);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
    kernel<<<grid, threads, L.total, stream>>>(p, out.data_ptr<float>());
  };
  if (vec == 4) launch(qf::quant_fwd_kernel<4>);
  else if (vec == 2) launch(qf::quant_fwd_kernel<2>);
  else launch(qf::quant_fwd_kernel<1>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The backward of quant_fwd. go [B, G, d_out]; zoff [G, d_in + 1] / zent [G, 2 tph nap] (int32) the per-group map from
// each input dim to the anchor entries that read it (entry e < tph nap: anc_a[e], else anc_b[e - tph nap]), increasing.
// Returns gz [B, G, d_in], gW [rows, d_out] and the per-CTA scalar partials [grid, 3] (log_beta, log_gamma, log_tau).
std::vector<torch::Tensor> quant_bwd(const torch::Tensor& go, const torch::Tensor& z, const torch::Tensor& anc_a,
                                     const torch::Tensor& anc_b, const torch::Tensor& zoff, const torch::Tensor& zent,
                                     const torch::Tensor& W, const torch::Tensor& E,
                                     const c10::optional<torch::Tensor>& dmask, const torch::Tensor& log_beta,
                                     const torch::Tensor& log_gamma, const torch::Tensor& log_tau, int64_t lo, int64_t hi,
                                     int64_t Q, double qlo, double qhi) {
  const c10::cuda::CUDAGuard guard(z.device());
  const qf::Params p = read_params(z, anc_a, anc_b, W, E, dmask, log_beta, log_gamma, log_tau, lo, hi, Q, qlo, qhi);
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
  const qf::SmemB L(p.tph, p.nap, p.d_in, p.d_out, p.d_out / vec);
  const bool vec_atomics = at::cuda::getCurrentDeviceProperties()->major >= 9;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
    kernel<<<grid, threads, L.total, stream>>>(p, go.data_ptr<float>(), zoff.data_ptr<int>(), zent.data_ptr<int>(),
                                               gW.data_ptr<float>(), gz.data_ptr<float>(), gscal.data_ptr<float>(), vec_atomics);
  };
  if (vec == 4) launch(qf::quant_bwd_kernel<4>);
  else if (vec == 2) launch(qf::quant_bwd_kernel<2>);
  else launch(qf::quant_bwd_kernel<1>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gz, gW, gscal};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scalars", &p2_scalars, "p2::table_scalars for every table: forward of the lutorch_ex::p2_scalars op");
  m.def("table_exponents", &table_exponents, "per-(group, channel) fake-quant exponents e of the table");
  m.def("quant_fwd", &quant_fwd, "FusedQuantisedConfidenceLUT's fused forward: addressing, p2 integers, fake-quant read");
  m.def("quant_bwd", &quant_bwd, "FusedQuantisedConfidenceLUT's fused backward (recomputes phase A from z)");
}
