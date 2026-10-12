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

torch::Tensor quant_fwd(const torch::Tensor& z, const torch::Tensor& anc_a, const torch::Tensor& anc_b,
                        const torch::Tensor& W, const torch::Tensor& E, const c10::optional<torch::Tensor>& dmask,
                        const torch::Tensor& log_beta, const torch::Tensor& log_gamma, const torch::Tensor& log_tau,
                        int64_t lo, int64_t hi, int64_t Q, double qlo, double qhi) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.is_contiguous() && z.dim() == 3,
              "z: contiguous fp32 CUDA [B, G, d_in]");
  TORCH_CHECK(W.scalar_type() == torch::kFloat32 && W.is_contiguous() && W.dim() == 2, "W: contiguous fp32 [rows, d_out]");
  TORCH_CHECK(anc_a.scalar_type() == torch::kInt16 && anc_a.is_contiguous() && anc_a.dim() == 3 &&
              anc_b.scalar_type() == torch::kInt16 && anc_b.is_contiguous() && anc_b.sizes() == anc_a.sizes(),
              "anchors: contiguous int16 [G, tph, nap], pairs");
  const c10::cuda::CUDAGuard guard(z.device());
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scalars", &p2_scalars, "p2::table_scalars for every table: forward of the lutorch_ex::p2_scalars op");
  m.def("table_exponents", &table_exponents, "per-(group, channel) fake-quant exponents e of the table");
  m.def("quant_fwd", &quant_fwd, "FusedQuantisedConfidenceLUT's fused forward: addressing, p2 integers, fake-quant read");
}
