// ConfidenceLUTCuda: hand-written CUDA forward + backward for the ConfidenceLUT read (read_top_n 1 and 2).
//
// One CTA owns one group g and `rows_per_cta` consecutive samples b of it. Per (b, g) row the CTA
//   phase A  (thread per table)   margins u_j from the z row in shared memory, the MSB-first address c, the
//                                 least-confident bit j*, the confidence score s and (n = 2) the blend weight v;
//   phase B  (stripe x chunk)     the score-weighted read sum_t w_t W[c_t] (forward), or the weight-gradient
//                                 scatter + the per-table dot products W[c_t] . go (backward), VEC floats per load;
//   phase C                       forward: reduce the stripes into the output row; backward: the score / blend
//                                 gradients scattered into the z row (shared memory) and the scalar gradients.
// Nothing per-table goes through global memory: the address, score, mask and margins live in registers / shared
// memory, and the backward recomputes them from z instead of saving them.
//
// Launch knobs (host side, see confidence_cuda.py): threads per CTA (forward and backward separately), rows per
// CTA, vector width VEC (1, 2, 4) and whether the weight-gradient scatter uses vector atomics (sm_90+).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int MAX_NAP = 16;

struct Params {
    const float* z;          // [B, G, d_in]
    const int16_t* anc_a;    // [G, tph, nap]
    const int16_t* anc_b;    // [G, tph, nap], nullptr in single-anchor mode
    const float* W;          // [G * tph * K, d_out]
    const uint8_t* keep;     // [B, G, tph] table-dropout keep flags, nullptr = no dropout
    const float* log_beta;   // 0-dim device scalars (read in-kernel: no host sync)
    const float* log_gamma;
    const float* log_tau;    // read only when N == 2
    float keep_scale;        // 1 / keep_prob
    float eps;
    int B, G, tph, nap, d_in, d_out, rows_per_cta;
};

__host__ __device__ inline size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

// Shared-memory layout, identical on host (size) and device (pointers).
struct Smem {
    size_t zs, gos, gzs, w1, w2, cs, cas, rc, ra, us, part, sa, sb, total;
    __host__ __device__ Smem(int tph, int nap, int d_in, int d_out, int nstripe, bool bwd) {
        size_t o = 0;
        zs = o;   o = align16(o + sizeof(float) * d_in);
        gos = o;  o = align16(o + (bwd ? sizeof(float) * d_out : 0));
        gzs = o;  o = align16(o + (bwd ? sizeof(float) * d_in : 0));
        w1 = o;   o = align16(o + sizeof(float) * tph);
        w2 = o;   o = align16(o + sizeof(float) * tph);
        cs = o;   o = align16(o + sizeof(int) * tph);
        cas = o;  o = align16(o + sizeof(int) * tph);
        rc = o;   o = align16(o + (bwd ? sizeof(float) * tph : 0));
        ra = o;   o = align16(o + (bwd ? sizeof(float) * tph : 0));
        us = o;   o = align16(o + (bwd ? sizeof(float) * tph * nap : 0));
        part = o; o = align16(o + (bwd ? 0 : sizeof(float) * nstripe * d_out));
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

// L2 eviction-priority policy for the W reads (knob l2_hint): 0 none, 1 evict_last, 2 evict_first. Built once per
// CTA with createpolicy (sm_80+), then passed to ld.global.nc.L2::cache_hint -- the same mechanism Triton emits for
// eviction_policy='evict_last'.
template <int HINT>
__device__ inline uint64_t make_l2_policy() {
    uint64_t pol = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (HINT == 1) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    if constexpr (HINT == 2) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
    return pol;
}

template <int VEC, int HINT>
__device__ inline void load_w(const float* p, float (&v)[VEC], uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (HINT != 0) {
        if constexpr (VEC == 4) {
            asm volatile("ld.global.nc.L2::cache_hint.v4.f32 {%0, %1, %2, %3}, [%4], %5;"
                         : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p), "l"(pol));
        } else if constexpr (VEC == 2) {
            asm volatile("ld.global.nc.L2::cache_hint.v2.f32 {%0, %1}, [%2], %3;"
                         : "=f"(v[0]), "=f"(v[1]) : "l"(p), "l"(pol));
        } else {
            asm volatile("ld.global.nc.L2::cache_hint.f32 %0, [%1], %2;" : "=f"(v[0]) : "l"(p), "l"(pol));
        }
        return;
    }
#endif
    load_vec<VEC>(p, v);
}

// log(sigmoid(x)), the same formula Inductor emits: min(0, x) - log1p(exp(-|x|)).
__device__ inline float log_sigmoid(float x) { return fminf(0.f, x) - log1pf(expf(-fabsf(x))); }

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

// Phase A, shared by forward and backward: one table's address, score and blend weight.
// Writes cs/cas (flat W rows), w1/w2 (the read weights) and, when us != nullptr, the margins.
template <int N>
__device__ inline void address_table(const Params& p, int g, int b, int t, const float* zs, const int16_t* sa,
                                     const int16_t* sb, float beta, float gamma, float tau,
                                     int* cs, int* cas, float* w1, float* w2, float* us) {
    const int nap = p.nap;
    float M = 0.f, L = 0.f, mstar = INFINITY;
    int c = 0, jstar = 0;
#pragma unroll
    for (int j = 0; j < MAX_NAP; ++j) {
        if (j < nap) {
            const int i = t * nap + j;
            const float u = zs[sa[i]] - (sb ? zs[sb[i]] : 0.f);
            if (us) us[i] = u;
            c |= int(u > p.eps) << (nap - 1 - j);
            const float m = fabsf(u);
            M += m;
            L += log_sigmoid(beta * m);
            if (m < mstar) { mstar = m; jstar = j; }       // first minimum, as torch.min
        }
    }
    float s = M * expf(gamma * L);
    if (p.keep) s *= p.keep[((size_t)b * p.G + g) * p.tph + t] ? p.keep_scale : 0.f;
    const int base = (g * p.tph + t) << nap;
    cs[t] = base + c;
    if constexpr (N == 2) {
        const float v = 1.f / (1.f + expf(2.f * mstar / tau));   // sigmoid(-2 |u_j*| / tau)
        cas[t] = base + (c ^ (1 << (nap - 1 - jstar)));
        w1[t] = s * (1.f - v);
        w2[t] = s * v;
    } else {
        w1[t] = s;
    }
}

template <int N, int VEC, int HINT>
__global__ void confidence_fwd_kernel(Params p, float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int nchunk = p.d_out / VEC;
    const int nstripe = blockDim.x / nchunk;
    const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, false);
    float* zs = reinterpret_cast<float*>(smem_raw + L.zs);
    float* w1 = reinterpret_cast<float*>(smem_raw + L.w1);
    float* w2 = reinterpret_cast<float*>(smem_raw + L.w2);
    int* cs = reinterpret_cast<int*>(smem_raw + L.cs);
    int* cas = reinterpret_cast<int*>(smem_raw + L.cas);
    float* part = reinterpret_cast<float*>(smem_raw + L.part);
    int16_t* sa = reinterpret_cast<int16_t*>(smem_raw + L.sa);
    int16_t* sb = p.anc_b ? reinterpret_cast<int16_t*>(smem_raw + L.sb) : nullptr;

    const int tid = threadIdx.x, bd = blockDim.x;
    const int g = blockIdx.x % p.G;
    const int b0 = (blockIdx.x / p.G) * p.rows_per_cta;
    const int an = p.tph * p.nap;
    for (int i = tid; i < an; i += bd) {                 // this group's anchors, once per CTA
        sa[i] = p.anc_a[(size_t)g * an + i];
        if (sb) sb[i] = p.anc_b[(size_t)g * an + i];
    }
    const float beta = expf(*p.log_beta), gamma = expf(*p.log_gamma);
    const float tau = (N == 2) ? expf(*p.log_tau) : 1.f;
    const int stripe = tid / nchunk, chunk = tid % nchunk;
    const uint64_t pol = make_l2_policy<HINT>();

    for (int r = 0; r < p.rows_per_cta; ++r) {
        const int b = b0 + r;
        if (b >= p.B) break;                              // uniform across the CTA
        const size_t row = (size_t)b * p.G + g;
        for (int i = tid; i < p.d_in; i += bd) zs[i] = p.z[row * p.d_in + i];
        __syncthreads();
        for (int t = tid; t < p.tph; t += bd)
            address_table<N>(p, g, b, t, zs, sa, sb, beta, gamma, tau, cs, cas, w1, w2, nullptr);
        __syncthreads();
        if (stripe < nstripe) {
            float acc[VEC] = {};
            for (int t = stripe; t < p.tph; t += nstripe) {
                float v[VEC];
                load_w<VEC, HINT>(p.W + (size_t)cs[t] * p.d_out + chunk * VEC, v, pol);
                const float a = w1[t];
#pragma unroll
                for (int k = 0; k < VEC; ++k) acc[k] = fmaf(a, v[k], acc[k]);
                if constexpr (N == 2) {
                    load_w<VEC, HINT>(p.W + (size_t)cas[t] * p.d_out + chunk * VEC, v, pol);
                    const float a2 = w2[t];
#pragma unroll
                    for (int k = 0; k < VEC; ++k) acc[k] = fmaf(a2, v[k], acc[k]);
                }
            }
#pragma unroll
            for (int k = 0; k < VEC; ++k) part[stripe * p.d_out + chunk * VEC + k] = acc[k];
        }
        __syncthreads();
        for (int i = tid; i < p.d_out; i += bd) {
            float y = 0.f;
            for (int s = 0; s < nstripe; ++s) y += part[s * p.d_out + i];
            out[row * p.d_out + i] = y;
        }
        __syncthreads();                                  // part / zs reused by the next row
    }
}

__device__ inline float block_sum(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    if (threadIdx.x == 0) for (int w = 0; w < nw; ++w) s += red[w];
    return s;                                             // valid in thread 0
}

template <int N, int VEC, int HINT>
__global__ void confidence_bwd_kernel(Params p, const float* __restrict__ go, float* __restrict__ gW,
                                      float* __restrict__ gz, float* __restrict__ gscal, bool vec_atomics) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int nchunk = p.d_out / VEC;
    const int nstripe = blockDim.x / nchunk;
    const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, true);
    float* zs = reinterpret_cast<float*>(smem_raw + L.zs);
    float* gos = reinterpret_cast<float*>(smem_raw + L.gos);
    float* gzs = reinterpret_cast<float*>(smem_raw + L.gzs);
    float* w1 = reinterpret_cast<float*>(smem_raw + L.w1);
    float* w2 = reinterpret_cast<float*>(smem_raw + L.w2);
    int* cs = reinterpret_cast<int*>(smem_raw + L.cs);
    int* cas = reinterpret_cast<int*>(smem_raw + L.cas);
    float* rc = reinterpret_cast<float*>(smem_raw + L.rc);
    float* ra = reinterpret_cast<float*>(smem_raw + L.ra);
    float* us = reinterpret_cast<float*>(smem_raw + L.us);
    int16_t* sa = reinterpret_cast<int16_t*>(smem_raw + L.sa);
    int16_t* sb = p.anc_b ? reinterpret_cast<int16_t*>(smem_raw + L.sb) : nullptr;
    __shared__ float red[32];

    const int tid = threadIdx.x, bd = blockDim.x;
    const int g = blockIdx.x % p.G;
    const int b0 = (blockIdx.x / p.G) * p.rows_per_cta;
    const int an = p.tph * p.nap, nap = p.nap;
    for (int i = tid; i < an; i += bd) {
        sa[i] = p.anc_a[(size_t)g * an + i];
        if (sb) sb[i] = p.anc_b[(size_t)g * an + i];
    }
    const float beta = expf(*p.log_beta), gamma = expf(*p.log_gamma);
    const float tau = (N == 2) ? expf(*p.log_tau) : 1.f;
    const int stripe = tid / nchunk, chunk = tid % nchunk;
    float g_lbeta = 0.f, g_lgamma = 0.f, g_ltau = 0.f;   // this thread's share of the scalar gradients
    const uint64_t pol = make_l2_policy<HINT>();

    for (int r = 0; r < p.rows_per_cta; ++r) {
        const int b = b0 + r;
        if (b >= p.B) break;
        const size_t row = (size_t)b * p.G + g;
        for (int i = tid; i < p.d_in; i += bd) { zs[i] = p.z[row * p.d_in + i]; gzs[i] = 0.f; }
        for (int i = tid; i < p.d_out; i += bd) gos[i] = go[row * p.d_out + i];
        __syncthreads();
        for (int t = tid; t < p.tph; t += bd) {
            address_table<N>(p, g, b, t, zs, sa, sb, beta, gamma, tau, cs, cas, w1, w2, us);
            rc[t] = 0.f;
            ra[t] = 0.f;
        }
        __syncthreads();
        // Phase B: weight-gradient scatter (gW[c] += w go) and the dot products r = W[c] . go.
        if (stripe < nstripe) {
            float gv[VEC];
#pragma unroll
            for (int k = 0; k < VEC; ++k) gv[k] = gos[chunk * VEC + k];
            for (int t = stripe; t < p.tph; t += nstripe) {
                float v[VEC], d[VEC];
                const size_t off = (size_t)cs[t] * p.d_out + chunk * VEC;
                load_w<VEC, HINT>(p.W + off, v, pol);
                float dot = 0.f;
#pragma unroll
                for (int k = 0; k < VEC; ++k) { dot = fmaf(v[k], gv[k], dot); d[k] = w1[t] * gv[k]; }
                atomic_add_vec<VEC>(gW + off, d, vec_atomics);
                atomicAdd(&rc[t], dot);
                if constexpr (N == 2) {
                    const size_t off2 = (size_t)cas[t] * p.d_out + chunk * VEC;
                    load_w<VEC, HINT>(p.W + off2, v, pol);
                    float dot2 = 0.f;
#pragma unroll
                    for (int k = 0; k < VEC; ++k) { dot2 = fmaf(v[k], gv[k], dot2); d[k] = w2[t] * gv[k]; }
                    atomic_add_vec<VEC>(gW + off2, d, vec_atomics);
                    atomicAdd(&ra[t], dot2);
                }
            }
        }
        __syncthreads();
        // Phase C: score / blend gradients -> margins -> the z row; scalar gradients.
        for (int t = tid; t < p.tph; t += bd) {
            float M = 0.f, Lsum = 0.f, mstar = INFINITY, sig_m = 0.f;
            int jstar = 0;
#pragma unroll
            for (int j = 0; j < MAX_NAP; ++j) {
                if (j < nap) {
                    const float m = fabsf(us[t * nap + j]);
                    M += m;
                    Lsum += log_sigmoid(beta * m);
                    sig_m += m / (1.f + expf(beta * m));           // sigma(-beta m) * m
                    if (m < mstar) { mstar = m; jstar = j; }
                }
            }
            const float E = expf(gamma * Lsum);
            const float s = M * E;
            float mk = 1.f;
            if (p.keep) mk = p.keep[row * p.tph + t] ? p.keep_scale : 0.f;
            float g_s, g_v = 0.f, v = 0.f;
            if constexpr (N == 2) {
                v = 1.f / (1.f + expf(2.f * mstar / tau));
                g_s = ((1.f - v) * rc[t] + v * ra[t]) * mk;
                g_v = s * mk * (ra[t] - rc[t]);
            } else {
                g_s = rc[t] * mk;
            }
            const float gsg = g_s * s * gamma;
            g_lbeta += gsg * beta * sig_m;
            g_lgamma += gsg * Lsum;
            const float dv_dm = (N == 2) ? g_v * (-2.f / tau) * v * (1.f - v) : 0.f;
            if constexpr (N == 2) g_ltau += g_v * v * (1.f - v) * 2.f * mstar / tau;
#pragma unroll
            for (int j = 0; j < MAX_NAP; ++j) {
                if (j < nap) {
                    const int i = t * nap + j;
                    const float u = us[i];
                    const float m = fabsf(u);
                    float gm = g_s * E + gsg * beta / (1.f + expf(beta * m));
                    if (j == jstar) gm += dv_dm;
                    const float gu = gm * float((u > 0.f) - (u < 0.f));   // d|u|/du = sign(u), 0 at 0
                    atomicAdd(&gzs[sa[i]], gu);
                    if (sb) atomicAdd(&gzs[sb[i]], -gu);
                }
            }
        }
        __syncthreads();
        for (int i = tid; i < p.d_in; i += bd) gz[row * p.d_in + i] = gzs[i];
        __syncthreads();
    }
    // Per-CTA partials of the scalar gradients; summed on the host side (no contended global atomics).
    float s0 = block_sum(g_lbeta, red);
    float s1 = block_sum(g_lgamma, red);
    float s2 = block_sum(g_ltau, red);
    if (tid == 0) {
        gscal[blockIdx.x * 3 + 0] = s0;
        gscal[blockIdx.x * 3 + 1] = s1;
        gscal[blockIdx.x * 3 + 2] = s2;
    }
}

Params make_params(const torch::Tensor& z, const torch::Tensor& anc_a, const c10::optional<torch::Tensor>& anc_b,
                   const torch::Tensor& W, const c10::optional<torch::Tensor>& keep, double keep_scale,
                   const torch::Tensor& log_beta, const torch::Tensor& log_gamma, const torch::Tensor& log_tau,
                   int nap, double eps, int rows_per_cta) {
    TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.is_contiguous() && z.dim() == 3, "z: contiguous fp32 CUDA [B, G, d_in]");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32 && W.is_contiguous() && W.dim() == 2, "W: contiguous fp32 [rows, d_out]");
    TORCH_CHECK(anc_a.scalar_type() == torch::kInt16 && anc_a.is_contiguous() && anc_a.dim() == 3, "anchors: contiguous int16 [G, tph, nap]");
    TORCH_CHECK(nap >= 1 && nap <= MAX_NAP, "nap must be in [1, 16]");
    Params p{};
    p.z = z.data_ptr<float>();
    p.anc_a = anc_a.data_ptr<int16_t>();
    p.anc_b = anc_b.has_value() ? anc_b->data_ptr<int16_t>() : nullptr;
    p.W = W.data_ptr<float>();
    if (keep.has_value())
        TORCH_CHECK(keep->scalar_type() == torch::kBool && keep->is_contiguous() && keep->numel() == z.size(0) * z.size(1) * anc_a.size(1),
                    "keep: contiguous bool [B, G, tph]");
    p.keep = keep.has_value() ? reinterpret_cast<const uint8_t*>(keep->data_ptr()) : nullptr;
    p.log_beta = log_beta.data_ptr<float>();
    p.log_gamma = log_gamma.data_ptr<float>();
    p.log_tau = log_tau.data_ptr<float>();
    p.keep_scale = float(keep_scale);
    p.eps = float(eps);
    p.B = z.size(0); p.G = z.size(1); p.d_in = z.size(2);
    p.tph = anc_a.size(1); p.nap = nap; p.d_out = W.size(1);
    p.rows_per_cta = rows_per_cta;
    TORCH_CHECK(anc_a.size(0) == p.G && anc_a.size(2) == nap, "anchor shape mismatch");
    TORCH_CHECK(W.size(0) == (int64_t)p.G * p.tph << nap, "W rows != G * tph * 2**nap");
    TORCH_CHECK(((int64_t)p.G * p.tph << nap) <= INT32_MAX, "flat cell index must fit int32");
    return p;
}

void check_launch(int threads, int vec, int d_out) {
    TORCH_CHECK(vec == 1 || vec == 2 || vec == 4, "vec must be 1, 2 or 4");
    TORCH_CHECK(d_out % vec == 0, "d_out must be divisible by vec");
    TORCH_CHECK(threads % 32 == 0 && threads >= 32 && threads <= 1024, "threads must be a multiple of 32 in [32, 1024]");
    TORCH_CHECK(threads >= d_out / vec, "threads must be >= d_out / vec (one stripe at least)");
}

#define DISPATCH_N_VEC(N, VEC, ...)                                                       \
    [&] {                                                                                 \
        if (N == 1 && VEC == 4) { constexpr int kN = 1, kV = 4; __VA_ARGS__(); }          \
        else if (N == 1 && VEC == 2) { constexpr int kN = 1, kV = 2; __VA_ARGS__(); }     \
        else if (N == 1 && VEC == 1) { constexpr int kN = 1, kV = 1; __VA_ARGS__(); }     \
        else if (N == 2 && VEC == 4) { constexpr int kN = 2, kV = 4; __VA_ARGS__(); }     \
        else if (N == 2 && VEC == 2) { constexpr int kN = 2, kV = 2; __VA_ARGS__(); }     \
        else if (N == 2 && VEC == 1) { constexpr int kN = 2, kV = 1; __VA_ARGS__(); }     \
        else TORCH_CHECK(false, "read_top_n must be 1 or 2");                             \
    }()

#define DISPATCH_HINT(H, ...)                                                             \
    [&] {                                                                                 \
        if (H == 0) { constexpr int kH = 0; __VA_ARGS__(); }                              \
        else if (H == 1) { constexpr int kH = 1; __VA_ARGS__(); }                         \
        else if (H == 2) { constexpr int kH = 2; __VA_ARGS__(); }                         \
        else TORCH_CHECK(false, "l2_hint must be 0 (none), 1 (evict_last) or 2 (evict_first)"); \
    }()

// Persisting-L2 access-policy window over [base, base + bytes) on `stream` for the lifetime of this object (knob
// l2_window = hitRatio; <= 0 = off). The persisting carve-out (a device-wide limit) is set to min(bytes, the device
// maximum) and left in place; the stream's window is cleared again on destruction so other kernels on the stream
// are not affected.
struct L2Window {
    cudaStream_t stream;
    bool on = false;
    L2Window(cudaStream_t s, const void* base, size_t bytes, double ratio) : stream(s) {
        if (ratio <= 0.0) return;
        TORCH_CHECK(ratio <= 1.0, "l2_window (hitRatio) must be in (0, 1]");
        int dev = 0, max_persist = 0, max_win = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_win, cudaDevAttrMaxAccessPolicyWindowSize, dev));
        TORCH_CHECK(max_persist > 0 && max_win > 0, "l2_window: this device has no persisting L2 / access-policy window");
        const size_t want = std::min(bytes, (size_t)max_persist);
        size_t cur = 0;
        C10_CUDA_CHECK(cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize));
        if (cur != want) C10_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want));
        cudaStreamAttrValue a{};
        a.accessPolicyWindow.base_ptr = const_cast<void*>(base);
        a.accessPolicyWindow.num_bytes = std::min(bytes, (size_t)max_win);
        a.accessPolicyWindow.hitRatio = float(ratio);
        a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        C10_CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &a));
        on = true;
    }
    ~L2Window() {
        if (!on) return;
        cudaStreamAttrValue a{};
        a.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &a);
    }
};

}  // namespace

std::vector<int64_t> l2_info() {
    int dev = 0, l2 = 0, max_persist = 0, max_win = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_win, cudaDevAttrMaxAccessPolicyWindowSize, dev));
    return {l2, max_persist, max_win};
}

// Undo everything an l2_window leaves behind: persisting lines stay resident (and the carve-out stays reserved)
// after the stream window is cleared, shrinking the normal L2 for every later kernel. Synchronises the device.
void l2_reset() {
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    C10_CUDA_CHECK(cudaCtxResetPersistingL2Cache());
    C10_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
}

torch::Tensor confidence_fwd(torch::Tensor z, torch::Tensor anc_a, c10::optional<torch::Tensor> anc_b,
                             torch::Tensor W, c10::optional<torch::Tensor> keep, double keep_scale,
                             torch::Tensor log_beta, torch::Tensor log_gamma, torch::Tensor log_tau,
                             int64_t nap, double eps, int64_t read_top_n,
                             int64_t threads, int64_t rows_per_cta, int64_t vec, int64_t l2_hint, double l2_window) {
    const c10::cuda::CUDAGuard guard(z.device());
    Params p = make_params(z, anc_a, anc_b, W, keep, keep_scale, log_beta, log_gamma, log_tau, nap, eps, rows_per_cta);
    check_launch(threads, vec, p.d_out);
    auto out = torch::empty({p.B, p.G, p.d_out}, z.options());
    if (p.B == 0) return out;
    const int nstripe = threads / (p.d_out / vec);
    const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, false);
    const int grid = p.G * ((p.B + rows_per_cta - 1) / rows_per_cta);
    auto stream = at::cuda::getCurrentCUDAStream();
    const L2Window win(stream, W.data_ptr(), W.nbytes(), l2_window);
    DISPATCH_HINT(l2_hint, [&] {
        DISPATCH_N_VEC(read_top_n, vec, [&] {
            auto k = confidence_fwd_kernel<kN, kV, kH>;
            if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
            k<<<grid, threads, L.total, stream>>>(p, out.data_ptr<float>());
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> confidence_bwd(torch::Tensor go, torch::Tensor z, torch::Tensor anc_a,
                                          c10::optional<torch::Tensor> anc_b, torch::Tensor W,
                                          c10::optional<torch::Tensor> keep, double keep_scale,
                                          torch::Tensor log_beta, torch::Tensor log_gamma, torch::Tensor log_tau,
                                          int64_t nap, double eps, int64_t read_top_n,
                                          int64_t threads, int64_t rows_per_cta, int64_t vec, bool vec_atomics,
                                          int64_t l2_hint, double l2_window, int64_t l2_window_target) {
    const c10::cuda::CUDAGuard guard(z.device());
    Params p = make_params(z, anc_a, anc_b, W, keep, keep_scale, log_beta, log_gamma, log_tau, nap, eps, rows_per_cta);
    check_launch(threads, vec, p.d_out);
    TORCH_CHECK(go.is_contiguous() && go.scalar_type() == torch::kFloat32 && go.size(0) == p.B && go.size(1) == p.G
                && go.size(2) == p.d_out, "go: contiguous fp32 [B, G, d_out]");
    TORCH_CHECK(l2_window_target == 0 || l2_window_target == 1, "l2_window_target must be 0 (W) or 1 (grad W)");
    auto gW = torch::zeros_like(W);
    auto gz = torch::empty_like(z);
    const int grid = p.G * ((p.B + rows_per_cta - 1) / rows_per_cta);
    auto gscal = torch::zeros({std::max(grid, 1), 3}, z.options());
    if (p.B == 0) return {gW, gz, gscal};
    const int nstripe = threads / (p.d_out / vec);
    const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, true);
    auto stream = at::cuda::getCurrentCUDAStream();
    const L2Window win(stream, l2_window_target == 0 ? W.data_ptr() : gW.data_ptr(), W.nbytes(), l2_window);
    DISPATCH_HINT(l2_hint, [&] {
        DISPATCH_N_VEC(read_top_n, vec, [&] {
            auto k = confidence_bwd_kernel<kN, kV, kH>;
            if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
            k<<<grid, threads, L.total, stream>>>(p, go.data_ptr<float>(), gW.data_ptr<float>(), gz.data_ptr<float>(),
                                                  gscal.data_ptr<float>(), vec_atomics);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gW, gz, gscal};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("confidence_fwd", &confidence_fwd, "ConfidenceLUT fused forward (CUDA)");
    m.def("confidence_bwd", &confidence_bwd, "ConfidenceLUT fused backward (CUDA)");
    m.def("l2_reset", &l2_reset, "sync, reset persisting L2 lines, set the persisting carve-out to 0");
    m.def("l2_info", &l2_info, "[L2 bytes, max persisting L2 bytes, max access-policy window bytes]");
}
