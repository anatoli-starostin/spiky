// FusedManifestoHardLUT / FusedManifestoSoftLUT: hand-written CUDA forward + backward for the Gen-1 Manifesto read,
// built the way fused_confidence.cu is (one forward and one backward kernel, nothing per-table in global memory).
//
// The Manifesto math (manifesto_hard.py, manifesto_soft.py, uncertainty.py), per (b, g, t), with the table-dropout
// factor m (the [B, G, tph] mask ManifestoLUT._table_dropout_mask returns, 0 or 1/keep_prob; 1 without dropout):
//   margins u_j = z[a_j] - z[b_j] (single anchors: z[a_j]); address c = MSB-first [u_j > eps]; j* = first argmin |u_j|;
//   neighbour c' = c with bit j* flipped; U = 0.5 / (1 + |u_j*|).
//   HARD  value  m W[c];                grad W[c] += m go;                  (no table gradient for c')
//   SOFT  value  m ((1-U) W[c] + U W[c']); grad W[c] += m (1-U) go, grad W[c'] += m U go.
//   Both: d value~ / d u_j* = m (W[c] - W[c']) . go * 0.5 sign(u_j*) / (1 + |u_j*|)^2, scattered +to a_j*, -to b_j*
//   (single anchors: +to a_j* only). For HARD this is the straight-through surrogate (the gradient of the blend whose
//   value is replaced by the hard read); for SOFT it is the exact derivative of the blend.
//
// One CTA owns one group g and `rows_per_cta` consecutive samples b of it. Per (b, g) row:
//   phase A  (thread per table)  margins from the z row in shared memory, c, j*, c', U, the read weights;
//   phase B  (stripe x chunk)    forward: the weighted read sum_t w W[c] (+ w' W[c']); backward: the weight-gradient
//                                scatter and the dot products W[c] . go, W[c'] . go, VEC elements per load;
//   phase C                      forward: reduce the stripes into the output row; backward: the margin gradient of
//                                each table's j* pair into the z row (shared memory), written once.
// The backward recomputes the addressing from z instead of saving anything per table.
//
// Table dtype T: fp32 or bf16 (upconverted in registers). Every reduction, the output and the weight-gradient buffer
// are fp32 (the caller casts the gradient to the table dtype once).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int MAX_NAP = 16;
constexpr int HARD = 0;
constexpr int SOFT = 1;

struct Params {
    const float* z;          // [B, G, d_in], fp32 input (nullptr when the input is bf16)
    const __nv_bfloat16* zb; // the same, bf16 input, upconverted when staged (nullptr when fp32)
    const int16_t* anc_a;    // [G, tph, nap]
    const int16_t* anc_b;    // [G, tph, nap], nullptr in single-anchor mode
    const float* W;          // [G * tph * K, d_out], fp32 table (nullptr when the table is bf16)
    const __nv_bfloat16* Wb; // the same, bf16 table (nullptr when the table is fp32)
    const float* mask;       // [B, G, tph] table-dropout factor (ManifestoLUT._table_dropout_mask), nullptr = none
    float eps;
    int B, G, tph, nap, d_in, d_out, rows_per_cta;
};

__host__ __device__ inline size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

// Shared-memory layout, identical on host (size) and device (pointers).
struct Smem {
    size_t zs, gos, gzs, w1, w2, cs, cas, rc, ra, cf, ia, ib, part, sa, sb, total;
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
        cf = o;   o = align16(o + (bwd ? sizeof(float) * tph : 0));    // m * 0.5 sign(u*) / (1 + |u*|)^2
        ia = o;   o = align16(o + (bwd ? sizeof(int) * tph : 0));      // anchor coordinates of the j* pair
        ib = o;   o = align16(o + (bwd ? sizeof(int) * tph : 0));
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

template <int VEC>
__device__ inline void load_vec(const __nv_bfloat16* p, float (&v)[VEC]) {
    static_assert(VEC == 1 || VEC == 2 || VEC == 4 || VEC == 8, "bf16 vec must be 1, 2, 4 or 8");
    if constexpr (VEC == 1) {
        v[0] = __bfloat162float(__ushort_as_bfloat16(__ldg(reinterpret_cast<const unsigned short*>(p))));
    } else {
        using Raw = std::conditional_t<VEC == 2, unsigned int, std::conditional_t<VEC == 4, uint2, uint4>>;
        const Raw x = __ldg(reinterpret_cast<const Raw*>(p));
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
        for (int k = 0; k < VEC / 2; ++k) {
            const float2 f = __bfloat1622float2(h[k]);
            v[2 * k] = f.x;
            v[2 * k + 1] = f.y;
        }
    }
}

// The input row is staged once per row into fp32 shared memory; a bf16 input is upconverted there (no fp32 copy of
// the input in global memory). The branch is uniform across the CTA.
__device__ inline float load_z(const Params& p, size_t i) {
    return p.zb ? __bfloat162float(p.zb[i]) : p.z[i];
}

template <typename T> __device__ inline const T* table(const Params& p);
template <> __device__ inline const float* table<float>(const Params& p) { return p.W; }
template <> __device__ inline const __nv_bfloat16* table<__nv_bfloat16>(const Params& p) { return p.Wb; }

template <int VEC>
__device__ inline void atomic_add_vec(float* dst, const float (&v)[VEC], bool vec_atomics) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (vec_atomics) {
        if constexpr (VEC == 8) {
            atomicAdd(reinterpret_cast<float4*>(dst), make_float4(v[0], v[1], v[2], v[3]));
            atomicAdd(reinterpret_cast<float4*>(dst + 4), make_float4(v[4], v[5], v[6], v[7]));
            return;
        }
        if constexpr (VEC == 4) { atomicAdd(reinterpret_cast<float4*>(dst), make_float4(v[0], v[1], v[2], v[3])); return; }
        if constexpr (VEC == 2) { atomicAdd(reinterpret_cast<float2*>(dst), make_float2(v[0], v[1])); return; }
    }
#endif
#pragma unroll
    for (int k = 0; k < VEC; ++k) atomicAdd(dst + k, v[k]);
}

// Phase A, shared by forward and backward: one table's address, neighbour and read weights. When cf != nullptr
// (backward) also the margin-gradient coefficient and the anchor coordinates of the j* pair.
template <int MODE>
__device__ inline void address_table(const Params& p, int g, int b, int t, const float* zs, const int16_t* sa,
                                     const int16_t* sb, int* cs, int* cas, float* w1, float* w2,
                                     float* cf, int* ia, int* ib) {
    const int nap = p.nap;
    float mstar = INFINITY, ustar = 0.f;
    int c = 0, jstar = 0;
#pragma unroll
    for (int j = 0; j < MAX_NAP; ++j) {
        if (j < nap) {
            const int i = t * nap + j;
            const float u = zs[sa[i]] - (sb ? zs[sb[i]] : 0.f);
            c |= int(u > p.eps) << (nap - 1 - j);
            const float m = fabsf(u);
            if (m < mstar) { mstar = m; ustar = u; jstar = j; }   // first minimum, as torch.min
        }
    }
    const float mk = p.mask ? p.mask[((size_t)b * p.G + g) * p.tph + t] : 1.f;
    const int base = (g * p.tph + t) << nap;
    cs[t] = base + c;
    cas[t] = base + (c ^ (1 << (nap - 1 - jstar)));
    const float U = 0.5f / (1.f + mstar);
    if constexpr (MODE == SOFT) {
        w1[t] = mk * (1.f - U);
        w2[t] = mk * U;
    } else {
        w1[t] = mk;
        w2[t] = 0.f;
    }
    if (cf) {
        const float q = 1.f + mstar;
        cf[t] = mk * 0.5f * float((ustar > 0.f) - (ustar < 0.f)) / (q * q);   // d|u|/du = sign(u), 0 at 0
        const int i = t * nap + jstar;
        ia[t] = sa[i];
        ib[t] = sb ? sb[i] : -1;
    }
}

template <int MODE, int VEC, typename T>
__global__ void manifesto_fwd_kernel(Params p, float* __restrict__ out, __nv_bfloat16* __restrict__ outb) {
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
    const int b0 = static_cast<int>((static_cast<int64_t>(blockIdx.x) / p.G) * p.rows_per_cta);
    const int an = p.tph * p.nap;
    for (int i = tid; i < an; i += bd) {                 // this group's anchors, once per CTA
        sa[i] = p.anc_a[(size_t)g * an + i];
        if (sb) sb[i] = p.anc_b[(size_t)g * an + i];
    }
    const int stripe = tid / nchunk, chunk = tid % nchunk;

    for (int r = 0; r < p.rows_per_cta; ++r) {
        const int b = b0 + r;
        if (b >= p.B) break;                              // uniform across the CTA
        const size_t row = (size_t)b * p.G + g;
        for (int i = tid; i < p.d_in; i += bd) zs[i] = load_z(p, row * p.d_in + i);
        __syncthreads();
        for (int t = tid; t < p.tph; t += bd)
            address_table<MODE>(p, g, b, t, zs, sa, sb, cs, cas, w1, w2, nullptr, nullptr, nullptr);
        __syncthreads();
        if (stripe < nstripe) {
            float acc[VEC] = {};
            for (int t = stripe; t < p.tph; t += nstripe) {
                float v[VEC];
                const float a = w1[t];
                if (MODE == SOFT || a != 0.f) {          // a dropped table (m = 0) reads nothing in hard mode
                    load_vec<VEC>(table<T>(p) + (size_t)cs[t] * p.d_out + chunk * VEC, v);
#pragma unroll
                    for (int k = 0; k < VEC; ++k) acc[k] = fmaf(a, v[k], acc[k]);
                }
                if constexpr (MODE == SOFT) {
                    load_vec<VEC>(table<T>(p) + (size_t)cas[t] * p.d_out + chunk * VEC, v);
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
            if (outb) outb[row * p.d_out + i] = __float2bfloat16(y);   // bf16-output mode: one RNE rounding at the store
            else out[row * p.d_out + i] = y;
        }
        __syncthreads();                                  // part / zs reused by the next row
    }
}

template <int MODE, int VEC, typename T>
__global__ void manifesto_bwd_kernel(Params p, const float* __restrict__ go, float* __restrict__ gW,
                                     float* __restrict__ gz, __nv_bfloat16* __restrict__ gzb, bool vec_atomics) {
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
    float* cf = reinterpret_cast<float*>(smem_raw + L.cf);
    int* ia = reinterpret_cast<int*>(smem_raw + L.ia);
    int* ib = reinterpret_cast<int*>(smem_raw + L.ib);
    int16_t* sa = reinterpret_cast<int16_t*>(smem_raw + L.sa);
    int16_t* sb = p.anc_b ? reinterpret_cast<int16_t*>(smem_raw + L.sb) : nullptr;

    const int tid = threadIdx.x, bd = blockDim.x;
    const int g = blockIdx.x % p.G;
    const int b0 = static_cast<int>((static_cast<int64_t>(blockIdx.x) / p.G) * p.rows_per_cta);
    const int an = p.tph * p.nap;
    for (int i = tid; i < an; i += bd) {
        sa[i] = p.anc_a[(size_t)g * an + i];
        if (sb) sb[i] = p.anc_b[(size_t)g * an + i];
    }
    const int stripe = tid / nchunk, chunk = tid % nchunk;

    for (int r = 0; r < p.rows_per_cta; ++r) {
        const int b = b0 + r;
        if (b >= p.B) break;
        const size_t row = (size_t)b * p.G + g;
        for (int i = tid; i < p.d_in; i += bd) { zs[i] = load_z(p, row * p.d_in + i); gzs[i] = 0.f; }
        for (int i = tid; i < p.d_out; i += bd) gos[i] = go[row * p.d_out + i];
        __syncthreads();
        for (int t = tid; t < p.tph; t += bd) {
            address_table<MODE>(p, g, b, t, zs, sa, sb, cs, cas, w1, w2, cf, ia, ib);
            rc[t] = 0.f;
            ra[t] = 0.f;
        }
        __syncthreads();
        // Phase B: weight-gradient scatter and the dot products W[c] . go, W[c'] . go (both modes need both: the
        // margin gradient is (W[c] - W[c']) . go). A dropped table (m = 0) contributes nothing and is skipped.
        if (stripe < nstripe) {
            float gv[VEC];
#pragma unroll
            for (int k = 0; k < VEC; ++k) gv[k] = gos[chunk * VEC + k];
            for (int t = stripe; t < p.tph; t += nstripe) {
                if (cf[t] == 0.f && w1[t] == 0.f) continue;   // dropped (or no gradient at all for this table)
                float v[VEC], d[VEC];
                const size_t off = (size_t)cs[t] * p.d_out + chunk * VEC;
                load_vec<VEC>(table<T>(p) + off, v);
                float dot = 0.f;
#pragma unroll
                for (int k = 0; k < VEC; ++k) { dot = fmaf(v[k], gv[k], dot); d[k] = w1[t] * gv[k]; }
                if (w1[t] != 0.f) atomic_add_vec<VEC>(gW + off, d, vec_atomics);
                const size_t off2 = (size_t)cas[t] * p.d_out + chunk * VEC;
                load_vec<VEC>(table<T>(p) + off2, v);
                float dot2 = 0.f;
#pragma unroll
                for (int k = 0; k < VEC; ++k) { dot2 = fmaf(v[k], gv[k], dot2); d[k] = w2[t] * gv[k]; }
                if constexpr (MODE == SOFT) {
                    if (w2[t] != 0.f) atomic_add_vec<VEC>(gW + off2, d, vec_atomics);
                }
                atomicAdd(&rc[t], dot);
                atomicAdd(&ra[t], dot2);
            }
        }
        __syncthreads();
        // Phase C: the j* margin gradient of each table into the z row.
        for (int t = tid; t < p.tph; t += bd) {
            const float gu = cf[t] * (rc[t] - ra[t]);
            if (gu != 0.f) {
                atomicAdd(&gzs[ia[t]], gu);
                if (ib[t] >= 0) atomicAdd(&gzs[ib[t]], -gu);
            }
        }
        __syncthreads();
        for (int i = tid; i < p.d_in; i += bd) {           // bf16 input: grad z stored in bf16, one RNE rounding
            if (gzb) gzb[row * p.d_in + i] = __float2bfloat16(gzs[i]);
            else gz[row * p.d_in + i] = gzs[i];
        }
        __syncthreads();
    }
}

Params make_params(const torch::Tensor& z, const torch::Tensor& anc_a, const c10::optional<torch::Tensor>& anc_b,
                   const torch::Tensor& W, const c10::optional<torch::Tensor>& mask,
                   int nap, double eps, int rows_per_cta) {
    TORCH_CHECK(z.is_cuda() && (z.scalar_type() == torch::kFloat32 || z.scalar_type() == torch::kBFloat16) && z.is_contiguous()
                && z.dim() == 3, "z: contiguous fp32 or bf16 CUDA [B, G, d_in]");
    TORCH_CHECK((W.scalar_type() == torch::kFloat32 || W.scalar_type() == torch::kBFloat16) && W.is_contiguous() && W.dim() == 2,
                "W: contiguous fp32 or bf16 [rows, d_out]");
    TORCH_CHECK(anc_a.scalar_type() == torch::kInt16 && anc_a.is_contiguous() && anc_a.dim() == 3, "anchors: contiguous int16 [G, tph, nap]");
    if (anc_b.has_value())
        TORCH_CHECK(anc_b->scalar_type() == torch::kInt16 && anc_b->is_contiguous() && anc_b->sizes() == anc_a.sizes(),
                    "anchor_b: contiguous int16, same shape as anchor_a");
    TORCH_CHECK(nap >= 1 && nap <= MAX_NAP, "nap must be in [1, 16]");
    TORCH_CHECK(rows_per_cta >= 1, "rows_per_cta must be >= 1");
    Params p{};
    const bool zbf16 = z.scalar_type() == torch::kBFloat16;
    p.z = zbf16 ? nullptr : z.data_ptr<float>();
    p.zb = zbf16 ? reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()) : nullptr;
    p.anc_a = anc_a.data_ptr<int16_t>();
    p.anc_b = anc_b.has_value() ? anc_b->data_ptr<int16_t>() : nullptr;
    const bool bf16 = W.scalar_type() == torch::kBFloat16;
    p.W = bf16 ? nullptr : W.data_ptr<float>();
    p.Wb = bf16 ? reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()) : nullptr;
    if (mask.has_value())
        TORCH_CHECK(mask->scalar_type() == torch::kFloat32 && mask->is_contiguous() && mask->numel() == z.size(0) * z.size(1) * anc_a.size(1),
                    "mask: contiguous fp32 [B, G, tph]");
    p.mask = mask.has_value() ? mask->data_ptr<float>() : nullptr;
    p.eps = float(eps);
    p.B = z.size(0); p.G = z.size(1); p.d_in = z.size(2);
    p.tph = anc_a.size(1); p.nap = nap; p.d_out = W.size(1);
    p.rows_per_cta = rows_per_cta;
    TORCH_CHECK(anc_a.size(0) == p.G && anc_a.size(2) == nap, "anchor shape mismatch");
    TORCH_CHECK(W.size(0) == (int64_t)p.G * p.tph << nap, "W rows != G * tph * 2**nap");
    TORCH_CHECK(((int64_t)p.G * p.tph << nap) <= INT32_MAX, "flat cell index must fit int32");
    return p;
}

void check_launch(int threads, int vec, int d_out, bool bf16) {
    TORCH_CHECK(vec == 1 || vec == 2 || vec == 4 || (bf16 && vec == 8),
                bf16 ? "vec must be 1, 2, 4 or 8 for a bf16 table" : "vec must be 1, 2 or 4 for an fp32 table");
    TORCH_CHECK(d_out % vec == 0, "d_out must be divisible by vec");
    TORCH_CHECK(threads % 32 == 0 && threads >= 32 && threads <= 1024, "threads must be a multiple of 32 in [32, 1024]");
    TORCH_CHECK(threads >= d_out / vec, "threads must be >= d_out / vec (one stripe at least)");
}

#define DISPATCH_MODE_VEC(MODE, VEC, ...)                                                       \
    [&] {                                                                                       \
        if (MODE == HARD && VEC == 4) { constexpr int kM = HARD, kV = 4; __VA_ARGS__(); }       \
        else if (MODE == HARD && VEC == 2) { constexpr int kM = HARD, kV = 2; __VA_ARGS__(); }  \
        else if (MODE == HARD && VEC == 1) { constexpr int kM = HARD, kV = 1; __VA_ARGS__(); }  \
        else if (MODE == HARD && VEC == 8) { constexpr int kM = HARD, kV = 8; __VA_ARGS__(); }  \
        else if (MODE == SOFT && VEC == 4) { constexpr int kM = SOFT, kV = 4; __VA_ARGS__(); }  \
        else if (MODE == SOFT && VEC == 2) { constexpr int kM = SOFT, kV = 2; __VA_ARGS__(); }  \
        else if (MODE == SOFT && VEC == 1) { constexpr int kM = SOFT, kV = 1; __VA_ARGS__(); }  \
        else if (MODE == SOFT && VEC == 8) { constexpr int kM = SOFT, kV = 8; __VA_ARGS__(); }  \
        else TORCH_CHECK(false, "mode must be 0 (hard) or 1 (soft)");                           \
    }()

template <typename Fn>
void dispatch_table(const torch::Tensor& W, Fn&& fn) {
    if (W.scalar_type() == torch::kBFloat16) fn(__nv_bfloat16{});
    else fn(float{});
}

}  // namespace

torch::Tensor manifesto_fwd(torch::Tensor z, torch::Tensor anc_a, c10::optional<torch::Tensor> anc_b,
                            torch::Tensor W, c10::optional<torch::Tensor> mask,
                            int64_t nap, double eps, int64_t mode,
                            int64_t threads, int64_t rows_per_cta, int64_t vec, bool out_bf16) {
    const c10::cuda::CUDAGuard guard(z.device());
    Params p = make_params(z, anc_a, anc_b, W, mask, nap, eps, rows_per_cta);
    check_launch(threads, vec, p.d_out, W.scalar_type() == torch::kBFloat16);
    // fp32 output, or bf16 when the caller asks (only for a bf16 input whose groups map 1:1 onto output heads).
    TORCH_CHECK(!out_bf16 || z.scalar_type() == torch::kBFloat16, "out_bf16 needs a bf16 input");
    auto out = torch::empty({p.B, p.G, p.d_out}, z.options().dtype(out_bf16 ? torch::kBFloat16 : torch::kFloat32));
    if (p.B == 0) return out;
    const int nstripe = threads / (p.d_out / vec);
    const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, false);
    const int64_t grid = (int64_t)p.G * ((p.B + rows_per_cta - 1) / rows_per_cta);
    TORCH_CHECK(grid <= INT32_MAX, "grid too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    dispatch_table(W, [&](auto tag) {
        using T = decltype(tag);
        DISPATCH_MODE_VEC(mode, vec, [&] {
            if constexpr (kV == 8 && std::is_same_v<T, float>) {
                TORCH_CHECK(false, "vec 8 is bf16-only");
            } else {
                auto k = manifesto_fwd_kernel<kM, kV, T>;
                if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
                k<<<(unsigned)grid, threads, L.total, stream>>>(
                    p, out_bf16 ? nullptr : out.data_ptr<float>(),
                    out_bf16 ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr);
            }
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> manifesto_bwd(torch::Tensor go, torch::Tensor z, torch::Tensor anc_a,
                                         c10::optional<torch::Tensor> anc_b, torch::Tensor W,
                                         c10::optional<torch::Tensor> mask,
                                         int64_t nap, double eps, int64_t mode,
                                         int64_t threads, int64_t rows_per_cta, int64_t vec, bool vec_atomics) {
    const c10::cuda::CUDAGuard guard(z.device());
    Params p = make_params(z, anc_a, anc_b, W, mask, nap, eps, rows_per_cta);
    check_launch(threads, vec, p.d_out, W.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(go.is_contiguous() && go.scalar_type() == torch::kFloat32 && go.size(0) == p.B && go.size(1) == p.G
                && go.size(2) == p.d_out, "go: contiguous fp32 [B, G, d_out]");
    auto gW = torch::zeros(W.sizes(), W.options().dtype(torch::kFloat32));
    const bool zbf16 = z.scalar_type() == torch::kBFloat16;   // grad z in the input dtype (one rounding in-kernel)
    auto gz = torch::empty(z.sizes(), z.options().dtype(zbf16 ? torch::kBFloat16 : torch::kFloat32));
    if (p.B == 0) return {gW, gz};
    const int nstripe = threads / (p.d_out / vec);
    const Smem L(p.tph, p.nap, p.d_in, p.d_out, nstripe, true);
    const int64_t grid = (int64_t)p.G * ((p.B + rows_per_cta - 1) / rows_per_cta);
    TORCH_CHECK(grid <= INT32_MAX, "grid too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    dispatch_table(W, [&](auto tag) {
        using T = decltype(tag);
        DISPATCH_MODE_VEC(mode, vec, [&] {
            if constexpr (kV == 8 && std::is_same_v<T, float>) {
                TORCH_CHECK(false, "vec 8 is bf16-only");
            } else {
                auto k = manifesto_bwd_kernel<kM, kV, T>;
                if (L.total > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.total));
                k<<<(unsigned)grid, threads, L.total, stream>>>(p, go.data_ptr<float>(), gW.data_ptr<float>(),
                                                                 zbf16 ? nullptr : gz.data_ptr<float>(),
                                                                 zbf16 ? reinterpret_cast<__nv_bfloat16*>(gz.data_ptr()) : nullptr,
                                                                 vec_atomics);
            }
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gW, gz};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("manifesto_fwd", &manifesto_fwd, "Manifesto (hard / soft) fused forward (CUDA)");
    m.def("manifesto_bwd", &manifesto_bwd, "Manifesto (hard / soft) fused backward (CUDA)");
}
