// Cell-stationary ("TS at register granularity") LUT inference kernels, plus the
// output-stationary gather baseline they are measured against.
//
// The layer, for all kernels here:
//
//     y[b,k] = sum_{t=0}^{NT-1} c[b,t] * T[t, j[b,t], k],     k = 0..N-1
//
// with int8 tables T of shape [NT, R, N] (a row is N contiguous bytes), uint8
// indices j [B, NT], fp32 coefficients c [B, NT], fp32 output y [B, N].
// Accumulation is fp32 throughout. `use_coef=0` is the c == 1 mode (which is what
// the repo's trained checkpoints actually run at inference: forward_mode="hard").
//
// THREE KERNELS
//
// 1. gather  -- the baseline. Output-stationary: a thread owns 4 output lanes of one
//    token and loops over the NT tables, reading one 4-byte chunk of each selected
//    row and accumulating in registers. One write per (token, lane), no atomics.
//    This is the generalisation of experiments/ffn_replacement/benchmark/gather_cuda.cu
//    (which is hardcoded to D=48 fp32/bf16) to arbitrary N and int8 tables.
//
// 2. cs_v0 -- the naive cell-stationary kernel. A thread owns exactly one cell slice
//    (table t, row r, K lanes). It loads its K table bytes once and then scans M
//    tokens per tile, adding into register accumulators acc[M][K] on a hit
//    (j[b,t] == r), and flushes with one atomicAdd per (token, lane) touched.
//
// 3. cs_v1 -- the strengthened cell-stationary kernel. Same ownership, plus:
//      * wide lane groups K (8..64) and tiny minibatch M (1..4), so the per-token
//        compare is amortised over K useful adds instead of 1;
//      * K table bytes held packed as K/4 uint32 registers;
//      * a block of 256*TB threads covers TB tables x 256 rows, so the TB partial
//        contributions a token receives inside a block are reduced in shared memory
//        and leave as ONE vector atomic per (token, lane group) instead of TB;
//      * optionally a thread-block cluster of CLU blocks extends that reduction
//        across CLU*TB tables through distributed shared memory;
//      * float4 atomics (or 4 scalar ones if the vector form is unavailable);
//      * the j/c tile staged in shared memory once per block per token tile.
//
// WHAT "PERSISTENT" MEANS HERE. Both cell-stationary kernels are launched ONCE over
// the whole (table, row, lane-group) grid and stream the entire batch inside the
// kernel, so a thread reads its table bytes exactly once per launch. The grid is far
// larger than what fits resident, so the hardware runs it in waves -- which is
// exactly the "C column passes" of the design, with C chosen by the scheduler
// instead of by hand. Table traffic is one pass over the table set either way; the
// wave count is reported by the harness as grid_blocks / resident_blocks.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace {

#define DEVI __device__ __forceinline__

// ---------------------------------------------------------------- small helpers

// 4 packed int8 -> 4 floats. The cast through int8_t is what sign-extends; going
// through uint8 and subtracting 256 conditionally is measurably worse.
DEVI void unpack4(uint32_t w, float v[4]) {
  v[0] = (float)(int8_t)(w & 0xffu);
  v[1] = (float)(int8_t)((w >> 8) & 0xffu);
  v[2] = (float)(int8_t)((w >> 16) & 0xffu);
  v[3] = (float)(int8_t)((w >> 24) & 0xffu);
}

// One 16-byte atomic add if the hardware/toolkit has the vector form, else four
// scalar ones. LAC_VEC_ATOMIC is defined by the build script after a probe compile.
DEVI void atomic_add4(float* p, float a, float b, float c, float d) {
#ifdef LAC_VEC_ATOMIC
  atomicAdd(reinterpret_cast<float4*>(p), make_float4(a, b, c, d));
#else
  atomicAdd(p + 0, a);
  atomicAdd(p + 1, b);
  atomicAdd(p + 2, c);
  atomicAdd(p + 3, d);
#endif
}

// ---------------------------------------------------------------- 1. baseline gather
//
// grid  = (ceil(B / TOK_PER_BLK), tsplit)
// block = (LANE_WORKERS, TOK_PER_BLK)
//
// tsplit > 1 splits the table range across blocks to get parallelism at small B; it
// costs atomics, so the harness sweeps it and reports the best per B (the baseline
// is tuned, not handicapped).

__global__ void gather_kernel(
    const int8_t* __restrict__ T, const uint8_t* __restrict__ J,
    const float* __restrict__ C, float* __restrict__ Y,
    int B, int NT, int R, int N4, int tsplit, int use_coef) {
  const int tok = blockIdx.x * blockDim.y + threadIdx.y;
  if (tok >= B) return;
  const int per = NT / tsplit;
  const int t0 = blockIdx.y * per;
  const int t1 = t0 + per;
  const uint32_t* __restrict__ T32 = reinterpret_cast<const uint32_t*>(T);
  const uint8_t* __restrict__ Jt = J + (size_t)tok * NT;
  const float* __restrict__ Ct = C + (size_t)tok * NT;

  for (int u = threadIdx.x; u < N4; u += blockDim.x) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 4
    for (int t = t0; t < t1; ++t) {
      const int r = Jt[t];
      const uint32_t w = T32[((size_t)t * R + r) * N4 + u];
      const float cc = use_coef ? Ct[t] : 1.0f;
      float v[4];
      unpack4(w, v);
      a0 = fmaf(cc, v[0], a0);
      a1 = fmaf(cc, v[1], a1);
      a2 = fmaf(cc, v[2], a2);
      a3 = fmaf(cc, v[3], a3);
    }
    float* yp = Y + (size_t)tok * 4 * N4 + 4 * u;
    if (tsplit == 1) {
      *reinterpret_cast<float4*>(yp) = make_float4(a0, a1, a2, a3);
    } else {
      atomic_add4(yp, a0, a1, a2, a3);
    }
  }
}

// ---------------------------------------------------------------- 2. cs_v0 (naive)
//
// grid  = (NT, N / K)          block = (R)   [R must be 256]
// thread (r) owns cell slice (t = blockIdx.x, r, lanes [blockIdx.y*K, +K)).
//
// acc[M][K] is kept exactly as specified even though, for a single (t, r), a token
// can contribute at most once -- so the accumulator never actually accumulates. That
// is the point of measuring it: the register budget it forces is paid for nothing.

template <int K, int M>
__global__ __launch_bounds__(256) void cs_v0_kernel(
    const int8_t* __restrict__ T, const uint8_t* __restrict__ J,
    const float* __restrict__ C, float* __restrict__ Y,
    int B, int NT, int R, int N, int use_coef) {
  const int r = threadIdx.x;
  const int t = blockIdx.x;
  const int lane0 = blockIdx.y * K;

  float val[K];
  {
    const int8_t* row = T + ((size_t)t * R + r) * N + lane0;
#pragma unroll
    for (int k = 0; k < K; ++k) val[k] = (float)row[k];
  }

  __shared__ uint8_t js[M];
  __shared__ float cs[M];

  for (int b0 = 0; b0 < B; b0 += M) {
    for (int m = threadIdx.x; m < M; m += 256) {
      const int b = b0 + m;
      js[m] = (b < B) ? J[(size_t)b * NT + t] : (uint8_t)0;
      cs[m] = (b < B) ? (use_coef ? C[(size_t)b * NT + t] : 1.0f) : 0.0f;
    }
    __syncthreads();

    // The K-wide work is BRANCHED on the hit, not predicated. That distinction is the
    // whole reason K can pay for itself: predicating it costs K multiplies per
    // (thread, token) whether the row is selected or not, which makes the total work
    // NT*R*N*B independent of K -- measured, and it is why the first version of this
    // kernel got nothing at all out of widening the lane group. A real branch costs 1
    // compare per (thread, token) plus K work on the 1-in-R hits; a warp covers 32
    // consecutive rows, so the whole warp skips 224/256 of the time.
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if ((js[m] == (uint8_t)r) && (b0 + m < B)) {
        const float cc = cs[m];
        float* yp = Y + (size_t)(b0 + m) * N + lane0;
#pragma unroll
        for (int k = 0; k < K; ++k) atomicAdd(yp + k, cc * val[k]);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------- 3. cs_v1 (strengthened)
//
// grid  = (NT / (TB*CLU), N / K, CLU)   block = (256, TB)   cluster = (1,1,CLU)
// thread (r = threadIdx.x, tl = threadIdx.y) owns cell slice
//     (t = (blockIdx.x*CLU + blockIdx.z)*TB + tl,  r,  lanes [blockIdx.y*K, +K)).
//
// The block (and, with CLU>1, the cluster) reduces the TB*CLU partial contributions a
// token receives into one shared-memory buffer and issues ONE vector atomic per
// (token, 4 lanes) instead of TB*CLU of them.

template <int K, int M, int TB, int CLU>
__global__ __cluster_dims__(1, 1, CLU) __launch_bounds__(256 * TB) void cs_v1_kernel(
    const int8_t* __restrict__ T, const uint8_t* __restrict__ J,
    const float* __restrict__ C, float* __restrict__ Y,
    int B, int NT, int R, int N, int use_coef) {
  const int r = threadIdx.x;
  const int tl = threadIdx.y;
  const int tg = (blockIdx.x * CLU + blockIdx.z) * TB;
  const int t = tg + tl;
  const int lane0 = blockIdx.y * K;
  const int tid = tl * 256 + r;

  // K table bytes, packed K/4 to a register, loaded once and never re-read.
  uint32_t packed[K / 4];
  {
    const uint32_t* row =
        reinterpret_cast<const uint32_t*>(T + ((size_t)t * R + r) * N + lane0);
#pragma unroll
    for (int q = 0; q < K / 4; ++q) packed[q] = row[q];
  }

  __shared__ uint8_t js[M][TB];
  __shared__ float cs[M][TB];
  __shared__ float red[M][K];

  cg::cluster_group clu = cg::this_cluster();
  float* red_dst = &red[0][0];
  if (CLU > 1) red_dst = clu.map_shared_rank(&red[0][0], 0);
  const bool is_root = (CLU == 1) || (clu.block_rank() == 0);

  const int FLUSH = (M * K) / 4;   // float4 units to push to global per token tile

  for (int b0 = 0; b0 < B; b0 += M) {
    // --- stage the j/c tile for this block's TB tables -----------------------
    if (tid < M * TB) {
      const int m = tid / TB, q = tid % TB;
      const int b = b0 + m;
      js[m][q] = (b < B) ? J[(size_t)b * NT + tg + q] : (uint8_t)0;
      cs[m][q] = (b < B) ? (use_coef ? C[(size_t)b * NT + tg + q] : 1.0f) : 0.0f;
    }
    // --- zero the reduction buffer (root block's copy) -----------------------
    if (is_root) {
      for (int i = tid; i < M * K; i += 256 * TB) red[i / K][i % K] = 0.0f;
    }
    if (CLU > 1) clu.sync(); else __syncthreads();

    // --- the scan -------------------------------------------------------------
    // BRANCHED on the hit, not predicated. This is the single most important line in
    // the kernel and the first version got it wrong: unpacking and multiplying the K
    // lanes unconditionally costs K work per (thread, token), so the total becomes
    // NT*R*N*B and is INDEPENDENT of K -- widening the lane group then buys exactly
    // nothing, which is what the first measurement showed (K=8 and K=64 within noise
    // of each other). With a real branch the cost is 1 compare per (thread, token)
    // plus K work on the 1-in-R hits. threadIdx.x is the row, so a warp spans 32
    // consecutive rows and skips the body 224/256 of the time.
    //
    // There is no register accumulator here, and that is not an omission: a thread
    // owning ONE row of ONE table receives at most one contribution per token, so
    // acc[M][K] never accumulates anything. ptxas elided it in the predicated version
    // too (K=32,M=1 came out at 52 registers, not 52+32). The thing that actually
    // accumulates across tables is the shared buffer below.
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if ((js[m][tl] == (uint8_t)r) && (b0 + m < B)) {
        const float cc = cs[m][tl];
        float v[4];
#pragma unroll
        for (int q = 0; q < K / 4; ++q) {
          unpack4(packed[q], v);
          atomicAdd(red_dst + m * K + 4 * q + 0, cc * v[0]);
          atomicAdd(red_dst + m * K + 4 * q + 1, cc * v[1]);
          atomicAdd(red_dst + m * K + 4 * q + 2, cc * v[2]);
          atomicAdd(red_dst + m * K + 4 * q + 3, cc * v[3]);
        }
      }
    }
    if (CLU > 1) clu.sync(); else __syncthreads();

    // --- one vector atomic per (token, 4 lanes) for the whole block/cluster ---
    if (is_root) {
      for (int u = tid; u < FLUSH; u += 256 * TB) {
        const int m = (4 * u) / K, k = (4 * u) % K;
        const int b = b0 + m;
        if (b < B) {
          const float* s = &red[m][k];
          atomic_add4(Y + (size_t)b * N + lane0 + k, s[0], s[1], s[2], s[3]);
        }
      }
    }
    if (CLU > 1) clu.sync(); else __syncthreads();
  }
}

// ------------------------------------------------- 4. cs_v2 (conflict-free reduction)
//
// cs_v1's block reduction turned out to be 85% of its runtime (104.5 ms of 123.7 ms at
// B=24576, measured by ablation), and all of it was shared-memory atomics. Those atomics
// are unnecessary: threadIdx.x is the row, j[b,t] names exactly ONE row, so for each
// (table-in-block, token) there is exactly one thread that hits -- a unique writer. The
// reduction buffer can therefore be written with plain stores, needs no zeroing, and is
// summed over the TB tables by the flushing threads. That is the "cross-thread reduction"
// arm: TB contributions collapse to exactly one atomic, the full TB factor, not the
// birthday-limited factor an intra-thread version gets.
//
// TRANS selects the table layout: 0 = [NT, R, N] (as stored), 1 = [R, NT, N] (permuted
// offline), which makes the threads that reduce together adjacent in memory.

template <int K, int M, int TB, int TRANS>
__global__ __launch_bounds__(256 * TB) void cs_v2_kernel(
    const int8_t* __restrict__ T, const uint8_t* __restrict__ J,
    const float* __restrict__ C, float* __restrict__ Y,
    int B, int NT, int R, int N, int use_coef) {
  const int r = threadIdx.x;
  const int tl = threadIdx.y;
  const int tg = blockIdx.x * TB;
  const int t = tg + tl;
  const int lane0 = blockIdx.y * K;
  const int tid = tl * 256 + r;

  uint32_t packed[K / 4];
  {
    const size_t off = TRANS ? ((size_t)r * NT + t) * N + lane0
                             : ((size_t)t * R + r) * N + lane0;
    const uint32_t* row = reinterpret_cast<const uint32_t*>(T + off);
#pragma unroll
    for (int q = 0; q < K / 4; ++q) packed[q] = row[q];
  }

  __shared__ uint8_t js[M][TB];
  __shared__ float cs[M][TB];
  __shared__ float red[TB][M][K];      // unique writer per (tl, m): plain stores
  const int FLUSH = (M * K) / 4;

  for (int b0 = 0; b0 < B; b0 += M) {
    if (tid < M * TB) {
      const int m = tid / TB, q = tid % TB;
      const int b = b0 + m;
      js[m][q] = (b < B) ? J[(size_t)b * NT + tg + q] : (uint8_t)0;
      cs[m][q] = (b < B) ? (use_coef ? C[(size_t)b * NT + tg + q] : 1.0f) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if ((js[m][tl] == (uint8_t)r) && (b0 + m < B)) {
        const float cc = cs[m][tl];
        float v[4];
#pragma unroll
        for (int q = 0; q < K / 4; ++q) {
          unpack4(packed[q], v);
          red[tl][m][4 * q + 0] = cc * v[0];
          red[tl][m][4 * q + 1] = cc * v[1];
          red[tl][m][4 * q + 2] = cc * v[2];
          red[tl][m][4 * q + 3] = cc * v[3];
        }
      }
    }
    __syncthreads();
    // Strided, not `if (tid < FLUSH)`: M*K/4 can exceed the block size (e.g. K=128,
    // M=32, TB=1 needs 1024 units from 256 threads), and the guarded form silently
    // dropped the tail. Caught by the correctness gate, which is what it is for.
    for (int u = tid; u < FLUSH; u += 256 * TB) {
      const int m = (4 * u) / K, k = (4 * u) % K;
      const int b = b0 + m;
      if (b < B) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int q = 0; q < TB; ++q) {
          const float* s = &red[q][m][k];
          a0 += s[0]; a1 += s[1]; a2 += s[2]; a3 += s[3];
        }
        atomic_add4(Y + (size_t)b * N + lane0 + k, a0, a1, a2, a3);
      }
    }
    __syncthreads();
  }
}

// ------------------------------------------------- 5. cs_v3 (intra-thread G tables)
//
// A thread owns G tables at the SAME row r and the same K lanes, accumulates the hits
// among those G in registers, and emits one atomic group per (token, row) that had at
// least one hit. The dedup factor is hits / emitted groups, which index_stats.py
// measures on the real layer and which the birthday bound
// (G/R)/(1-(1-1/R)^G) predicts for independent indices.
//
// REGISTER ACCOUNTING, which is not what the spec assumed: the accumulator is acc[K],
// NOT acc[M][K]. A thread's contribution to token m is finished once it has scanned its
// own G tables for that token, so the accumulator is reused across m and M costs no
// registers at all. That is what lets M stay large enough to amortise the staging.
// The table values are the real register cost: G*K/4 packed words.

template <int K, int M, int G, int TRANS>
__global__ __launch_bounds__(256) void cs_v3_kernel(
    const int8_t* __restrict__ T, const uint8_t* __restrict__ J,
    const float* __restrict__ C, float* __restrict__ Y,
    int B, int NT, int R, int N, int use_coef) {
  const int r = threadIdx.x;
  const int t0 = blockIdx.x * G;
  const int lane0 = blockIdx.y * K;

  uint32_t packed[G][K / 4];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const size_t off = TRANS ? ((size_t)r * NT + t0 + g) * N + lane0
                             : ((size_t)(t0 + g) * R + r) * N + lane0;
    const uint32_t* row = reinterpret_cast<const uint32_t*>(T + off);
#pragma unroll
    for (int q = 0; q < K / 4; ++q) packed[g][q] = row[q];
  }

  __shared__ uint8_t js[M][G];
  __shared__ float cs[M][G];

  for (int b0 = 0; b0 < B; b0 += M) {
    for (int i = threadIdx.x; i < M * G; i += 256) {
      const int m = i / G, g = i % G;
      const int b = b0 + m;
      js[m][g] = (b < B) ? J[(size_t)b * NT + t0 + g] : (uint8_t)255;
      cs[m][g] = (b < B) ? (use_coef ? C[(size_t)b * NT + t0 + g] : 1.0f) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if (b0 + m < B) {
        float acc[K];
        bool any = false;
#pragma unroll
        for (int g = 0; g < G; ++g) {
          if (js[m][g] == (uint8_t)r) {
            const float cc = cs[m][g];
            float v[4];
            if (!any) {
#pragma unroll
              for (int q = 0; q < K / 4; ++q) {
                unpack4(packed[g][q], v);
                acc[4 * q + 0] = cc * v[0];
                acc[4 * q + 1] = cc * v[1];
                acc[4 * q + 2] = cc * v[2];
                acc[4 * q + 3] = cc * v[3];
              }
              any = true;
            } else {
#pragma unroll
              for (int q = 0; q < K / 4; ++q) {
                unpack4(packed[g][q], v);
                acc[4 * q + 0] = fmaf(cc, v[0], acc[4 * q + 0]);
                acc[4 * q + 1] = fmaf(cc, v[1], acc[4 * q + 1]);
                acc[4 * q + 2] = fmaf(cc, v[2], acc[4 * q + 2]);
                acc[4 * q + 3] = fmaf(cc, v[3], acc[4 * q + 3]);
              }
            }
          }
        }
        if (any) {
          float* yp = Y + (size_t)(b0 + m) * N + lane0;
#pragma unroll
          for (int q = 0; q < K / 4; ++q)
            atomic_add4(yp + 4 * q, acc[4 * q + 0], acc[4 * q + 1],
                        acc[4 * q + 2], acc[4 * q + 3]);
        }
      }
    }
    __syncthreads();
  }
}

void check(const torch::Tensor& T, const torch::Tensor& J, const torch::Tensor& C,
           const torch::Tensor& Y);

// ---------------------------------------------------------------- diagnostics
//
// Timing ablations, to attribute cs_v1's cost to its three parts instead of guessing.
// MODE 0 = the full kernel; 1 = everything but the global flush; 2 = the row scan only
// (no shared atomics, no flush, no barriers beyond staging). Only MODE 0 produces a
// correct result -- 1 and 2 exist purely to be timed. The `keep` accumulator and its
// runtime-conditional store are what stop ptxas deleting the scan it is measuring.

template <int K, int M, int TB, int MODE>
__global__ __launch_bounds__(256 * TB) void cs_v1_diag_kernel(
    const int8_t* __restrict__ T, const uint8_t* __restrict__ J,
    const float* __restrict__ C, float* __restrict__ Y,
    int B, int NT, int R, int N, int use_coef) {
  const int r = threadIdx.x;
  const int tl = threadIdx.y;
  const int tg = blockIdx.x * TB;
  const int t = tg + tl;
  const int lane0 = blockIdx.y * K;
  const int tid = tl * 256 + r;

  uint32_t packed[K / 4];
  {
    const uint32_t* row =
        reinterpret_cast<const uint32_t*>(T + ((size_t)t * R + r) * N + lane0);
#pragma unroll
    for (int q = 0; q < K / 4; ++q) packed[q] = row[q];
  }
  __shared__ uint8_t js[M][TB];
  __shared__ float cs[M][TB];
  __shared__ float red[M][K];
  const int FLUSH = (M * K) / 4;
  float keep = 0.f;

  for (int b0 = 0; b0 < B; b0 += M) {
    if (tid < M * TB) {
      const int m = tid / TB, q = tid % TB;
      const int b = b0 + m;
      js[m][q] = (b < B) ? J[(size_t)b * NT + tg + q] : (uint8_t)0;
      cs[m][q] = (b < B) ? (use_coef ? C[(size_t)b * NT + tg + q] : 1.0f) : 0.0f;
    }
    if (MODE < 2) {
      for (int i = tid; i < M * K; i += 256 * TB) red[i / K][i % K] = 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if ((js[m][tl] == (uint8_t)r) && (b0 + m < B)) {
        const float cc = cs[m][tl];
        float v[4];
#pragma unroll
        for (int q = 0; q < K / 4; ++q) {
          unpack4(packed[q], v);
          if (MODE < 2) {
            atomicAdd(&red[m][4 * q + 0], cc * v[0]);
            atomicAdd(&red[m][4 * q + 1], cc * v[1]);
            atomicAdd(&red[m][4 * q + 2], cc * v[2]);
            atomicAdd(&red[m][4 * q + 3], cc * v[3]);
          } else {
            keep += cc * (v[0] + v[1] + v[2] + v[3]);
          }
        }
      }
    }
    __syncthreads();
    for (int u = (MODE < 1 ? tid : FLUSH); u < FLUSH; u += 256 * TB) {
      const int m = (4 * u) / K, k = (4 * u) % K;
      const int b = b0 + m;
      if (b < B) {
        const float* s = &red[m][k];
        atomic_add4(Y + (size_t)b * N + lane0 + k, s[0], s[1], s[2], s[3]);
      }
    }
    __syncthreads();
  }
  if (keep == 12345.678f) Y[0] = keep;
}

template <int K, int M, int MODE>
__global__ __launch_bounds__(256) void cs_v0_diag_kernel(
    const int8_t* __restrict__ T, const uint8_t* __restrict__ J,
    const float* __restrict__ C, float* __restrict__ Y,
    int B, int NT, int R, int N, int use_coef) {
  const int r = threadIdx.x;
  const int t = blockIdx.x;
  const int lane0 = blockIdx.y * K;
  float val[K];
  {
    const int8_t* row = T + ((size_t)t * R + r) * N + lane0;
#pragma unroll
    for (int k = 0; k < K; ++k) val[k] = (float)row[k];
  }
  __shared__ uint8_t js[M];
  __shared__ float cs[M];
  float keep = 0.f;
  for (int b0 = 0; b0 < B; b0 += M) {
    for (int m = threadIdx.x; m < M; m += 256) {
      const int b = b0 + m;
      js[m] = (b < B) ? J[(size_t)b * NT + t] : (uint8_t)0;
      cs[m] = (b < B) ? (use_coef ? C[(size_t)b * NT + t] : 1.0f) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if ((js[m] == (uint8_t)r) && (b0 + m < B)) {
        const float cc = cs[m];
        float* yp = Y + (size_t)(b0 + m) * N + lane0;
#pragma unroll
        for (int k = 0; k < K; ++k) {
          if (MODE < 1) atomicAdd(yp + k, cc * val[k]);
          else keep += cc * val[k];
        }
      }
    }
    __syncthreads();
  }
  if (keep == 12345.678f) Y[0] = keep;
}

void cs_v1_diag(torch::Tensor T, torch::Tensor J, torch::Tensor C, torch::Tensor Y,
                int64_t K, int64_t M, int64_t TB, int64_t mode, int64_t use_coef) {
  check(T, J, C, Y);
  const int NT = T.size(0), R = T.size(1), N = T.size(2);
  const int B = J.size(0);
  dim3 blk(256, (int)TB);
#define D1(kk, mm, tb, md)                                                       \
  if (K == kk && M == mm && TB == tb && mode == md) {                            \
    dim3 grd(NT / tb, N / kk);                                                   \
    cs_v1_diag_kernel<kk, mm, tb, md>                                            \
        <<<grd, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(                    \
            T.data_ptr<int8_t>(), J.data_ptr<uint8_t>(), C.data_ptr<float>(),    \
            Y.data_ptr<float>(), B, NT, R, N, (int)use_coef); } else
  D1(32, 32, 4, 0) D1(32, 32, 4, 1) D1(32, 32, 4, 2)
  D1(16, 32, 4, 0) D1(16, 32, 4, 1) D1(16, 32, 4, 2)
  D1(64, 32, 2, 0) D1(64, 32, 2, 1) D1(64, 32, 2, 2)
  { TORCH_CHECK(false, "unsupported diag v1 ", K, ",", M, ",", TB, ",", mode); }
#undef D1
  C10_CUDA_CHECK(cudaGetLastError());
}

void cs_v0_diag(torch::Tensor T, torch::Tensor J, torch::Tensor C, torch::Tensor Y,
                int64_t K, int64_t M, int64_t mode, int64_t use_coef) {
  check(T, J, C, Y);
  const int NT = T.size(0), R = T.size(1), N = T.size(2);
  const int B = J.size(0);
#define D0(kk, mm, md)                                                           \
  if (K == kk && M == mm && mode == md) {                                        \
    dim3 grd(NT, N / kk);                                                        \
    cs_v0_diag_kernel<kk, mm, md>                                                \
        <<<grd, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(                    \
            T.data_ptr<int8_t>(), J.data_ptr<uint8_t>(), C.data_ptr<float>(),    \
            Y.data_ptr<float>(), B, NT, R, N, (int)use_coef); } else
  D0(4, 48, 0) D0(4, 48, 1) D0(16, 96, 0) D0(16, 96, 1)
  { TORCH_CHECK(false, "unsupported diag v0 ", K, ",", M, ",", mode); }
#undef D0
  C10_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- launchers

void check(const torch::Tensor& T, const torch::Tensor& J, const torch::Tensor& C,
           const torch::Tensor& Y) {
  TORCH_CHECK(T.is_cuda() && J.is_cuda() && C.is_cuda() && Y.is_cuda(), "cuda only");
  TORCH_CHECK(T.dtype() == torch::kInt8, "T must be int8");
  TORCH_CHECK(J.dtype() == torch::kUInt8, "J must be uint8");
  TORCH_CHECK(C.dtype() == torch::kFloat32 && Y.dtype() == torch::kFloat32, "fp32");
  TORCH_CHECK(T.is_contiguous() && J.is_contiguous() && C.is_contiguous() && Y.is_contiguous(),
              "contiguous");
}

void gather(torch::Tensor T, torch::Tensor J, torch::Tensor C, torch::Tensor Y,
            int64_t tsplit, int64_t use_coef, int64_t tok_per_blk) {
  check(T, J, C, Y);
  const int NT = T.size(0), R = T.size(1), N = T.size(2);
  const int B = J.size(0);
  TORCH_CHECK(N % 4 == 0, "N must be a multiple of 4");
  TORCH_CHECK(NT % tsplit == 0, "tsplit must divide NT");
  const int N4 = N / 4;
  const int tpb = (int)tok_per_blk;
  int workers = N4 < 256 ? N4 : 256;
  if (workers * tpb > 1024) workers = 1024 / tpb;
  dim3 blk(workers, tpb);
  dim3 grd((B + tpb - 1) / tpb, (int)tsplit);
  gather_kernel<<<grd, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
      T.data_ptr<int8_t>(), J.data_ptr<uint8_t>(), C.data_ptr<float>(), Y.data_ptr<float>(),
      B, NT, R, N4, (int)tsplit, (int)use_coef);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int K, int M>
void run_v0(torch::Tensor& T, torch::Tensor& J, torch::Tensor& C, torch::Tensor& Y,
            int B, int NT, int R, int N, int use_coef) {
  dim3 grd(NT, N / K);
  cs_v0_kernel<K, M><<<grd, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      T.data_ptr<int8_t>(), J.data_ptr<uint8_t>(), C.data_ptr<float>(), Y.data_ptr<float>(),
      B, NT, R, N, use_coef);
}

void cs_v0(torch::Tensor T, torch::Tensor J, torch::Tensor C, torch::Tensor Y,
           int64_t K, int64_t M, int64_t use_coef) {
  check(T, J, C, Y);
  const int NT = T.size(0), R = T.size(1), N = T.size(2);
  const int B = J.size(0);
  TORCH_CHECK(R == 256, "cs kernels assume R == 256 (one row per thread of a 256-thread block)");
  TORCH_CHECK(N % K == 0, "K must divide N");
#define V0(kk, mm) \
  if (K == kk && M == mm) { run_v0<kk, mm>(T, J, C, Y, B, NT, R, N, (int)use_coef); } else
  V0(1, 192) V0(1, 128) V0(1, 64) V0(2, 96) V0(2, 64) V0(4, 48) V0(4, 64) V0(4, 32)
  // larger tiles than the original spec's M*K ~= 224, added after measurement: v0 turns
  // out to be bound by its atomic count, which M does not change, so these mostly
  // confirm that rather than help.
  V0(4, 96) V0(4, 128) V0(8, 32) V0(8, 48) V0(8, 96) V0(2, 128) V0(16, 48) V0(16, 96)
  { TORCH_CHECK(false, "unsupported (K,M) for v0: ", K, ",", M); }
#undef V0
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int K, int M, int TB, int CLU>
void run_v1(torch::Tensor& T, torch::Tensor& J, torch::Tensor& C, torch::Tensor& Y,
            int B, int NT, int R, int N, int use_coef) {
  dim3 grd(NT / (TB * CLU), N / K, CLU);
  dim3 blk(256, TB);
  cs_v1_kernel<K, M, TB, CLU><<<grd, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
      T.data_ptr<int8_t>(), J.data_ptr<uint8_t>(), C.data_ptr<float>(), Y.data_ptr<float>(),
      B, NT, R, N, use_coef);
}

void cs_v1(torch::Tensor T, torch::Tensor J, torch::Tensor C, torch::Tensor Y,
           int64_t K, int64_t M, int64_t TB, int64_t CLU, int64_t use_coef) {
  check(T, J, C, Y);
  const int NT = T.size(0), R = T.size(1), N = T.size(2);
  const int B = J.size(0);
  TORCH_CHECK(R == 256, "cs kernels assume R == 256");
  TORCH_CHECK(N % K == 0, "K must divide N");
  TORCH_CHECK(NT % (TB * CLU) == 0, "TB*CLU must divide NT");
#define V1(kk, mm, tb, cl) \
  if (K == kk && M == mm && TB == tb && CLU == cl) {                     \
    run_v1<kk, mm, tb, cl>(T, J, C, Y, B, NT, R, N, (int)use_coef); } else
  // (K, M, TB, CLU)
  V1(8, 4, 4, 1) V1(8, 2, 4, 1) V1(8, 1, 4, 1)
  V1(16, 4, 4, 1) V1(16, 2, 4, 1) V1(16, 1, 4, 1)
  V1(32, 2, 4, 1) V1(32, 1, 4, 1)
  V1(32, 1, 2, 1) V1(64, 1, 2, 1) V1(64, 1, 1, 1) V1(32, 2, 1, 1)
  V1(16, 4, 2, 1) V1(8, 4, 2, 1)
  V1(32, 1, 4, 2) V1(32, 1, 4, 4) V1(32, 1, 4, 8)
  V1(16, 2, 4, 4) V1(16, 2, 4, 8) V1(16, 4, 4, 8)
  // The spec capped M at 4 on the argument that the scan cost is one compare per
  // (thread, token) whatever M is. True -- but the block reduction it asks for in the
  // same breath costs TWO barriers plus a shared-memory zero-and-flush per TOKEN TILE,
  // i.e. per M tokens, and at M<=4 that overhead dominates everything else by two
  // orders of magnitude (measured). These larger tiles amortise it. M is capped at 32
  // because the per-thread hit mask is one uint32.
  V1(8, 8, 4, 1) V1(8, 16, 4, 1) V1(8, 32, 4, 1)
  V1(16, 8, 4, 1) V1(16, 16, 4, 1) V1(16, 32, 4, 1)
  V1(32, 8, 4, 1) V1(32, 16, 4, 1) V1(32, 32, 4, 1)
  V1(64, 8, 2, 1) V1(64, 16, 2, 1) V1(64, 32, 2, 1)
  V1(32, 16, 2, 1) V1(32, 32, 1, 1) V1(16, 32, 2, 1)
  V1(32, 16, 4, 4) V1(32, 16, 4, 8) V1(16, 32, 4, 8)
  { TORCH_CHECK(false, "unsupported (K,M,TB,CLU) for v1: ", K, ",", M, ",", TB, ",", CLU); }
#undef V1
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int K, int M, int TB, int TRANS>
void run_v2(torch::Tensor& T, torch::Tensor& J, torch::Tensor& C, torch::Tensor& Y,
            int B, int NT, int R, int N, int use_coef) {
  dim3 grd(NT / TB, N / K);
  dim3 blk(256, TB);
  cs_v2_kernel<K, M, TB, TRANS><<<grd, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
      T.data_ptr<int8_t>(), J.data_ptr<uint8_t>(), C.data_ptr<float>(), Y.data_ptr<float>(),
      B, NT, R, N, use_coef);
}

template <int K, int M, int G, int TRANS>
void run_v3(torch::Tensor& T, torch::Tensor& J, torch::Tensor& C, torch::Tensor& Y,
            int B, int NT, int R, int N, int use_coef) {
  dim3 grd(NT / G, N / K);
  cs_v3_kernel<K, M, G, TRANS><<<grd, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      T.data_ptr<int8_t>(), J.data_ptr<uint8_t>(), C.data_ptr<float>(), Y.data_ptr<float>(),
      B, NT, R, N, use_coef);
}

// With trans=1 the caller passes the permuted table, shape [R, NT, N].
void shape_of(const torch::Tensor& T, int trans, int& NT, int& R, int& N) {
  if (trans) { R = T.size(0); NT = T.size(1); }
  else       { NT = T.size(0); R = T.size(1); }
  N = T.size(2);
}

void cs_v2(torch::Tensor T, torch::Tensor J, torch::Tensor C, torch::Tensor Y,
           int64_t K, int64_t M, int64_t TB, int64_t trans, int64_t use_coef) {
  check(T, J, C, Y);
  int NT, R, N;
  shape_of(T, (int)trans, NT, R, N);
  const int B = J.size(0);
  TORCH_CHECK(R == 256, "cs kernels assume R == 256");
  TORCH_CHECK(N % K == 0 && NT % TB == 0, "K|N and TB|NT required");
#define V2(kk, mm, tb, tr)                                                     \
  if (K == kk && M == mm && TB == tb && trans == tr) {                         \
    run_v2<kk, mm, tb, tr>(T, J, C, Y, B, NT, R, N, (int)use_coef); } else
#define V2P(kk, mm, tb) V2(kk, mm, tb, 0) V2(kk, mm, tb, 1)
  V2P(16, 32, 4) V2P(32, 32, 4) V2P(64, 32, 4)
  V2P(32, 32, 2) V2P(64, 32, 2) V2P(128, 32, 2)
  V2P(64, 16, 4) V2P(128, 16, 4) V2P(128, 32, 1) V2P(64, 32, 1)
  V2P(32, 16, 4) V2P(16, 32, 2)
  { TORCH_CHECK(false, "unsupported cs_v2 ", K, ",", M, ",", TB, ",", trans); }
#undef V2P
#undef V2
  C10_CUDA_CHECK(cudaGetLastError());
}

void cs_v3(torch::Tensor T, torch::Tensor J, torch::Tensor C, torch::Tensor Y,
           int64_t K, int64_t M, int64_t G, int64_t trans, int64_t use_coef) {
  check(T, J, C, Y);
  int NT, R, N;
  shape_of(T, (int)trans, NT, R, N);
  const int B = J.size(0);
  TORCH_CHECK(R == 256, "cs kernels assume R == 256");
  TORCH_CHECK(N % K == 0 && NT % G == 0, "K|N and G|NT required");
#define V3(kk, mm, gg, tr)                                                     \
  if (K == kk && M == mm && G == gg && trans == tr) {                          \
    run_v3<kk, mm, gg, tr>(T, J, C, Y, B, NT, R, N, (int)use_coef); } else
#define V3P(kk, mm, gg) V3(kk, mm, gg, 0) V3(kk, mm, gg, 1)
  V3P(32, 32, 1) V3P(32, 32, 8) V3P(64, 32, 8) V3P(16, 32, 32)
  // G is capped at 128 with K=4: the packed table values cost G*K/4 registers, so
  // G=256 needs 256 registers at K=4 and cannot be expressed with the 4-int8-per-word
  // packing at all below K=4. The intra-thread arm is register-bound well before its
  // dedup factor becomes interesting.
  V3P(8, 32, 64) V3P(4, 32, 128) V3P(4, 32, 64) V3P(8, 32, 16)
  V3P(16, 16, 32) V3P(32, 32, 4) V3P(16, 32, 8) V3P(8, 32, 32)
  { TORCH_CHECK(false, "unsupported cs_v3 ", K, ",", M, ",", G, ",", trans); }
#undef V3P
#undef V3
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather", &gather, "output-stationary gather+sum baseline");
  m.def("cs_v2", &cs_v2, "cell-stationary, conflict-free cross-thread reduction");
  m.def("cs_v3", &cs_v3, "cell-stationary, G tables per thread (intra-thread dedup)");
  m.def("cs_v0", &cs_v0, "naive cell-stationary");
  m.def("cs_v1", &cs_v1, "strengthened cell-stationary");
  m.def("cs_v0_diag", &cs_v0_diag, "cs_v0 timing ablation (mode 1 drops the atomics)");
  m.def("cs_v1_diag", &cs_v1_diag, "cs_v1 timing ablation (mode 1 drops the flush, 2 the reduction)");
}
