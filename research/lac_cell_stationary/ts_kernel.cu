// Table-stationary (TS) gather for the p2_int8 read.
//
// The counterpart of the accumulator-stationary `read_cells`: there the accumulators sit in
// registers and the table rows stream past; here the TABLE ROWS sit in registers for the life of
// the block and the tokens stream past.
//
// THREAD IDENTITY. A block is 256 threads and owns G tables x L output lanes. Thread r owns ROW r
// -- all 256 rows are resident simultaneously, one per thread -- and holds, in registers, row r's
// L lanes of each of its block's G tables, int8-packed 4 to a 32-bit word: G*L/4 words, loaded once
// at block start (`load`, below) and never reloaded. That is what makes it table-stationary.
//
// GRID. (T / G) table-groups x (D / L) lane-slices. Every block streams the WHOLE batch in tiles of
// BT tokens, so the table set is read exactly once per launch: (T/G)*(D/L) blocks * 256 rows * G*L
// bytes = T * K * D bytes = the whole table set.
//
// PER TILE.
//   stage   the tile's cells for this block's G tables into shared, 3 bytes per (token, table)
//   zero    a shared int32 accumulator acc[BT][L]
//   2G passes, one per (table, cell slot): thread r scans the tile and, for every token whose cell
//           for that (table, slot) is r, adds its L lanes into acc[token]. NO shared atomics --
//           within one (table, slot) pass a token has exactly ONE writer, because its cell index is
//           a single value and exactly one thread owns that row. A __syncthreads() between passes.
//   flush   atomicAdd acc into the global output.
//
// TWO DELIBERATE DEVIATIONS FROM THE SPEC, both forced, both called out in the report:
//
//  1. THE OUTPUT AND ITS ATOMICS ARE int32, NOT float. The spec said to convert acc to float and
//     atomicAdd floats. That cannot be bit-exact against the AS kernel and the gate demands bit
//     exactness. AS sums all T tables in int32 and converts ONCE; a float flush would sum T/G
//     separately-converted partials, and with |total| reaching 2*T*128*2^10 = 2^26 against float's
//     24-bit mantissa those roundings do not agree. int32 atomicAdd is exact and associative, so
//     the partials recombine to the identical integer, and the single conversion then happens
//     outside. Traffic is unchanged (4 bytes either way) and integer atomics are not slower.
//  2. `__launch_bounds__(256)` is stated explicitly so ptxas cannot trade the register-resident
//     table for occupancy it cannot use -- only 256 threads exist per block by construction.
//
// The output buffer must be ZEROED by the caller before launch; the wrapper does it and the caller
// is told to count that zeroing in any timing, symmetrically with the AS arm.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr uint8_t DISCARD = 15;          // csrc/pow2_scalars.cuh:25
constexpr int SMEM_LIMIT = 101376;       // cudaDevAttrMaxSharedMemoryPerBlockOptin, sm_120

__device__ __forceinline__ void unpack4(uint32_t w, int v[4]) {
  v[0] = (int)(int8_t)(w & 0xffu);
  v[1] = (int)(int8_t)((w >> 8) & 0xffu);
  v[2] = (int)(int8_t)((w >> 16) & 0xffu);
  v[3] = (int)(int8_t)((w >> 24) & 0xffu);
}

// __launch_bounds__(256, 1): the SECOND argument is load-bearing and was missing in the first
// version. With only the block size given, ptxas targets maximum occupancy and caps itself at
// 80-128 registers, which is not enough to hold G*L/4 table words -- so it put `val` in LOCAL
// memory and every instantiation spilled 1.7-6.4 kB. Declaring that one block per SM suffices
// lets it use the full 255. One block per SM is what actually happens anyway: at 40-90 kB of
// shared memory per block, two blocks cannot be resident.
// TLAY: 0 = the stored layout [T*K, stride] (lane-innermost, what AS needs).
//       1 = BLOCKED [T/G][D/L][K][G][L] -- each block's (K cells x G tables x L lanes) tile fully
//           contiguous, 32 KiB at G*L=128. Built host-side, once, outside any timed region.
// CLAY: 0 = cells [N, 1, T, 3] (what AS consumes).
//       1 = TRANSPOSED [T, N, 3] -- a block's G tables each give 3*Bt contiguous bytes per tile.
template <int G, int L, int BT, int TLAY, int CLAY>
__global__ __launch_bounds__(256, 1) void ts_kernel(
    const int8_t* __restrict__ W, const uint8_t* __restrict__ CELLS,
    int* __restrict__ OUT, int N, int T, int K, int D, int stride) {
  const int r = threadIdx.x;                  // this thread owns row r, for every table of the block
  const int tg = blockIdx.x * G;              // first table of the block
  const int lane0 = blockIdx.y * L;           // first output lane of the block

  // ---- the register-resident table: G tables x L lanes of row r, loaded once, never reloaded ----
  uint32_t val[G][L / 4];
  if constexpr (TLAY == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const uint32_t* row =
          reinterpret_cast<const uint32_t*>(W + ((size_t)(tg + g) * K + r) * stride + lane0);
#pragma unroll
      for (int q = 0; q < L / 4; ++q) val[g][q] = row[q];
    }
  } else {
    // tile base, then [cell r][table g][lane l] inside it: thread r reads G*L contiguous bytes.
    const int8_t* tile = W + ((size_t)blockIdx.x * (D / L) + blockIdx.y) * (size_t)K * G * L;
    const uint32_t* base = reinterpret_cast<const uint32_t*>(tile + (size_t)r * G * L);
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
      for (int q = 0; q < L / 4; ++q) val[g][q] = base[g * (L / 4) + q];
    }
  }

  extern __shared__ char smem[];
  int* acc = reinterpret_cast<int*>(smem);                          // [BT][L] int32
  uint8_t* csh = reinterpret_cast<uint8_t*>(smem) + (size_t)BT * L * 4;   // [BT][G][3]

  for (int b0 = 0; b0 < N; b0 += BT) {
    const int nb = (N - b0 < BT) ? (N - b0) : BT;

    // ---- stage this tile's cells for the block's G tables; coalesced over (token, table) ----
    if constexpr (CLAY == 0) {
      for (int i = threadIdx.x; i < nb * G; i += 256) {
        const int n = i / G, g = i - n * G;
        const size_t src = ((size_t)(b0 + n) * T + tg + g) * 3;
        const size_t dst = (size_t)i * 3;
        csh[dst] = CELLS[src];
        csh[dst + 1] = CELLS[src + 1];
        csh[dst + 2] = CELLS[src + 2];
      }
    } else {
      // [T, N, 3]: g outer, n strided, so consecutive threads read consecutive tokens of the
      // same table -- 3*nb contiguous bytes per table per tile instead of 3*G of every 3*T.
      for (int g = 0; g < G; ++g) {
        const uint8_t* src = CELLS + ((size_t)(tg + g) * N + b0) * 3;
        for (int n = threadIdx.x; n < nb; n += 256) {
          const size_t dst = ((size_t)n * G + g) * 3;
          csh[dst] = src[(size_t)n * 3];
          csh[dst + 1] = src[(size_t)n * 3 + 1];
          csh[dst + 2] = src[(size_t)n * 3 + 2];
        }
      }
    }
    for (int i = threadIdx.x; i < nb * L; i += 256) acc[i] = 0;
    __syncthreads();

    // ---- 2G passes. Within a pass, each token has exactly one writing thread, so the adds are
    // plain stores to shared, not atomics. The barrier after each pass is what makes that safe
    // across passes: two different (table, slot) passes CAN target the same acc[n][l].
#pragma unroll
    for (int g = 0; g < G; ++g) {
      // The g loop MUST be unrolled -- val[g][q] is a register array and needs a compile-time
      // index. The slot loop must NOT be: it does not index val at all, so unrolling it only
      // doubles the live working set for nothing.
#pragma unroll 1
      for (int slot = 0; slot < 2; ++slot) {
        // `unroll 1`: the token scan must NOT be unrolled. Letting ptxas unroll it multiplies the
        // live temporaries of the (already 2G-times unrolled) body and pushed every instantiation
        // into spilling the register-resident table -- which is the one thing this kernel exists
        // to avoid. Pinning it keeps `val` in registers.
#pragma unroll 1
        for (int n = 0; n < nb; ++n) {
          const uint8_t* c = csh + ((size_t)n * G + g) * 3;
          const int sh = (slot == 0) ? (c[2] & 15) : (c[2] >> 4);
          // A REAL branch, not a predicate: the L-lane body must not run on a miss, or the cost
          // becomes independent of L and wide lanes look free. Verified in the SASS.
          if (sh != (int)DISCARD && c[slot] == (uint8_t)r) {
            int* a = acc + (size_t)n * L;
            int v[4];
#pragma unroll
            for (int q = 0; q < L / 4; ++q) {
              unpack4(val[g][q], v);
              a[4 * q + 0] += v[0] << sh;
              a[4 * q + 1] += v[1] << sh;
              a[4 * q + 2] += v[2] << sh;
              a[4 * q + 3] += v[3] << sh;
            }
          }
        }
        __syncthreads();
      }
    }

    // ---- flush: strided, so a tile of BT*L elements is covered whatever the block size ----
    for (int i = threadIdx.x; i < nb * L; i += 256) {
      const int n = i / L, l = i - n * L;
      atomicAdd(OUT + (size_t)(b0 + n) * D + lane0 + l, acc[i]);
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------ launcher

template <int G, int L, int BT, int TLAY, int CLAY>
void launch(const torch::Tensor& W, const torch::Tensor& C, torch::Tensor& O,
            int N, int T, int K, int D, int stride) {
  constexpr size_t SMEM = (size_t)BT * L * 4 + (size_t)BT * G * 3;
  static_assert(L % 4 == 0, "L must be a multiple of 4 (int8 packed 4 per word)");
  TORCH_CHECK(SMEM <= SMEM_LIMIT,
              "TS config G=", G, " L=", L, " Bt=", BT, " needs ", (long)SMEM,
              " B of shared memory (acc ", (long)BT * L * 4, " + cells ", (long)BT * G * 3,
              ") against the ", SMEM_LIMIT, " B per-block limit: over by ",
              (long)SMEM - SMEM_LIMIT, " B. Refusing rather than substituting another kernel.");
  auto k = ts_kernel<G, L, BT, TLAY, CLAY>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
  dim3 grid(T / G, D / L);
  k<<<grid, 256, SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<int8_t>(), C.data_ptr<uint8_t>(), O.data_ptr<int>(), N, T, K, D, stride);
  C10_CUDA_CHECK(cudaGetLastError());
}

// tlay / clay select the layout variants; N and T must be passed explicitly because the
// transposed tensors no longer carry them in the shape the baseline does.
torch::Tensor ts_read(torch::Tensor W, torch::Tensor CELLS, int64_t D,
                      int64_t G, int64_t L, int64_t BT,
                      int64_t tlay, int64_t clay, int64_t N_in, int64_t T_in) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kChar && W.is_contiguous(),
              "W must be contiguous int8 CUDA");
  TORCH_CHECK(CELLS.is_cuda() && CELLS.scalar_type() == torch::kByte && CELLS.is_contiguous(),
              "CELLS must be contiguous uint8 CUDA");
  int N, T, stride, K;
  if (clay == 0) {
    TORCH_CHECK(CELLS.dim() == 4 && CELLS.size(1) == 1 && CELLS.size(3) == 3,
                "baseline CELLS must be [N, 1, T, 3]");
    N = CELLS.size(0);
    T = CELLS.size(2);
  } else {
    TORCH_CHECK(CELLS.dim() == 3 && CELLS.size(2) == 3, "transposed CELLS must be [T, N, 3]");
    T = CELLS.size(0);
    N = CELLS.size(1);
  }
  TORCH_CHECK(N_in == N && T_in == T, "N/T mismatch: kernel sees ", N, "/", T,
              ", caller said ", N_in, "/", T_in);
  if (tlay == 0) {
    TORCH_CHECK(W.dim() == 2, "baseline W must be [T*K, stride]");
    stride = W.size(1);
    K = W.size(0) / T;
    TORCH_CHECK(W.size(0) == (int64_t)T * K, "W rows must be T*K");
    TORCH_CHECK(D <= stride, "D must not exceed the row stride");
  } else {
    // blocked: a flat buffer of T/G * D/L * K * G * L bytes; K is derived, stride unused.
    stride = (int)D;
    K = (int)(W.numel() / ((int64_t)T * D));
    TORCH_CHECK(W.numel() == (int64_t)T * K * D, "blocked W must hold T*K*D bytes");
  }
  TORCH_CHECK(T % G == 0, "G must divide T (", T, ")");
  TORCH_CHECK(D % L == 0, "L must divide D (", D, ")");
  // int32 output, zeroed: see the header note on why the flush is integer and not float.
  auto out = torch::zeros({N, 1, D}, W.options().dtype(torch::kInt));

#define TSL(g, l, bt, tl, cl)                                                             \
  if (G == g && L == l && BT == bt && tlay == tl && clay == cl) {                         \
    launch<g, l, bt, tl, cl>(W, CELLS, out, N, T, K, (int)D, stride); } else
  // The four layout combinations, for the spill-free configs only -- instantiating all of
  // them for every config would quadruple a build that is already 23 kernels.
#define TS4(g, l, bt) TSL(g, l, bt, 0, 0) TSL(g, l, bt, 1, 0) TSL(g, l, bt, 0, 1) TSL(g, l, bt, 1, 1)
  TS4(2, 64, 256) TS4(4, 32, 256) TS4(4, 32, 512) TS4(8, 16, 256) TS4(8, 16, 1024)
  TS4(16, 8, 512) TS4(32, 4, 512)
#define TS(g, l, bt) TSL(g, l, bt, 0, 0)
  // The three points named in the task -- instantiated so their refusal is a real dispatch
  // message with real arithmetic, not a claim.
  TS(8, 96, 256) TS(32, 24, 1024) TS(16, 48, 512)
  // Fitting points, gated below.
  TS(8, 64, 256) TS(8, 32, 256) TS(16, 32, 512) TS(32, 16, 512) TS(32, 16, 256)
  TS(8, 64, 512) TS(16, 32, 256)
  // Smaller table budgets, added after every G*L/4 >= 128 config spilled.
  TS(8, 16, 256) TS(16, 16, 256) TS(16, 16, 512) TS(32, 8, 512) TS(8, 16, 1024)
  // 32 table words turned out to be the spill-free ceiling; these span G at that fixed budget,
  // which is the axis that sets the global-atomic traffic (proportional to 1/G).
  TS(4, 32, 256) TS(4, 32, 512) TS(16, 8, 512) TS(32, 4, 512)
  // Step 2: push L as far as the register budget allows, by shrinking G instead. L is the
  // only parameter the measured runtime depends on, so this is the only axis that can help.
  TS(2, 64, 256) TS(4, 64, 256) TS(2, 128, 192) TS(1, 128, 256)
  {
    TORCH_CHECK(false, "no TS instantiation for G=", G, " L=", L, " Bt=", BT,
                " tlay=", tlay, " clay=", clay);
  }
#undef TS
#undef TS4
#undef TSL
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ts_read", &ts_read, "table-stationary int8 gather; returns int32 [N,1,D]");
}
