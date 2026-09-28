// Direct scatter with atomic accumulation -- the third dataflow, where nothing rests.
//
// AS keeps the accumulators in registers and streams table rows past them; TS keeps the table
// rows in registers and streams tokens past them. This keeps NEITHER: every (token, table, cell)
// contribution is computed by whichever thread owns it and scattered straight into an accumulator.
// The two arms differ only in WHERE that accumulator lives.
//
//   ARM A  acc is the global output. Every contribution is a global atomicAdd. No stationary
//          operand at all, no shared memory, no barriers. The output must be pre-zeroed.
//   ARM B  acc[Bt][D] is a shared-memory tile owned by the block. Contributions are shared
//          atomicAdds; one barrier; then a PLAIN-STORE flush to global. The block is the unique
//          writer of its Bt tokens, so there are no global atomics and no pre-zeroing.
//
// Shared conventions with the AS/TS kernels, unchanged: tables int8 [T*K, stride] row-contiguous
// (lane innermost), cells uint8 [N, T, 3] = (c1, c2, sh1 | sh2 << 4) with DISCARD = 15, output
// int32 in units of 2^-6, inner op acc[j] += (lane & m) << s with m = (sh == DISCARD) ? 0 : -1.
// int32 accumulation is what makes both arms bit-exact against AS despite reassociating the sum.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr uint8_t DISCARD = 15;
constexpr int SMEM_LIMIT = 101376;

// ------------------------------------------------------------------ ARM A: global atomics
//
// THREAD MAPPING. gid = ((n * T) + t) * (D/L) + lane_tile.  So threadIdx.x varies fastest over
// LANE TILE, then table, then token: adjacent threads hold adjacent lane tiles of the SAME (n, t),
// which is what makes a warp's atomic writes contiguous (32 * L * 4 bytes) and the three cell
// bytes a warp-wide broadcast out of L1. Both cell slots are a loop of at most 2 INSIDE the
// thread, as specified, rather than a grid axis -- with slot as a grid axis the two slots of one
// (n, t) would land in different warps and each would re-read the same cell bytes.
//
// The accumulate is written as a bare atomicAdd whose result is discarded so ptxas can emit RED
// (fire-and-forget) rather than ATOM (which returns the old value); verified in the SASS.

template <int L>
__global__ void scatter_global_kernel(
    const int8_t* __restrict__ W, const uint8_t* __restrict__ CELLS,
    int* __restrict__ OUT, int N, int T, int K, int D, int stride) {
  const int tiles = D / L;
  const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)N * T * tiles;
  if (gid >= total) return;

  const int lt = (int)(gid % tiles);
  const long long nt = gid / tiles;
  const int t = (int)(nt % T);
  const int n = (int)(nt / T);
  const int lane0 = lt * L;

  const uint8_t* c = CELLS + ((size_t)n * T + t) * 3;
  const int sh1 = c[2] & 15, sh2 = c[2] >> 4;
  int* op = OUT + (size_t)n * D + lane0;

#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    const int sh = slot ? sh2 : sh1;
    if (sh == (int)DISCARD) continue;           // a real branch; nothing is read or written
    const int8_t* row = W + ((size_t)t * K + c[slot]) * stride + lane0;
#pragma unroll
    for (int j = 0; j < L; ++j) {
      const int v = (int)row[j];
      atomicAdd(op + j, v << sh);               // result unused -> RED.global.add
    }
  }
}

// ------------------------------------------------------------------ ARM B: shared accumulate
//
// THREAD MAPPING. Block owns Bt tokens and the whole D-wide row for each: acc[Bt][DP] int32 in
// shared, DP = D + PAD. The block's work is Bt * T (token, table) pairs x (D/L) lane tiles,
// flattened and strided over the block's threads so that, again, adjacent threads hold adjacent
// lane tiles of the same (n, t).
//
// The block is the ONLY writer of rows [b0, b0+Bt) of the output: the grid's x axis partitions
// tokens and nothing else touches them. Hence the flush is a plain store, there are no global
// atomics, and the output needs no pre-zeroing. (The caller still allocates with torch::zeros for
// shape parity with arm A; that is noted in the deviation list.)
//
// BANK CONFLICTS. acc is int32, so 32 consecutive int32 are 32 distinct banks. A thread writing L
// consecutive lanes touches L consecutive banks, and 32 adjacent threads with stride L cover
// lanes [0, 32L), i.e. each bank is hit L times per warp-wide access -> an L-way conflict.
// PAD breaks the row-to-row alignment only; it does not help the within-row stride, so the
// conflict factor is L regardless of PAD. PAD is kept as a parameter and defaulted to 0, with the
// prediction reported rather than assumed.

template <int BT, int L, int PAD>
__global__ __launch_bounds__(256) void scatter_shared_kernel(
    const int8_t* __restrict__ W, const uint8_t* __restrict__ CELLS,
    int* __restrict__ OUT, int N, int T, int K, int D, int stride) {
  const int DP = D + PAD;
  extern __shared__ int acc[];                  // [BT][DP]
  const int b0 = blockIdx.x * BT;
  const int nb = (N - b0 < BT) ? (N - b0) : BT;

  for (int i = threadIdx.x; i < BT * DP; i += blockDim.x) acc[i] = 0;
  __syncthreads();

  const int tiles = D / L;
  const long long work = (long long)nb * T * tiles;
  for (long long w = threadIdx.x; w < work; w += blockDim.x) {
    const int lt = (int)(w % tiles);
    const long long nt = w / tiles;
    const int t = (int)(nt % T);
    const int nl = (int)(nt / T);
    const int lane0 = lt * L;
    const uint8_t* c = CELLS + ((size_t)(b0 + nl) * T + t) * 3;
    const int sh1 = c[2] & 15, sh2 = c[2] >> 4;
    int* ap = acc + nl * DP + lane0;
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int sh = slot ? sh2 : sh1;
      if (sh == (int)DISCARD) continue;
      const int8_t* row = W + ((size_t)t * K + c[slot]) * stride + lane0;
#pragma unroll
      for (int j = 0; j < L; ++j) atomicAdd(ap + j, ((int)row[j]) << sh);
    }
  }
  __syncthreads();

  // plain stores: this block is the unique writer of tokens [b0, b0+nb)
  for (int i = threadIdx.x; i < nb * D; i += blockDim.x) {
    const int nl = i / D, l = i - nl * D;
    OUT[(size_t)(b0 + nl) * D + l] = acc[nl * DP + l];
  }
}

// ------------------------------------------------------------------ launchers

template <int L>
void launch_a(const torch::Tensor& W, const torch::Tensor& C, torch::Tensor& O,
              int N, int T, int K, int D, int stride, int threads) {
  const long long total = (long long)N * T * (D / L);
  const long long blocks = (total + threads - 1) / threads;
  TORCH_CHECK(blocks < (1LL << 31), "grid too large");
  scatter_global_kernel<L><<<(int)blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<int8_t>(), C.data_ptr<uint8_t>(), O.data_ptr<int>(), N, T, K, D, stride);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int BT, int L, int PAD>
void launch_b(const torch::Tensor& W, const torch::Tensor& C, torch::Tensor& O,
              int N, int T, int K, int D, int stride, int threads) {
  const size_t smem = (size_t)BT * (D + PAD) * 4;
  TORCH_CHECK(smem <= SMEM_LIMIT,
              "scatter-shared Bt=", BT, " L=", L, " pad=", PAD, " needs ", (long)smem,
              " B of shared memory (Bt * (D + pad) * 4) against the ", SMEM_LIMIT,
              " B per-block limit: over by ", (long)smem - SMEM_LIMIT,
              " B. Refusing rather than substituting another kernel.");
  auto k = scatter_shared_kernel<BT, L, PAD>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  const int blocks = (N + BT - 1) / BT;
  k<<<blocks, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<int8_t>(), C.data_ptr<uint8_t>(), O.data_ptr<int>(), N, T, K, D, stride);
  C10_CUDA_CHECK(cudaGetLastError());
}

void common_checks(const torch::Tensor& W, const torch::Tensor& CELLS, int64_t D, int64_t L,
                   int64_t threads, int& N, int& T, int& K, int& stride) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kChar && W.is_contiguous() && W.dim() == 2,
              "W must be contiguous int8 CUDA [T*K, stride]");
  TORCH_CHECK(CELLS.is_cuda() && CELLS.scalar_type() == torch::kByte && CELLS.is_contiguous() &&
                  CELLS.dim() == 4 && CELLS.size(1) == 1 && CELLS.size(3) == 3,
              "CELLS must be contiguous uint8 CUDA [N, 1, T, 3]");
  N = CELLS.size(0);
  T = CELLS.size(2);
  stride = W.size(1);
  K = W.size(0) / T;
  TORCH_CHECK(W.size(0) == (int64_t)T * K, "W rows must be T*K");
  TORCH_CHECK(D <= stride, "D must not exceed the row stride");
  TORCH_CHECK(D % L == 0, "L (", L, ") must divide D (", D, ")");
  TORCH_CHECK(threads > 0 && threads <= 1024 && threads % 32 == 0,
              "threads per block must be a positive multiple of 32 and <= 1024, got ", threads);
}

torch::Tensor scatter_global(torch::Tensor W, torch::Tensor CELLS, int64_t D, int64_t L,
                             int64_t threads) {
  int N, T, K, stride;
  common_checks(W, CELLS, D, L, threads, N, T, K, stride);
  auto out = torch::zeros({N, 1, D}, W.options().dtype(torch::kInt));   // arm A NEEDS this
#define A(l) if (L == l) { launch_a<l>(W, CELLS, out, N, T, K, (int)D, stride, (int)threads); } else
  A(1) A(2) A(4) A(8) A(16)
  { TORCH_CHECK(false, "no scatter_global instantiation for L=", L, " (have 1, 2, 4, 8, 16)"); }
#undef A
  return out;
}

torch::Tensor scatter_shared(torch::Tensor W, torch::Tensor CELLS, int64_t D, int64_t L,
                             int64_t BT, int64_t PAD, int64_t threads) {
  int N, T, K, stride;
  common_checks(W, CELLS, D, L, threads, N, T, K, stride);
  // UNINITIALISED on purpose. Arm B plain-stores every element it owns and the block is the
  // unique writer of its tokens, so nothing is ever read before it is written and the ~100.7 MB
  // memset arm A needs is exactly the cost this dataflow exists to avoid. If any element were
  // left unwritten this would surface as garbage in the gate -- which is why the gate poisons
  // the caching allocator with a sentinel before every arm B call.
  auto out = torch::empty({N, 1, D}, W.options().dtype(torch::kInt));
#define B(bt, l, p)                                                                      \
  if (BT == bt && L == l && PAD == p) {                                                  \
    launch_b<bt, l, p>(W, CELLS, out, N, T, K, (int)D, stride, (int)threads); } else
#define BL(bt) B(bt, 1, 0) B(bt, 2, 0) B(bt, 4, 0) B(bt, 8, 0) B(bt, 16, 0)
  BL(4) BL(8) BL(12) BL(16) BL(24)
  B(16, 4, 1) B(16, 4, 4) B(8, 4, 1)      // padded variants, for the bank-conflict question
  {
    TORCH_CHECK(false, "no scatter_shared instantiation for Bt=", BT, " L=", L, " pad=", PAD,
                " (have Bt in {4,8,12,16,24} x L in {1,2,4,8,16} at pad 0, plus a few padded)");
  }
#undef BL
#undef B
  return out;
}

// ------------------------------------------------------------------ launch-overhead floor
// An empty kernel, so the underfill study can quote the floor that no gridding change can beat.
__global__ void empty_kernel() {}

void empty_launch(int64_t blocks, int64_t threads) {
  empty_kernel<<<(int)blocks, (int)threads, 0, c10::cuda::getCurrentCUDAStream()>>>();
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_global", &scatter_global, "arm A: direct scatter, global atomics");
  m.def("scatter_shared", &scatter_shared, "arm B: direct scatter, shared accumulate");
  m.def("empty_launch", &empty_launch, "empty kernel, for the launch-overhead floor");
}
