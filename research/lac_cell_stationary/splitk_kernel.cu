// Split-K over tables: the AS gather with its T-table loop cut into S chunks.
//
// Motivation is the step-1 underfill decomposition: AS's grid is ceil(N/block_n) x H blocks, and
// at D=1024 block_n is forced to 16, so the GPU does not fill until B ~ 2,720 tokens and the
// wall-clock is pinned near-constant at 0.133-0.145 ms for all 16 <= B <= 2048. Splitting the
// table loop multiplies the block count by S without touching block_n, which is the only knob
// that can fill the grid at small B.
//
// Each block accumulates a partial sum over ITS table chunk, in the same int32 registers the AS
// kernel uses, and combines with int32 atomicAdd into a pre-zeroed output. Integer addition is
// associative and order-independent, so bit-exactness against read_cells holds BY CONSTRUCTION
// for every S -- no float reassociation anywhere. No tensor cores.
//
// The geometry is deliberately identical to the AS baseline so the two are directly comparable:
// block_n = 16, UPR = D/16 = 64, 1024 threads per block, load16=false (the char4 path with a real
// DISCARD skip). At S = 1 the kernel performs exactly AS's work, differing only in that it
// atomically adds into a zeroed buffer instead of storing -- which is the self-check.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace {

constexpr uint8_t DISCARD = 15;
constexpr int SMEM_LIMIT = 101376;

// S is a RUNTIME argument, not a template parameter: only the block shape is templated, and the
// table range is computed from blockIdx.z. One instantiation serves every S.
template <int BLOCK_N, int UPR>
__global__ __launch_bounds__(BLOCK_N * UPR) void splitk_kernel(
    const int8_t* __restrict__ W, const uint8_t* __restrict__ CELLS, int* __restrict__ OUT,
    int N, int H, int T, int K, int D, int stride, int tpc) {
  extern __shared__ uint8_t csh[];            // [tpc][BLOCK_N][3], this chunk's cells only
  const int nthr = BLOCK_N * UPR;
  const int ltok = threadIdx.x / UPR;         // token within the tile
  const int unit = threadIdx.x % UPR;         // 16-lane unit within the row
  const int n0 = blockIdx.x * BLOCK_N;
  const int h = blockIdx.y;
  const int t0 = blockIdx.z * tpc;
  const int tn = (T - t0 < tpc) ? (T - t0) : tpc;    // tables in this chunk

  // ---- stage this chunk's cells, same [t][ltok][3] layout the AS kernel uses ----
  for (int u = threadIdx.x; u < BLOCK_N * tn; u += nthr) {
    const int lt = u / tn, tt = u - lt * tn;
    const int gt = n0 + lt;
    const size_t dst = ((size_t)tt * BLOCK_N + lt) * 3;
    if (gt < N) {
      const size_t src = (((size_t)gt * H + h) * T + t0 + tt) * 3;
      csh[dst] = CELLS[src];
      csh[dst + 1] = CELLS[src + 1];
      csh[dst + 2] = CELLS[src + 2];
    } else {
      csh[dst + 2] = DISCARD | (DISCARD << 4);       // both cells dead for a tail token
    }
  }
  __syncthreads();

  const int tok = n0 + ltok;
  if (tok >= N) return;                       // safe: no barrier follows
  const int lane0 = unit * 16;
  const int nl = (D - lane0 < 16) ? D - lane0 : 16;

  int acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = 0;

  const size_t pitch = (size_t)K * stride;
  for (int t = 0; t < tn; ++t) {
    const uint8_t* cell = csh + ((size_t)t * BLOCK_N + ltok) * 3;
    const int sh1 = cell[2] & 15, sh2 = cell[2] >> 4;
    const size_t tb = (size_t)(t0 + t) * pitch;
    for (int r = 0; r < 2; ++r) {
      const int sh = (r == 0) ? sh1 : sh2;
      if (sh == (int)DISCARD) continue;       // the load16=false form: a real skip
      const char4* c4 = reinterpret_cast<const char4*>(
          W + tb + (size_t)cell[r] * stride + (size_t)lane0);
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

  // ---- combine. Result unused so ptxas can emit REDG (fire-and-forget), not ATOMG. ----
  int* op = OUT + ((size_t)tok * H + h) * D + lane0;
  for (int j = 0; j < nl; ++j) atomicAdd(op + j, acc[j]);
}

torch::Tensor splitk(torch::Tensor W, torch::Tensor CELLS, int64_t D, int64_t S) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kChar && W.is_contiguous() && W.dim() == 2,
              "W must be contiguous int8 CUDA [T*K, stride]");
  TORCH_CHECK(CELLS.is_cuda() && CELLS.scalar_type() == torch::kByte && CELLS.is_contiguous() &&
                  CELLS.dim() == 4 && CELLS.size(3) == 3,
              "CELLS must be contiguous uint8 CUDA [N, H, T, 3]");
  const int N = CELLS.size(0), H = CELLS.size(1), T = CELLS.size(2), stride = W.size(1);
  const int K = W.size(0) / (T * H);
  TORCH_CHECK(W.size(0) == (int64_t)T * H * K, "W rows must be H*T*K");
  TORCH_CHECK(D > 0 && D <= stride && D % 16 == 0,
              "D must be positive, a multiple of 16, and <= the row stride; got D=", D,
              " stride=", stride);
  TORCH_CHECK(S >= 1 && S <= T, "S must be in [1, T]; got S=", S, " with T=", T);
  TORCH_CHECK(T % S == 0, "S must divide T exactly (no ragged table chunks are implemented); "
              "got S=", S, " with T=", T);

  constexpr int BLOCK_N = 16;
  const int upr = (int)(D / 16);
  TORCH_CHECK(upr == 64,
              "this arm is pinned to the AS baseline shape: D must be 1024 so UPR = D/16 = 64; "
              "got D=", D, " -> UPR=", upr, ". Refusing rather than running a different shape.");
  const int nthr = BLOCK_N * upr;
  TORCH_CHECK(nthr <= 1024, "block_n * UPR = ", nthr, " exceeds 1024 threads per block");

  const int tpc = (int)(T / S);
  const size_t smem = (size_t)tpc * BLOCK_N * 3;
  TORCH_CHECK(smem <= SMEM_LIMIT, "split-K S=", S, " needs ", (long)smem,
              " B of shared memory (tables-per-chunk ", tpc, " * block_n ", BLOCK_N,
              " * 3) against the ", SMEM_LIMIT, " B limit: over by ",
              (long)smem - SMEM_LIMIT, " B. Refusing rather than substituting another kernel.");

  auto out = torch::zeros({N, H, (int64_t)D}, W.options().dtype(torch::kInt));
  auto k = splitk_kernel<BLOCK_N, 64>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  dim3 grid((unsigned)((N + BLOCK_N - 1) / BLOCK_N), (unsigned)H, (unsigned)S);
  k<<<grid, nthr, smem, c10::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<int8_t>(), CELLS.data_ptr<uint8_t>(), out.data_ptr<int>(),
      N, H, T, K, (int)D, stride, tpc);
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

// Read-only: {blocks_per_sm, regs, static smem, local, maxThreads} for the real instantiation.
std::vector<int64_t> splitk_occupancy(int64_t dyn_smem) {
  const void* fn = (const void*)splitk_kernel<16, 64>;
  cudaFuncAttributes a{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, fn));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, 1024,
                                                               (size_t)dyn_smem));
  return {blocks, a.numRegs, (int64_t)a.sharedSizeBytes, (int64_t)a.localSizeBytes,
          a.maxThreadsPerBlock};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("splitk", &splitk, "split-K over tables; S is a runtime argument");
  m.def("occupancy", &splitk_occupancy, "read-only occupancy/attributes of the instantiation");
}
