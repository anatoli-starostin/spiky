# `spiky/lutorch/csrc` — CUDA sources for the fused routing+gather path

Two categories, and the distinction is load-bearing:

## Active — built on demand by `spiky.lutorch.fast_mhl_cuda_gather`

| file | what |
|---|---|
| `gather_fused.cu` | Fused routing + gather, index kept in shared. The RTX 5090 path. |

This is the only file `fast_mhl_cuda_gather.load()` ever compiles, and even that happens lazily on
first use — never at import. It is reached only when `fast_mhl_cuda_gather.patch()` is called AND
the detected device is 5090-class (see `is_5090_class_gpu()`).

## Passive — shipped, never compiled, never dispatched to

| file | what | measured on H100 |
|---|---|---|
| `h100_prototypes/gather_fused_v2_h100.cu` | Fused v2-routing + gather, index in shared | routing+gather 0.141 / 0.253 / 0.150 ms (0126/0127/0128); best H100 path found, still ~1.09–1.82× the vanilla dense slot |
| `h100_prototypes/route_v2_h100.cu` | Standalone routing, column-major shared z + token-inner map (bank-conflict-free) | 1.97× native at nap8; **regresses** to 0.68× at nap7 |
| `h100_prototypes/route_shared_h100.cu` | Standalone routing, z staged in shared, row-major | 1.33–1.43× native at nap7; superseded by the fused path |

**Nothing in `h100_prototypes/` is compiled, imported, or referenced by the dispatch.**
They are source-only, carried here so the H100 work is preserved in-tree and available for
later experimentation on the nebius H100 box — not because anything switches to them. On
H100 (and every other non-5090-class device) `fast_mhl_cuda_gather.patch(mode="auto")` is a no-op
and the model stays on its existing shipping gather path.

Why they are inert rather than wired up: the paper's own measurement is that this family of
kernels is **slower than the vanilla dense FFN on an H100** — routing is compute-bound
there, not memory-bound, and H100's dense matmuls are already near tensor-core peak. There
is no device for which turning these on is currently the right default. They were also
never wired into a callable Python API on the benchmark side ("integration TBD" in the
original notes), so activating them would need real integration work first, not just a
dispatch-table entry.

Full H100 optimization sweep, including the dead ends (L2 residency pinning, cp.async
double-buffering, tensor-core GEMM, `embedding_bag`, `torch.compile`), is in
`experiments/ffn_replacement/benchmark/H100_OPT_NOTES.md` on the `feature/ffn_replacement`
branch.

## If you do want to build one (manual, deliberate)

```python
from torch.utils.cpp_extension import load
ext = load(name="gather_fused_v2_h100",
           sources=["<...>/csrc/h100_prototypes/gather_fused_v2_h100.cu"],
           extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math"],
           extra_cflags=["-O3", "-std=c++20"])
```

`-std=c++20` is required on **both** flag lists: at the default standard this torch build
fails compiling its own `ATen/core/List_inl.h`. Nothing in these kernels needs C++20.

Each kernel binds through `TORCH_EXTENSION_NAME`, so the `name=` argument above decides the
module name and they never collide. The exposed entry points are `fused_v2`, `route_v2`,
and `route_shared` respectively; all three are bit-exact against the native routing +
Triton gather reference, and all assume the block-diagonal per-head anchor layout
(`multi_head_input=True`, head `h` reading z columns `[h*48, h*48+48)`).
