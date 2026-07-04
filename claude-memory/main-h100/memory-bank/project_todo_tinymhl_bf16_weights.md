---
name: TODO — TinyMHL bf16 weights for training (multi-alt path)
description: Investigate why bf16 weights are slower than fp32 in TinyMHL multi-alt STE+noise training; identify if hybrid storage or custom kernels can recover the expected memory/perf win.
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
## Empirical finding (2026-05-12)

For `TinyMultiHeadLut(backward_mode='ste', n_alternatives=3, argmax_noise_eps>0)`
at nanochat shapes, `weight_dtype=torch.bfloat16` is **WORSE** than
`weight_dtype=torch.float32` on both axes:

| Shape | fp32 weights | bf16 weights | Δ |
|---|---|---|---|
| NAP=6 (out_proj: 192→96, tph=2048) | 17.19 ms / 3.39 GB | 25.89 ms / 3.73 GB | +50% time, +10% mem |
| NAP=8 (V-lut: 96→32, tph=256)      | 26.63 ms / 1.66 GB | 28.93 ms / 1.90 GB |  +9% time, +14% mem |

(Numbers from a GPU-shared bench, but ratios are clean.)

## Why bf16 weights lose

1. **bf16 atomic add is slow on H100.** Our backward does
   `weights_grad_flat.index_add_(0, indices_main, grad_per_table.reshape(...))`
   which is a heavy atomic-accumulation kernel. PyTorch's bf16 path either
   serialises or uses a slower atomic variant — much worse than fp32 atomics.
2. **Dtype-mismatch overhead.** `grad_per_table` arrives as fp32 (grad of
   fp32 output). With fp32 weights everything is consistent; with bf16
   weights, conversions are needed at scatter time.
3. **No tensor-core win from bf16 storage.** Our hot ops are random-access
   gather + scatter + element-wise — NOT GEMM. Tensor cores only kick in
   inside the `autocast` block around the structured-bmm einsum, which
   already runs in bf16 regardless of `weight_dtype`. So bf16 weights
   don't unlock new tensor-core compute.
4. **Inductor materialises bf16→fp32 promotion buffers** in some paths,
   actually growing peak memory.

## TODOs

- **Hybrid storage**: keep weights as fp32 master, materialise a `bf16`
  view only for the structured-bmm gather (`weights.to(bf16)` inside the
  autocast). Halves bmm HBM read traffic without bf16-atomic penalty.
- **Avoid bf16 index_add_**: do the weights gradient scatter in fp32 to a
  temp buffer, cast to bf16 at apply-time (or use Adam's fp32 master copy).
- **bf16 atomic-friendly kernel**: SM_80+ has `__bfloat162` packed atomics;
  a custom CUDA kernel could regain throughput.

## How to apply

When proposing memory optimisations for TinyMHL training, don't reach for
`weight_dtype=torch.bfloat16` — current implementation makes it strictly
worse than fp32. The bf16 default in the constructor is for
**inference-only** small models (CIFAR-10 LUT nets), not training.
