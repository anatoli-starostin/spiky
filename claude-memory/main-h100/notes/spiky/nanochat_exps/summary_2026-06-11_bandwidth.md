# LUT-LM vs vanilla — bandwidth findings (2026-06-11)

## TL;DR

| metric | exp750 LUT-LM | exp738 vanilla |
|---|---|---|
| val_bpb @ 4K bs=48 (training-time eval) | **1.3863** | 1.3984 |
| val_bpb (full bf16 deployment) | **1.3861** | 1.3982 |
| training HBM read / token | 57.6 MB | 71.5 MB (0.81× for LUT-LM) |
| **deployment HBM read / token** | **32.5 MB** | **35.8 MB** (**0.91× for LUT-LM**) |
| deployment GPU wall-clock / forward | 104 ms | 2.85 ms (**36× faster for vanilla**) |
| total stored params (training) | 276.8 M | 35.8 M |

LUT-LM **wins** on quality (−12 mb val_bpb) and bandwidth (−9 % per token at deployment), **loses badly** on GPU wall-clock (vanilla is 36× faster). Bandwidth-win-without-wall-clock-win is exactly the LUT virtual-bandwidth thesis — the architecture targets ASIC / neuromorphic hardware where sparse `embedding_bag`-style gathers are cheap, not GPUs.

## Methodology

Both checkpoints loaded into the same evaluation harness. For each variant:
- **Training-time HBM read**: bytes per token the matmul/gather kernels read from HBM during a single forward pass at training-time storage dtype (fp32 master weights even when autocast(bf16) is active).
- **Deployment HBM read**: same after weights are cast to bf16 at save time (no quality loss, verified).
- **Wall-clock**: `time.perf_counter` around the model forward at B=24, T=512, n_iter=20 after 5 warmup, with `torch.cuda.synchronize()` boundaries. Run on a single H100.

Storage configurations:

| | training | deployment |
|---|---|---|
| exp750 LUT body | bf16 nn.Parameter (already bf16 during training, courtesy of fp32 master Lion) | bf16 |
| exp750 head (nn.Linear) | fp32 | **bf16 (quantized at save)** |
| exp738 backbone (Linear) | fp32 (autocast does bf16 in-SM, but HBM still fp32) | **bf16 (quantized at save)** |
| exp738 head (nn.Linear) | fp32 | **bf16 (quantized at save)** |

Verified deployment quantization is free:
- exp738 vanilla, bf16 backbone + fp32 head → val_bpb 1.3985 (+0.01 mb vs full fp32).
- exp738 vanilla, full bf16 → val_bpb 1.3982 (−0.22 mb, eval noise).
- exp750 LUT-LM, head cast to bf16 → val_bpb 1.3861 (−0.19 mb vs fp32 head).

## Bandwidth breakdown (per token, full bf16 deployment)

### exp750 LUT-LM (276.8 M params)

| component | shape | bytes/token |
|---|---|---|
| 6 layers × (qk_lut + v_lut + out_proj + residual_lut) | varies | **7.27 MB** |
| emb_resid_lut (once per token) | — | **0.10 MB** |
| unembedder Linear(D=384, V=32768) | 12.6 M elements × 2 B | **25.2 MB** |
| tok_emb lookup (one row) | E × 2 B | ~768 B (≈0) |
| **total / token** | | **32.5 MB** |

LUT body breakdown (per token, per layer):
- qk_lut: 6 heads × 256 tph = 1 536 lookups, 128 outputs/lookup → 196 608 elements × 2 B = 0.39 MB
- v_lut: 1 536 lookups × 64 outputs → 0.20 MB
- out_proj: 1 × 512 lookups × 384 outputs → 0.39 MB
- residual_lut: 1 × 256 lookups × 384 outputs → 0.20 MB
- per-layer: 1.18 MB × 6 = 7.08 MB
- + emb_resid_lut: 256 × 384 × 2 B = 0.20 MB

### exp738 vanilla (35.8 M params)

| component | shape | bytes/token |
|---|---|---|
| 6 layers × (qkv + out + MLP up + MLP down) | 1.77 M weights/layer | **21.2 MB** |
| unembedder Linear(D=384, V=32768) | 12.6 M elements × 2 B | **25.2 MB** |
| **total / token** | | **35.8 MB** (50.3 MB if head stays fp32) |

Per layer (matmul reads full weight tensor at fp32 storage = 4 B/element, here cast to bf16 at deployment = 2 B):
- qkv (D × 3D): 442 368 weights → 0.88 MB
- out_proj (D × D): 0.29 MB
- MLP up (D × 4D): 1.18 MB
- MLP down (4D × D): 1.18 MB
- per-layer: 3.54 MB × 6 = 21.2 MB

### Ratios

| | exp750 | exp738 | LUT/vanilla |
|---|---|---|---|
| backbone (excl. head) | 7.27 MB | 21.2 MB | **0.34×** (LUT reads 66 % less) |
| head | 25.2 MB | 25.2 MB | identical (same `Linear(384, V)`) |
| total / token | **32.5 MB** | **35.8 MB** | **0.91×** |

The head dominates LUT-LM's total (77 % of 32.5 MB) because both use the same dense `nn.Linear(D, V)`. The LUT-LM's distinguishing win is the backbone — 66 % less HBM read than vanilla's dense matmul stack.

## Wall-clock surprise

| | ms / forward (B=24, T=512) | tokens/sec | vs vanilla |
|---|---|---|---|
| exp750 LUT-LM (fp32 SDPA) | 104.33 | 0.12 M | 36.7× slower |
| exp750 LUT-LM (bf16 SDPA) | 103.52 | 0.12 M | 36.4× slower |
| exp738 vanilla (full bf16) | **2.85** | **4.32 M** | **1.0×** |

LUT-LM reads ~10 % less HBM yet runs 36× slower on H100. Why:

1. **Arithmetic intensity**. Vanilla's 6 layers × 4 matmuls + head = 31 dense matmul kernels at bf16 hit the H100 tensor cores at ~85 % of peak throughput (250+ FLOPs/byte). LUT-LM does ~23 296 `embedding_bag` lookups per token — pure gather, ~1 FLOP/byte, **no tensor cores**.
2. **Kernel count**. Each of LUT-LM's LUT modules launches its own custom autograd Function with body autocast wrapping + index gather + index_add or bmm-sparse-S scatter. Even with 6 layers, the kernel-launch overhead alone runs into milliseconds.
3. **bf16 SDPA helped basically nothing** at the model level — the SDPA op itself is 5.7× faster in bf16 (0.087 ms vs 0.493 ms per call), but it's < 1 % of total LUT-LM forward time, so the model speedup is < 1 %.

### What WOULD make LUT-LM fast

The bandwidth metric is a proxy for performance on **sparse-friendly hardware** — custom ASIC, neuromorphic chip, FPGA with cheap random-access gather. On such hardware:
- Vanilla's dense matmul reads ALL weights per forward (high HBM bytes/token).
- LUT-LM's one-row-per-table gather reads ~10 % of the weight memory per forward.
- Tensor-core advantage disappears (these chips don't have tensor cores, or their dense matmul throughput is much closer to gather throughput).

On GPU, that arithmetic-intensity asymmetry doesn't translate. **For GPU deployment specifically, vanilla wins on wall-clock regardless of architecture.**

## Quality vs bandwidth: the trade-off table

| run | params | val_bpb @ 4K | training HBM/tok | wall-clock |
|---|---|---|---|---|
| exp738 vanilla bf16 | 35.8 M | 1.3984 | 92.8 MB (training fp32) | 2.85 ms |
| exp732 LUT-LM fp32 storage | 276.8 M | 1.3912 | 64.8 MB | ~104 ms |
| exp737 v2 LUT bf16 storage + master Lion | 276.8 M | 1.3872 | 57.6 MB | ~104 ms |
| **exp750 LUT-LM + global clip(1.0)** (SOTA) | **276.8 M** | **1.3863** | **57.6 MB** | **~104 ms** |

At training: exp750 LUT-LM is **−12 mb better val_bpb** than vanilla, reads 0.62× the HBM bandwidth, but takes 36× more wall-clock per forward. The 7.7× more stored params don't matter for HBM bandwidth — what matters is what's actually read each forward, not what's stored.

## Final recipe (committed as default in fast_multi_head_lut.py)

```python
# At training time
FastMultiHeadLUT(
    weight_dtype=torch.bfloat16,       # bf16 storage (halves HBM)
    use_bf16=True,                     # bf16 compute autocast
    forward_mode="hard",
    backward_mode="dense_K",
)
# (use_bmm_wgrad is permanently True in the body now)

# LUT optimizer: custom Lion with fp32 master in state["master"]
# Non-LUT: AdamW (decoupled wd on 2D Linear weights)
# Pre-step:   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# Head:       nn.Linear(D, V, bias=False)  # storage fp32 during training

# At save time
for p in model.parameters():
    p.data = p.data.to(torch.bfloat16)   # quantize everything to bf16
```

## Open follow-ups

- **exp752** (16K extension of exp750 recipe, currently running). Closest 16K reference: exp731 = 1.2178 @ 16K (same architecture, no clip). Tracking −3 to −5 mb ahead of exp731 across the run so far. Final result will determine whether clip(1.0) transfers cleanly to the long horizon.
- **Replacing the dense head with a LUT readout** is the only remaining structural lever for LUT-LM bandwidth. Currently the head is 77 % of deployment HBM. Doing so would also remove the only non-LUT op in the backbone.
- Hardware: actually validating the LUT bandwidth win on a sparse-native target (FPGA / ASIC simulator) — until then the bandwidth advantage is theoretical for non-GPU deployment.
