# exp_n_0051 — reconstruction-auxiliary CompressionMHL (whole-module mirror), H8/d48/tph64, 16k

> **STATUS: code-before-run (smoke-tested, full 16k run pending owner confirmation).** Tests whether a
> training-only reconstruction auxiliary — forcing the FFN slot to be near-lossless — speeds convergence
> vs the free-row control **exp_g_0006 (1.228335)** at the matched standard 16k rung.

## Idea
Each block's forward FFN slot is a `CompressionMultiHeadLUT` (input `[N,384]` → output `[N,384]`). During
**training only**, we attach a **mirror** `CompressionMultiHeadLUT` of identical structure/capacity (same
H8/d48/tph64/nap6 hard, its own compress/luts/decompress, its own seed) that maps the forward slot's
**output → a reconstruction of its input**:

```
recon = mean over blocks of MSE(mirror(forward_output), forward_input.detach())
total_loss = task_CE + lambda_recon * recon        # lambda_recon = 0.1
```

The recon gradient flows back into `forward_output` (pushing the forward slot toward invertible / near-lossless
codes); the input target is **detached** so the encoder isn't pushed to collapse to a trivially-reconstructable
constant. One mirror CMHL per block (6 total).

## Inference is unchanged
The mirrors are **not** referenced in `model.forward`, so val/bpb (CE) is computed on the **plain** forward
CompressionMHL. Mirrors are trainable during training (added to the optimizer, LUT tables → nodecay group),
and are **dropped from the saved checkpoint** (`recon.*` keys excluded) — the shipped model is the unchanged slot.

## Config (matches exp_g_0006 / exp_n_0047 forward exactly)
H8 / d48 / tph64 / nap6, hard routing, device_bs 48, grad_accum 1, **16000 steps**, warmup 1600, seed 1,
clean val 245,760 tokens (eval_steps 10 × bs 48 × seq 512). Forward CMHL forced to the **independent loop path**
(`lut_batched_multi_head_input=false`) so it is bit-identical to exp_g_0006 (per-head temperatures) — the batched
default would share temps across heads and perturb the surrogate-gradient dynamics of the baseline. No edits to
the shared modules `fast_multi_head_lut.py` / `compression_mhl.py`; the mirror is just a second CMHL in train.py.

## Smoke test (40-step, real model size)
Builds ✓; mirror trains ✓; **recon finite and decreasing** (raw 0.65 → 0.39 over 40 steps after the zero-init
decompress transient) ✓; val/bpb path clean and mirror-excluded ✓.
Param counts: **forward trainable 27,343,200** (== exp_g_0006), **mirror aux trainable 11,211,360**,
**total trainable 38,554,560**.
