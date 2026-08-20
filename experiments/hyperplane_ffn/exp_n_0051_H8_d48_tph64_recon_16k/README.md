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

## Config
H8 / d48 / tph64 / nap6, hard routing, device_bs 48, grad_accum 1, **16000 steps**, warmup 1600, seed 1,
clean val 245,760 tokens (eval_steps 10 × bs 48 × seq 512). No edits to the shared modules
`fast_multi_head_lut.py` / `compression_mhl.py`; the mirror is just a second CMHL in train.py.

**Forward slot uses `batched_multi_head_input=true` (the batched FastMHL path).** With `learnable_temps=true`
this means the per-head temperature is **shared** — a single `(T_soft, T_sel)` pair across all 8 heads — rather
than per-head as in the exp_g_0006 loop-path control. So this run is **NOT bit-comparable to exp_g_0006 at the
temperature level**; it is the intended batched config. (The hard forward and per-head surrogate gradients are
otherwise identical to the loop path up to float reassociation; only the shared-temperature parametrization
differs — see the batched-vs-loop audit.) The mirror CMHL also uses the batched path (identical structure).

## Smoke test (real model size, batched forward)
Builds ✓; mirror trains ✓; **recon finite and decreasing** (after the zero-init decompress transient) ✓;
val/bpb path clean and mirror-excluded ✓.
Param counts (batched path): **forward trainable 27,343,116**, **mirror aux trainable 11,211,276**,
**total trainable 38,554,392**. (The batched path has 84 fewer params than the loop path — one shared
`(T_soft, T_sel)` pair per block instead of 8 per-head pairs.)
