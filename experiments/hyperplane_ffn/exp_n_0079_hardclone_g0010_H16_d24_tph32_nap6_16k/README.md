# exp_n_0079 — hard-routing clone of exp_g_0010

Exact clone of **exp_g_0010_H16_d24_tph32_nap6_hybridsmooth_16k**, changing only the LUT
forward mode from **`hybrid_smooth` → `hard`**. Everything else is identical: architecture
(depth 6, n_embd 384, 6 attn heads, seq 512), LUT config (H16 / inner-dim 24 / tables_per_head 32
/ nap 6, learnable temps, tied dense), batch (device 48, total 24,576 tok/step, grad_accum 1),
schedule (lr 3e-4 cosine, warmup 10%, wd 0.1, betas (0.9,0.95)), and **16,000 steps**.

**Config diff vs exp_g_0010** (only these two lines):
- `exp_name`: …_hybridsmooth_16k → exp_n_0079_hardclone_g0010_H16_d24_tph32_nap6_16k
- `lut_forward_mode`: `hybrid_smooth` → `hard`

Trainer = the maintained shared flexible trainer (byte-identical to exp_n_0052's train.py).
For this config it takes the per-head loop LUT path (no `lut_batched_multi_head_input` key,
default False) — same code path exp_g_0010 ran. **Param count verified identical: 22,624,704.**

**Purpose.** exp_g_0010 (hybrid_smooth) reached val_bpb 1.22671 but saved NO checkpoint.
This run provides the hard-routing counterpart at the same ~22.6M size WITH a reloadable
checkpoint, and measures the hybrid_smooth-vs-hard gap at this H16/d24 geometry.

Reference: exp_g_0010 (hybrid_smooth) final val_bpb **1.22671** (best 1.22663).
Outputs: `metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt`.
