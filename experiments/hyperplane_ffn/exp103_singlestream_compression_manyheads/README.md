# exp103 — many routing-heads (decoupled from attention heads), MAX-RANK config

Clone of exp102 that DECOUPLES the CompressionMHL routing-head count from the attention-head
count, to attack exp102's low-rank routing bottleneck (see the exp102-vs-exp023 analysis:
exp102's per-attention-head routing was confined to one shared 48-dim subspace).

**Idea:** q/k/v use **48 routing mini-heads**, each with its OWN learned compression W_c →
its own ≤48-dim routing subspace (`joint_head_compression=False`), emitted as `[N,48,64]`
via `multihead_output=True`. The 48 mini-heads are then **grouped 8→1** into the 6 attention
heads (`[N,48,64] → [N,6,8,64] → sum over the 8 → [N,6,64]`), then RoPE + causal SDPA as
exp102. So each attention head's routing now draws from **8 independent 48-dim subspaces =
up to 8×48 = 384-dim (full-rank)** routing, vs exp102's single 48-dim shared subspace.

**Sites** (all with `batched_multi_head_input=True` explicitly set, `forward_mode="hard"`,
`use_bf16=True`):
- q_lut, k_lut: CompressionMultiHeadLUTMH, n_heads=48, inner_in=48, inner_out=−1, nap=4, tph=4, out_width=64.
- v_lut: same, nap=6, tph=8.
- out_proj: CompressionMultiHeadLUT (unchanged from exp102): n_heads=8, inner_in=48, inner_out=48, nap=6, tph=128 → [N,384].

**Verified param counts (build+fwd/bwd smoke; grouped shapes [N,6,64] ✓, logits ✓):**
- q_lut 6,501,900 · k_lut 6,501,900 · v_lut 14,759,436 · out_proj 20,648,460 · rest 25,172,736
- **TOTAL 73,584,432 (~73.6M)**. Compression is ~40% of the q/k/v budget (48 independent
  W_c per site); this MAX-RANK run keeps inner_in=48 for full-rank routing.

Training: single-stream, batch 48 seq (24,576 tok/step), 16k steps, Lion-on-LUT-tables /
AdamW split, seed — all identical to exp102/exp101. Shared lutorch NOT modified (local
`mh_compression.CompressionMultiHeadLUTMH`).

**Targets to beat:** exp023 (single-stream Hyperplane, full-rank independent) 1.20632;
exp101 (all-tph32) 1.30213. This tests whether recovering routing rank via many independent
compressions closes exp102's gap.

Outputs: metrics.csv, summary.json, loss.png, checkpoint.pt.
