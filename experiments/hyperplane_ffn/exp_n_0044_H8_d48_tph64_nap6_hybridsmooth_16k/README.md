# exp_n_0044 — single-slot H8/d48, tph64, nap6, **hybrid_smooth** forward, tied, 16k

Clone of **exp_n_0039_H8_d48_tph128_16k** with **two changes: tph 128 → 64 AND forward mode hard →
hybrid_smooth** (config key `lut_forward_mode`). Single-slot CompressionMHL, H8, d48/48, nap6, tied,
learnable_temps=true, joint=false, no MeanAbsNorm/Lion, plain AdamW (0033 grouping). Everything except tph +
forward_mode (+ exp_name) is identical to 0039.

**Forward-mode findings (read from `src/spiky/lutorch/fast_multi_head_lut.py`):**
- Config key = **`lut_forward_mode`**; allowed values (validated against `_FORWARD_MODES`, raises otherwise) =
  **`("hard", "hybrid_smooth")`**.
- **`hard`** (default): hard sign-pack lookup — each table picks exactly ONE row from the sign-bit packing of the
  pairwise anchor differences; one row per table.
- **`hybrid_smooth`**: the forward is a **top-2 soft blend of the main (hard-selected) row and its Hamming-1
  alternate row at the least-confident anchor pair** — i.e. a weighted average of the selected row and the single
  neighbouring row you'd get by flipping the most-marginal sign bit, weighted by a sigmoid of the routing margin
  over the select temperature T_sel. So it's a 2-row blend at the decision boundary instead of a hard 1-row
  lookup (backward is soft in BOTH modes; hybrid_smooth's weight-grad is a 2-row scatter vs hard's 1-row).
- **hybrid_smooth adds NO parameters** — it reuses the same table weights + learnable temps; only the forward
  compute differs. (Confirmed: param count identical to a hard tph64 build.)

**Params = 27,343,200 (SMOKE-confirmed).** = exp_n_0033's 27,343,296 − 96 (H8 has 96 learnable-temp scalars vs
H16's 192). Per-component:
- **LUT tables: 9,437,184** (nodecay) = 8·64·2⁶·48/layer × 6 — **exactly 1× exp_n_0033's** 9,437,184 (8·48 =
  16·24 = 384, same H·d·2^nap·tph budget).
- **Compress+decompress weights: 1,769,472** (2-D, decay) — UNCHANGED from 0039 (H·d=384).
- **tok_emb (tied): 12,582,912** + **attn qkv/proj: 3,538,944** → decay(2-D) total **17,891,328**.
- nodecay total = **9,451,872** = tables 9,437,184 + temps 96 + LayerNorm 1-D 9,984 + proj biases 4,608.
- = 1.178× tied dense (same as exp_n_0033).

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=9,451,872 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=9,437,184 in nodecay]`

Differs from exp_n_0039 config in EXACTLY three keys: `lut_tables_per_head` (128→64), `lut_forward_mode`
(hard→hybrid_smooth), `exp_name`. Not launched yet (built + SMOKE-passed only); queued after exp_n_0043.
Question: at the leanest table budget (1× 0033), does the softer hybrid_smooth forward (2-row blend) beat hard
routing — i.e. is the smooth forward a free quality lever?
