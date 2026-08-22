# exp_n_0077 — Is the 1×→1.5×-batch LUT gain mostly ATTENTION or the LUT-FFN?

Diagnostic (no training): load the two trained checkpoints, graft weight groups
between them, eval val_bpb on a fixed val set. Complementary attention-pattern compare.

**Checkpoints** (same arch, 27,343,200 params: 6L / 384 / 6-head attn; LUT H8/d48/
tph64/nap6; tied dense):
- 1× (~1.22): `exp_n_0052_H8_d48_tph64_batched_control_16k/checkpoint.pt` (batched LUT).
- 1.5× (1.196862): `exp_n_0046_H8_d48_tph64_bs1p5x_16k/checkpoint.pt` (per-head *loop*
  LUT: `ffn.luts.0..7`). Converted to the batched layout for a common model (concat the
  8 heads' tables; offset head h's anchors by h·48 into the flat 384-space). Hard-forward
  is temp-independent, so this conversion is forward-equivalent — self-validated by the
  1.5× sanity eval reproducing ~1.1969.

**Weight groups (by state_dict key):**
- `attn` = `blocks.N.attn.{qkv,proj}.weight` + `blocks.N.ln1.{weight,bias}` (pre-attn norm).
- `ffn`  = `blocks.N.ffn.*` (tables+anchors+compress+decompress+temps) + `blocks.N.ln2.{weight,bias}`.
- `other`= `tok_emb.weight`, `ln_f.{weight,bias}`, `head.weight`.

**Part A** — 2×2 graft + symmetric reverse: eval `1x_pure`, `1.5x_pure`,
`1x⊕(attn←1.5x)`, `1x⊕(ffn←1.5x)`, and the two reverse grafts. Attribution = fraction
of the (1x−1.5x) gap recovered by each graft. **Caveat handled:** the two models were
trained independently, so a naive cross-model graft can break composability — reported
honestly (both directions; sanity runs must reproduce the known bpb, else the harness is off).

**Part B** — attention patterns on a fixed val batch, per head: entropy, look-back
distance, local (|i−j|≤8) vs long-range fraction, BOS-sink fraction; 1× vs 1.5×.

Wraps modules only; no shared-src edits. Outputs `attn_attribution.json` + `.png`.
