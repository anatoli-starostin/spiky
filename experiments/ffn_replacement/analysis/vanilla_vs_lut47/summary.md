# Vanilla vs LUT47 — what changes when the MLP FFN becomes a LUT

**Models** (both `model_build.MinimalGPT`, E=384, attention H=6, depth=6, RoPE, ClimbMix, same tokenizer):
- **A) Vanilla** — standard dense-MLP FFN. Run `exp_n_0216_vanilla48k_gelu_ckpt2k_seed1`, val_bpb **1.1151**.
  (The task named `exp_n_0177_vanilla48k_corrected`, but that folder has **no checkpoint.pt**; 0216 is the
  substitute — a 48K vanilla with an essentially identical val_bpb 1.115094 vs 0177's 1.115092 — and it has weights saved.)
- **B) LUT47** — FFN replaced by a light CompressionMultiHeadLUT (H8/tph64, nap8, read_top_n=2, float read, dropout+TV).
  Run `exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph64_h8_seed1_noquant_headdrop20`, val_bpb **1.1079** (beats vanilla by 0.0072).

Both checkpoints load with missing=0/unexpected=0 and run forward passes; all activations captured via forward hooks
on the identical attention path and on the FFN slot. Shared eval batch: 8×256 real ClimbMix val tokens (same tokens to both).

## 1. Attention patterns  (attn_entropy_heatmaps.png, attn_examples.png)
The attention *architecture* is identical, so any systematic difference is **demanded by the FFN**.
- **The LUT model's attention is sharper (lower entropy).** Mean attention entropy 3.10 bits (LUT) vs 3.60 (vanilla).
  Per-layer mean entropy: vanilla [5.82, 3.27, 3.17, 2.52, 3.21, 3.63] vs LUT [5.67, **2.17**, **2.34**, 2.73, 3.15, **2.57**].
  Layer 0 is diffuse in both (~5.7 bits, positional smearing); the LUT sharpens most at layers 1, 2, 5.
- **Induction heads exist in both**, comparable strength: vanilla top L3H1=0.135, LUT top L4H0=0.141 — similar magnitude,
  slightly relocated. Max previous-token attention 0.575 (vanilla) vs 0.487 (LUT).
- On a real sentence (see the trace), the LUT model's final-layer heads are extremely peaked (attend ~0.90 to one token)
  where the vanilla heads spread their mass across 2–3 tokens.
**Takeaway:** a discrete/low-rank LUT read wants cleaner, more decisive inputs, so the shared attention learns to be sharper.

## 2. Embeddings  (embed_norms.png, embed_svspectrum.png)
Separately trained, so absolute directions don't align — comparing *structure*:
- **Token embeddings**: near-full-rank and near-isotropic in both (effective rank / participation ratio ≈ 342 (vanilla) vs
  333 (LUT) out of 384; 90% energy at ~289/282 dims; isotropy — mean cosine to the mean vector — 0.14 vs 0.17, i.e. well spread).
  The LUT model's token space is marginally lower-rank.
- **Untied unembedder**: **low-rank and highly anisotropic in BOTH, and nearly identical** — effective rank ≈ 63, 90% energy
  by ~44 dims, isotropy ≈ 0.89 (a dominant shared direction, the usual LLM "rogue dimension"). Vanilla 63.0 vs LUT 63.1.
**Takeaway:** swapping the FFN barely touches the embedding/unembedding geometry; the read-in is high-rank/isotropic and the
read-out is low-rank/anisotropic in both — a model-agnostic property here.

## 3. FFN vs LUT internals  (ffn_delta.png, ffn_sparsity.png) — the biggest differences
- **The LUT FFN writes much smaller residual updates, and the model runs at a smaller residual scale.**
  Mean ||FFN delta|| per layer: vanilla [5.5, 3.3, 3.6, 4.5, 5.9, **12.8**] vs LUT [0.6, 1.6, 1.1, 1.3, 1.9, **5.4**].
  Mean residual norm: vanilla up to 18.2, LUT up to 9.9. Ratio ffn/resid ≈ 0.4–0.95 (vanilla) vs **0.2–0.55 (LUT)** — the dense
  MLP does substantially more of the residual-writing.
- **The LUT FFN's updates are low-rank, especially early.** Effective rank of the per-layer FFN-delta (across tokens):
  vanilla [225, 111, 261, 259, 249, 139] vs LUT [**5.5, 7.4**, 188, 194, 184, 24]. Early LUT layers write into a ~5–7 dimensional
  subspace — the signature of a discrete lookup (few distinct cell reads → low-rank output) — rising mid-network, dropping at L5.
- **Sparsity:** the LUT per-head read is sparser than the dense GELU hidden, most at layer 0 (16.6% near-zero vs <1%).
**Takeaway:** the LUT is a lighter, lower-rank, sparser operator than the MLP; it moves the residual less and in fewer directions.

## Single-sentence trace  (sentence_trace.md, trace_ffn_delta.png)
Side-by-side on `The capital of France is` and `The cat sat on the mat.` — next-token top-5, per-layer attn/FFN/residual norms
at each position, and per-head attention. It confirms the aggregate picture: on the France prompt the LUT FFN's final-layer
update is 2.38 vs the vanilla 10.38 (ffn/resid 0.19 vs 0.53), and the LUT's layer-5 heads attend ~0.90 to the last token
while the vanilla heads spread across 'is'/'of'/'capital'. (Neither small ClimbMix model actually predicts "Paris" — both are
weak; this is about mechanism, not factual recall.)

## Overall
Replacing the MLP with a LUT FFN **shifts work out of the FFN** (smaller, lower-rank, sparser residual writes) and **demands
sharper attention**, while leaving the token/unembedding geometry essentially unchanged — and it slightly **wins** on val_bpb
(1.1079 vs 1.1151). The LUT behaves like a decisive, low-rank router that offloads mixing onto a crisper attention path.

## Figures
- `attn_entropy_heatmaps.png` — per-(layer,head) attention entropy: vanilla | LUT | diff.
- `attn_examples.png` — example attention heatmap (L2 H0) for both models on the shared batch.
- `embed_norms.png` — per-token L2-norm distributions (token-embedding and unembedder).
- `embed_svspectrum.png` — singular-value spectra + effective rank (token-embedding and unembedder).
- `ffn_delta.png` — per-layer FFN residual-update norm and its effective rank.
- `ffn_sparsity.png` — per-layer FFN activation sparsity.
- `trace_ffn_delta.png` — FFN residual-update norm across layers on the France sentence's final token.
