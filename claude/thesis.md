# The thesis behind spiky / LUTGPT

*The scientific "why" of the project. The single most authoritative source is the **LUTGPT
research report** (`doc/lutorch/lutgpt_research_report.pdf`). The experimental arc that
tested all of this is in [experiment-journey.md](experiment-journey.md).*

## The premise (Izhikevich's "Spiking Manifesto", arXiv:2512.11843, Dec 2025)

Brains are ~1000× more energy-efficient than ANNs, attributed to two gaps: **sparsity**
(spiking neurons stay silent unless activated; ANNs run dense matmuls) and **encoding**.
The key idea: **ANNs encode information in the *magnitudes* of a vector; spiking nets encode
it in the *relative ordering (permutation)* of elements** (spike timings).

Ordering capacity is factorial (n elements → n! distinguishable patterns; 16! ≈ 2×10¹³) and
is invariant to scaling, shifting, and monotone transforms — exactly where magnitude
capacity collapses under normalization, quantization, and noise. Hence **"Permutations Are
All You Need"**: rebuild the transformer so every learned projection depends only on the
input's *ordering*, computed by comparators + table reads — **no multiply-accumulate**.
The biological ancestor is spike **polychronization** / polychronous groups
(Izhikevich, 2006).

## The primitive: differentiable lookup table (LUT)

A layer maps x → y in two steps:

1. **Encode.** N fixed "anchor pairs" (aⱼ, bⱼ); bitⱼ = 𝟙[x_{aⱼ} > x_{bⱼ}]; pack the N bits
   into a row index b\* ∈ {0 … 2ᴺ−1}. This index is **rank-coded and magnitude-blind** — it
   depends only on which of each pair is larger.
2. **Decode.** Gather a learned row y = W[b\*] from a table W ∈ ℝ^{2ᴺ × D_out}. Sum over
   `tph` tables per head.

Key parameters and design facts:

- **NAP (n_c)** = anchor pairs per table → K = 2^NAP rows. Controls per-table discrimination
  at *exponential* cost.
- **tph** = tables per head, summed. Adds capacity at *linear* cost.
- **Anchors are FIXED at init** (balanced full-coverage sampling), never learned — learning
  them was tried and refuted.
- **Hard forward** = one sign-pack + gather, zero multiplies. This is the manifesto-pure,
  deployable operation.
- **Soft / hybrid-smooth forward** = blend the main row with its least-confident Hamming-1
  neighbor (lower loss, 2 reads — the deployable approximation of a full softmax over rows).
- **Backward is the load-bearing trick:** an *honest* weight gradient (only the rows the
  forward actually touched) combined with a *soft surrogate* gradient for the input and
  temperature (backprop as if the forward were the full-K softmax y = Σ π(b)·W[b], through
  two learned temperatures T_soft, T_sel). A naive one-row straight-through backward still
  trains, but converges materially worse — the **full-K soft backward is a necessary
  condition**, not an optimization.
- **MeanAbsNorm before each LUT** pins mean(|x|) ≈ 1 so the pairwise differences stay in
  T_soft's calibrated regime. It is a training stabilizer; the hard forward is scale-invariant,
  so it does not change decoding.

## LUTGPT (the report's model)

A 6-layer transformer with a **dual residual stream**:

- an **E-stream** — the rank-coded backbone; every consumer is a LUT + MeanAbsNorm; mutated
  by `out_proj`;
- a **D-stream** — an ordinary Euclidean residual (width D = 384), read out by
  LayerNorm + a **linear unembedder** → logits.

Four LUT roles: `qk_lut` → SDPA (the one operation that genuinely needs magnitudes; RoPE on
top), `v_lut` → the attention convex sum, `out_proj` → E-stream (it **absorbs the FFN** —
there is no separate FFN), and `emb_resid_lut` / `residual_lut` → D-stream. Two real matmuls
remain (SDPA and the linear unembedder), so **LUTGPT is a hybrid, not fully matmul-free**.
Core primitive: `FastMultiHeadLut` in `src/spiky/lutorch/fast_multi_head_lut.py`; the narrow
config lives under `examples/lutgpt/`.

## Headline result

Setup: nanochat scale — ClimbMix, T = 512, V = 32,768, ~3.93×10⁸ tokens, single H100.
Metric = **validation bits-per-byte (bpb)**, lower is better, token-matched to a vanilla RoPE
transformer.

| model | params | val bpb |
|-------|--------|---------|
| vanilla (exp709) | 35.79M | **1.1922** ← baseline |
| LUTGPT full-width (exp754, hybrid_smooth) | 276.8M (7.7×) | **1.1842** ← beats vanilla by 8 mb |
| LUTGPT narrow (exp755) | 176.2M (4.9×) | 1.2044 |
| LUTGPT native-hard, 24K steps (exp760) | — | 1.2048 (ships hard) |

So the LUT family **lands on vanilla's side of the loss curve at matched data and token
budget** — an *existence proof*, **not** a parameter win. A parameter win is impossible by
construction (a K·tph table vs. one dense matrix). Soft → hard-eval costs +65–73 mb; native
hard training plus ~50% more steps closes that gap.

## Efficiency thesis (why bother, given it is slower on GPUs)

LUTGPT is **slower in wall-clock on every GPU** (5–8× params, a dense soft-surrogate
backward, gathers that don't use tensor cores). The bet is on future **rank-coded /
sparse-lookup / in-memory-compute hardware**, where its real properties matter:

- **~3–3.8× less HBM read per token** (a layer reads H·tph·D_out bytes, *independent of K*);
- **~20× higher compression density** (parameters stored per byte fetched);
- a **bounded, data-independent sparse active set** (<5% of parameters per token, decided by
  sign comparisons at compile time).

It is explicitly *not* claimed that such hardware exists yet.

## Companion papers (same "Permutations Are All You Need" line of work)

- **article v1 (Izhikevich / Kluchnikov / Starostin):** byte-level FineWeb, T = 32.
  Introduces **Ranking Attention** (Kendall-τ ≡ cosine of ±1 dominance signatures → standard
  SDPA on rank features) and **HyperLUT** (Kluchnikov: replace the 2^NAP table with a small
  MLP on the comparison bits; GPU-matmul-friendly, avoids table blowup). Best LUT within 0.025
  CE of vanilla, but at 26× params.
- **article v2 (Izhikevich / Starostin):** the sharpest result — make the transformer
  **purely combinatorial** (sum → concat, cosine → Kendall-τ, matmul → aggregated LUT reads
  via Borda counting on a dominance matrix), then **BitPermutationLUT** quantizes every weight
  to a **single bit** and reaches val CE **1.203 = vanilla exactly** (~25.6 MB bit-packed).
  The reading: the combinatorial reformulation finds the right representation, and once found,
  the precision requirement collapses to 1 bit with no quality loss.

## Open frontier (unsolved)

- LUT-based **attention** — reproduce softmax's data-dependent resolution inside a rank
  primitive.
- LUT-based **unembedder** — produce magnitude logits over a 32k vocab from a rank
  representation (the reason the D-stream exists today).
- HyperLUT → discrete-LUT distillation; MultiBitPermutationLUT (K-bit); longer context.
