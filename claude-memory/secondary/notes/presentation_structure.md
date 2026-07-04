# Purely Combinatorial Transformers

*From the Spiking Manifesto to bit-quantized permutation LUTs*

---

## 1. Starting point: the Spiking Manifesto

Izhikevich's core claim — the brain's computational advantage lies in combinatorial structure (orderings of spikes), which modern deep learning does not exploit. His proposed replacement for matrix multiplication: differentiable lookup tables.

## 2. First experiment: reproducing the manifesto model

Built `spiky`, a specialized PyTorch CUDA library, and implemented a variant of the manifesto architecture (a modification Eugene shared directly).

- **Result:** loss competitive with a vanilla transformer on the same task.
- **Problem:** ~100× larger than the vanilla baseline.

## 3. A deeper problem surfaces

While looking for ways to shrink the model, a theoretical issue became clear: the manifesto model is not actually purely combinatorial. Parts of it still rely on float magnitudes rather than pure orderings. This contradicts the philosophical premise — that computation should depend only on rank structure, the way spike orderings do in the brain.

## 4. Reformulating the math: classical ↔ combinatorial parallels

If computation must depend only on orderings, every building block needs a combinatorial analog. Three parallels emerge:

- **Summation → concatenation** (summing combinatorial objects is ill-defined).
- **Cosine distance → Kendall tau distance** (magnitude-free).
- **Matrix multiplication → a set of lookup tables** applied to a combinatorial vector.

## 5. Representing orderings

Two equivalent representations, each useful in different contexts:

- **Materialized ranking** — a float vector whose magnitudes exist only to encode an order.
- **Dominance matrix** — pairwise "which element beats which" representation.

Conversions between these spaces are building blocks of the architecture.

## 6. First purely combinatorial model

**Components:**

- RankAttention (attention with Kendall tau instead of cosine).
- Dominance projections at conversion points.
- Multi-head LUTs (from the manifesto) as the core rank-to-rank transform.
- No residual connections — instead, all per-layer embeddings are concatenated before the vocabulary projection, which is cheap because embeddings are small (dim 32).

**Result:** competitive loss with vanilla transformers, and by construction purely combinatorial — no magnitude leakage.

**Remaining problem:** still large, hundreds of millions of parameters.

## 7. The weight-reduction idea: PermutationLUT

A small but consequential tweak to Izhikevich's LUTs. His LUTs project directly into materialized-ordering space, which forces weights to be tuned with high precision to produce the right sort order.

**PermutationLUT** instead projects into dominance space: each LUT weight is reinterpreted as a vote for a dominance relation ("element i should rank above element j"). Votes are aggregated, then converted back to an ordering.

**Key observation:** dominance space is larger than embedding space (496 relations for dim 32), but votes are sparse, so this is not a bottleneck.

## 8. The payoff: extreme quantization

Because each weight is just a vote, PermutationLUT tolerates aggressive quantization.

- **BitPermutationLUT:** single-bit weights. Training uses FP16 latents (required for gradients), but once trained, the latents are discarded — the deployed model runs on 1-bit weights with no quality loss.
- A custom Adam variant was built for this training regime.

## 9. Current work: MultiBitPermutationLUT

An intermediate regime — small discrete weight sets (e.g. 16 values ≈ 4 bits). Biologically motivated: real synapses are not high-precision, but they are not binary either. Under active study.

## 10. The arc in one slide

Spiking Manifesto → reproduction → identification of the magnitude-leakage flaw → combinatorial math framework → first purely combinatorial model → PermutationLUT reframing → bit-quantized models → multi-bit regime.
