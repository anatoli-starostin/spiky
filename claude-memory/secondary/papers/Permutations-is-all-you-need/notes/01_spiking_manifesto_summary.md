# Spiking Manifesto — Summary

**Paper:** "Spiking Manifesto" by Eugene Izhikevich, arXiv:2512.11843, Dec 2025.

## Core Thesis

Modern AI models are ~1000x less efficient than the brain. The gap comes from two sources:

1. **Processing efficiency (sparsity):** SNN neurons are silent unless activated; ANNs require constant GPU matrix multiplications on dense activations.
2. **Encoding efficiency (factorial capacity):** 60 neurons each firing one spike produce 60! possible timing patterns — astronomically more than the linear capacity of a 60-dim real-valued vector.

## The Fundamental Distinction

> "For ANNs, pairwise relationships *between* vectors are paramount (dot products, cosine similarity). For SNNs, relationships *among elements within each vector* are paramount, as these relationships correspond to different timing of spikes."

This is the key insight: ANN embeddings encode information in magnitudes; SNN embeddings encode information in the **relative ordering (permutation)** of elements.

## The Model

Input latency vector **x** ∈ R^n maps to output **y** through lookup tables:

1. Each table monitors n_c "anchor" neuron pairs (a_k, b_k).
2. **Compare** latencies: is x[a_k] > x[b_k]?
3. **Concatenate** binary results into an index j ∈ {0, ..., 2^{n_c} - 1}.
4. **Look up** synaptic weight vector S[j] from the table.
5. **Sum** contributions from all n_t tables: y = Σ_i S_{i, H_i(x)}.

This is index-based retrieval — a form of locality-sensitive hashing. The computation requires only comparators and table reads, no multiply-accumulate units.

## Gradients and Training

The lookup function is piecewise constant → zero gradients almost everywhere. Solution: **surrogate gradients** using a smooth "uncertainty" function U(u):
- U(u) ≈ 0 for large |u| (confident comparison)
- U(0) = 0.5 (maximally uncertain)

Each table contributes equal-magnitude, opposite-sign gradients to its two anchor neurons → self-regularisation with zero-mean gradients.

## Encoding Capacity

- **ANN:** Linear capacity scales as e^{c(ε)·n} where c(ε) ~ ε².
- **SNN:** Capacity is m^n where m ~ 1/ε (temporal bins).
- With n=16 dimensions: 16! = 2×10^{13} permutations vs R^{512} needed for equivalent ANN capacity.

## Architecture Examples

### Deep SNN (Residual)
x^{l+1} = x^l + S_{x^l} — replaces ReLU + MatMul with LUT. Index caching replaces activation caching for backward pass.

### Spiking RNN
h_t = S_{h_{t-1}} + z_t. Resilient to vanishing/exploding gradients because:
- Different hidden states select different LUT rows (no repeated matrix multiplication)
- Pairwise comparisons depend on relative ordering, not absolute magnitudes

### SNN Transformer
Replaces attention with pairwise concatenation [z_i, z_j, PE_{i-j}] fed into LUTs. No softmax needed. "SNN attention is all you need." Per-token memory bandwidth: ANN ~1M+ values vs SNN ~120 values — a 10,000-fold difference.

## Polychronisation vs Rank-Order Coding

Distinguishes from Simon Thorpe's rank-order coding (which used only first spikes). The manifesto requires relative order of *all* spikes: "Flipping the order of the last two spikes completely changes the identity."

## Spikes as Locality-Sensitive Hashing

The hash function H(x) partitions input space into 2^{n_t × n_c} non-overlapping buckets. Small perturbations stay in the same bucket → reduces noise sensitivity, decreases overfitting, promotes generalisation.

## Key Quotes

- "Spiking networks are nature's way of implementing look-up tables."
- "The encoding capacity of SNNs dwarfs that of ANNs."
- "For a 16-dimensional SNN embedding, 16! ≈ 2×10^{13} — astronomically larger than any finite-precision ANN can match in R^{16}."
