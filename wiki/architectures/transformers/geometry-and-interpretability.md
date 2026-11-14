# Geometry and mechanistic interpretability of LUT models

> **Open problem.** We have never conducted a full-fledged study of this topic. This page is a list of what it would be interesting to study, with a few places in the repository that show what a first look can be. None of those is research on the topic.

## Idea

A trained dense model can be read: probes, logit lens, the singular spectra of weights, circuits. A trained LUT model is a different kind of object, a set of tables addressed by signs of comparisons, and it is not known what can be read out of it or how.

## Open questions

- *What internal representations emerge in LUT models?* What does a residual stream carry (the rank stream and the Euclidean stream of the [dual stream](permutational.md#the-dual-stream) are different objects), and what does a layer add to it?
- *What can we say about the table weights?* What a cell represents, whether rows of one table are semantically related, which anchor pairs matter, and how the heads of one layer divide the input.
- *How is the address space used?* How unevenly the cells are hit, whether the model uses the tables as a discrete code or as a continuous feature, and how this changes with training.
- *What does the geometry of the gradients and of the weights look like,* and how does it differ from dense layers? Does it explain which optimisers suit tables?
- *What is the role of depth?* Which layers do the work, which are hard for a LUT to imitate, how attention differs by layer, and whether trained solutions are modular.
- *Can the usual tools of mechanistic interpretability be carried over* (probes, activation patching, logit lens, sparse features), and what would be the analogue of a circuit for a model whose operations are comparisons and lookups?

## Examples of a first look

These are one-time analyses of single checkpoints or single runs, made for other purposes. They show what the tools can give, and they are not findings about LUT models in general.

- A measurement of the table gradients of the d24 LUT feed-forward on one batch at a few checkpoints, against the dense feed-forward: [spectra README](https://github.com/anatoli-starostin/spiky/blob/b38b57d70f4c30f877888ac148f9d9e60495b00d/experiments/lut_nanochat/results/lut-grad-spectra-analysis/README.md), [analysis scripts](https://github.com/anatoli-starostin/spiky/tree/b38b57d70f4c30f877888ac148f9d9e60495b00d/experiments/lut_nanochat/analysis/lut_grad_spectra). It motivated the table Muon analogue ([Basic optimisers 2](../../optimisation/basic-optimisers.md#2-muon)); see also [FFN replacement](ffn-replacement.md#6-gpt-2-scale-d24-nanochat-october-to-november-2026).
- A comparison of the singular spectra of the projections under Muon and AdamW in a fork of the d24 model: [forks analysis](https://github.com/anatoli-starostin/spiky/blob/b38b57d70f4c30f877888ac148f9d9e60495b00d/experiments/lut_nanochat/analysis/2026-10-16_forks/README.md).
- A static analysis of the weights of the dense d24 model, for which there is no matching analysis of LUT weights: [Stage A notes](https://github.com/anatoli-starostin/spiky/blob/b38b57d70f4c30f877888ac148f9d9e60495b00d/experiments/lut_nanochat/results/d24-dense-1xh100-s0/analysis/stageA_notes.md).
- A probe of the magnitudes of the soft signs, found to be spread up to 0.7 and never saturated: [Permutational transformers, magnitude leakage](permutational.md#magnitude-leakage).
- A logit lens over depth on two trained models that differ only in batch size: [LUT attention](lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768).
