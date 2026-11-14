# Lookup tables in Spiky

This area of the [Spiky wiki](../README.md) is a guide to what was tried with lookup-table (LUT) models in this repository: the LUT primitives, how they are optimised, the architectures built from them, and the problems around them. Each page tells the story at an abstract level (ideas, models, math, what worked and what did not) and links to the code, runs and notes in this repository for the details. Pages carry evidence labels (see [`CONVENTIONS.md`](../CONVENTIONS.md)), so start with the section that matches your interest from the map below.

**Status: in progress.** The lookup table primitives and the optimisation pages are written; the architectures and the lookup-related problems (lutwalker2d) are written.

## Where it all started

- **Thesis** (Izhikevich, *Spiking Manifesto*, 2025): the brain encodes information in the order of spikes; algorithms based on sorting and permutations can compete with matrix multiplication.
- **Primitive:** the differentiable lookup table, an abstract model of changes in spiking activity, without modelling spiking neurons.

## What we need from the primitive

1. It is based on lookup tables (lookup is the essence of what the brain keeps doing).
2. It is energy efficient: super efficient on specialised lookup hardware, ideally also a gain on GPUs.
3. It allows models at least on par with modern deep-learning architectures.

## The research framework

This is wide-area research to find that primitive. For now we deliberately restrict ourselves to primitives that have

- some **input hashing** logic, and
- some logic that forms the output from the hashing results (the **readout** mechanism),

and that **allow backpropagation**, so they can be used inside deep-learning pipelines.

Each idea becomes a PyTorch implementation and is tested on different ML problems (artificial or real), usually against a classic matmul baseline, with the potential energy win estimated for a silicon implementation. Applying LUTs to a problem is rarely a blind replacement of matmuls in a standard model: lookup tables force different internal representations to emerge, so it is a creative process.

Sometimes we step outside the framework to study problems that correlate with lookup tables, for example [converting trained LUTs into a classic spiking network](lookup-related-problems/luts-to-spikes.md).

## Map

| Section | What it covers |
| --- | --- |
| [1. Lookup table primitives](lookup-table-primitives/README.md) | [Hashing](lookup-table-primitives/hashing.md), [readout](lookup-table-primitives/readout.md), [gradient backpropagation](lookup-table-primitives/backpropagation.md) |
| [2. Optimisation techniques](optimisation/README.md) | [Basic optimisers](optimisation/basic-optimisers.md), [training efficiency](optimisation/training-efficiency.md), [normalisation](optimisation/normalisation.md), [regularisation](optimisation/regularisation.md), [special LUT optimisers](optimisation/special-optimisers.md) |
| [3. LUT architectures](architectures/README.md) | [Convolutional networks](architectures/convolutional.md), [transformers](architectures/transformers/README.md), [other ideas](architectures/other-ideas.md) |
| [4. Lookup-related problems](lookup-related-problems/README.md) | [LUTs to spikes (lutwalker2d)](lookup-related-problems/luts-to-spikes.md) |

Also: [Spiky wiki root](../README.md); [`CONVENTIONS.md`](../CONVENTIONS.md) (how to write pages, math, citing sources).
