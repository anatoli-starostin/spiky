# 1. Lookup table primitives (LUTs)

The primitive itself: how an input is turned into table addresses (hashing), how the output is formed from the rows that were read (readout), and how gradients flow back through both (backpropagation). The three pages are tightly linked; the third explains how gradients are calculated for the mechanisms of the first two.

- [Hashing mechanics](hashing.md): pairwise comparisons, input projections, ...
- [Ternary routing](ternary.md): hyperplanes with weights in {-1, 0, +1}, standardised weights, sparsity
- [Readout mechanics](readout.md): lookup-and-accumulate (LAC), dominance / Borda, weight quantisation, confidence, ...
- [Gradient backpropagation](backpropagation.md)
