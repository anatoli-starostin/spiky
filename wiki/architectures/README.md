# 3. LUT architectures

This section describes the model families built on lookup-table layers: what each one computes, how well it worked, and what we know so far about the representations the models learn (a question we are only starting to answer). Replacing a matrix multiplication by a table lookup is not a drop-in swap, so every page starts from the idea, then points to the code ("Where it lives", links pinned to commits of [`anatoli-starostin/spiky`](https://github.com/anatoli-starostin/spiky)). Numbers state their setting (for language models the context length and vocabulary size), and those from old branches carry the evaluation caveat of [`CONVENTIONS.md`](../CONVENTIONS.md). The picture in short comes first; the lists below lead to the details.

## The picture in short

- We tried LUT layers on two kinds of problems: image classification (MNIST, CIFAR-10) and language modelling. The convolutional work came first and stopped in March 2026; language models have been the main line since.
- The ideas are organised by idea, not by date, because they overlap in time; [How the line moved](history.md) gives the order.
- For images, LUT layers that read patches reach 99.06% on MNIST; on CIFAR-10 the best all-LUT network (86.05%, claimed in a journal) is a few points behind a convolutional baseline run in different code.
- For language models, the question is how much of the network can be built from tables: first attention alone, including variants that use only the order of values; then the feed-forward block, which is the line that reached GPT-2 scale and is behind the dense model at d24 (0.7794 against 0.7193 bits per byte).
- A few other architectures were tried once and have no conclusion yet ([Other ideas](other-ideas.md)); recursive setups are an open area.

Problems we tried:

- [Convolutional networks](convolutional.md): MNIST and CIFAR-10; LUT layers that read image patches (closed in March 2026).
- [Transformers](transformers/README.md): the main line.
  - [LUT attention models](transformers/lut-attention.md), including LUT SDPA (LUT queries and keys into classic scaled-dot-product scores, LUT values)
  - [Permutational transformers](transformers/permutational.md): rankings only, ranking attention, the dual stream, magnitude leakage
  - [Positional encodings](transformers/positional-encodings.md): fixed encodings and RoPE
  - [FFN replacement](transformers/ffn-replacement.md): classic attention, LUT feed-forward; the line that reached GPT-2 scale (context 2048, vocabulary 32768)
  - [Unembedders](transformers/unembedders.md): LUT output heads and tying
  - [Geometry and mechanistic interpretability of LUT models](transformers/geometry-and-interpretability.md) (open problems)
- [Other ideas](other-ideas.md): predictive-coding LUT stacks, a byte-level recursive LUT, and recursive setups as an open area (not started).

Related studies, not architectures: the Walker2D control problem is in [Lookup-related problems](../lookup-related-problems/README.md).

For the order in which the ideas appeared, see [How the line moved](history.md).
