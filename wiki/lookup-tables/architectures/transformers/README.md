# Transformers

The main line of LUT architectures: language models in which the attention block, the feed-forward block, or both are built from lookup tables. The pages below are organised by idea, not by date. The ideas overlap in time (several were tried in parallel in spring 2026), so the order of the pages is not a timeline; see [How the line moved](../history.md) for that.

Scales are always stated with the context length and the vocabulary size. *GPT-2 scale* means context 2048 with vocabulary 32768; context 512 with vocabulary 32768 is the fast NanoChat test scale; the early work was byte-level with a small vocabulary (257) and context 32 to 128.

**The picture in short.**

- The language-model work starts byte-level (vocabulary 257, context 32 to 128) and moves to the NanoChat corpus (vocabulary 32768, context 512 for fast test runs, 2048 at GPT-2 scale).
- Attention models that need no magnitudes (ranking attention) and a pure permutational transformer were tried first; the design that survived is LUT SDPA with a dual stream (a rank-coded stream plus a parallel Euclidean one), described in the June 2026 report; later tests used a single stream.
- The line that reached GPT-2 scale keeps dense attention and replaces only the feed-forward block. At the fast test scale it is close to the dense model in loss; at d24 it is behind (validation bits per byte 0.7794 against 0.7193, CORE 0.196 against 0.255, one seed each).
- RoPE works only with scaled-dot-product attention; fixed positional encodings combine with any attention model but are poor at scale. The output layer stayed dense and untied.

**Pages.**

- [LUT attention models](lut-attention.md): attention in which queries, keys and values come from tables, including LUT SDPA.
- [Permutational transformers](permutational.md): models that rely only on rankings of vectors, ranking attention, the dual stream, magnitude leakage.
- [Positional encodings](positional-encodings.md): fixed encodings and RoPE.
- [FFN replacement](ffn-replacement.md): classic attention, LUT feed-forward blocks; the line that reached GPT-2 scale.
- [Unembedders](unembedders.md): LUT output heads and tying.
- [Geometry and mechanistic interpretability of LUT models](geometry-and-interpretability.md): open problems, never studied in depth.

**Terms.** *LUT SDPA*: LUT queries and keys feed classic scaled-dot-product scores, LUT values are real vectors blended by the softmax, and the blend is read permutationally by the output table. *Dual stream*: a permutational stream plus a parallel Euclidean stream of higher dimension (later removed, replaced by a single stream). *Old evaluation*: the validation window depended on the training batch size; see [`CONVENTIONS.md`](../../CONVENTIONS.md).
