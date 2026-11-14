# Positional encodings

A causal language model needs to know where a token is. In classic transformers positional encodings usually exploit an operation in Euclidean space: a fixed or trainable embedding is added to the input, and RoPE rotates the queries and keys. In permutational spaces it is not easy to find an analogue of such an operation. Two kinds of positional information were used.

- *Fixed positional encodings* (a vector per position that is not trained) combine with any model of attention. We used them for a long time; they are known to be not good at large scale.
- *RoPE* is a technique that works only with scaled-dot-product attention (SDPA), so with classic attention and with [LUT SDPA](lut-attention.md#lut-sdpa), but not with the other attention models of this wiki, and it is very effective.

We know of no analogue of RoPE for rank-only attention (see [Permutational transformers](permutational.md#open-ideas)).

**The state of the work in short.** The early positional models were interesting, but we tested them only at byte level; some of them might work at GPT-2 scale. When RoPE was tried it was kept as the baseline, because it is simple and universal and works well at GPT-2 scale. There was no deep and detailed ablation of different approaches to positional encodings, and this page should be read as a record of what happened, not as a comparison of methods.

**Scales.** The encodings below were tried at byte level (vocabulary 257, context 32) and on the NanoChat BPE data (vocabulary 32768, context 512, the fast test scale of [Transformers](README.md)). No positional-encoding comparison exists at GPT-2 scale (context 2048). All numbers are old-evaluation numbers of [`CONVENTIONS.md`](../../CONVENTIONS.md), single seeds, comparable only inside their own batch of runs; run names collide between the byte-level folder ([`transformer_exps`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps)) and the BPE folder ([`nanochat_exps`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps)), so the folder is named.

## Idea / model

**Fixed positional encodings.** The first code of the line, the toy transformer of the *Spiking Manifesto*, has a learned vector per relative offset and table, turned into address bits by its sign ([code_from_Izhikevitch.c](https://github.com/anatoli-starostin/spiky/blob/f453ae40708c44a316a030d8e23570cf13921980/spiky_cuda/lut/code_from_Izhikevitch.c#L188-L204)); the pairwise LUT attention of March 2026 used buckets of the relative distance $j - i$, linear and logarithmic, as an extra feature of the pair (issue 40, 2026-02-09). The default in the byte-level series is a fixed random vector per position, drawn once and not trained, either *added* to the embedding or *concatenated* to it (a quarter or a half of the model width is kept for it), as in [`exp009`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp009_lut_vanilla_mirror) ([train.py](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp009_lut_vanilla_mirror/train.py)). Concatenation gives the tables dedicated address coordinates for position, which keeps position from interfering with the content comparisons.

**RoPE.** RoPE rotates pairs of coordinates of $q$ and $k$ by an angle proportional to the position, so that the dot product $q_i \cdot k_j$ depends on the relative offset $i-j$ only. It needs real vectors and a real inner product, which is why it works with SDPA and with the real-valued Q and K emitted by the tables of LUT SDPA, and acts after the table, in the smooth space where SDPA lives. The notes on RoPE for LUT models are [here](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/claude-memory/main-h100/memory-bank/project_rope_lut_lm.md). The dense baseline of the later work uses RoPE too.

## What was found

**Byte level, before RoPE** (context 32, vocabulary 257, validation cross-entropy; numbers from the `summary.json` files and the [SUMMARY](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/SUMMARY.md)). Only the rows inside one table are comparable: the rows of a table are runs of the same code with the one change named in the first column, one seed each. The two tables differ in model size and in the number of steps (100,000 and 50,000), so their numbers must not be compared with each other.

*214M parameters, tables for all projections, 100,000 steps:*

| Encoding | Run | Result |
| --- | --- | --- |
| fixed random, concatenated | 214M, tables for all projections ([`exp009`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp009_lut_vanilla_mirror)) | 1.4791 |
| fixed random, added | same ([`exp010`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp010_lut_additive_pe)) | 1.4926 |
| none | same ([`exp011`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp011_lut_no_pe)) | 1.6618 |

*5.0M parameters, full-coverage anchors, 50,000 steps:*

| Encoding | Run | Result |
| --- | --- | --- |
| fixed random, added to the embedding | [`exp066`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp066_mini_tph256_nap4_fullcoverage) | 1.6839 |
| derived from a permutation of the positions | [`exp070`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp070_mini_tph256_nap4_fc_posperm) | 1.9411 |
| none | [`exp071`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp071_mini_tph256_nap4_fc_nopos) | 1.9286 |
| learned soft permutation | [`exp072`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/exp072_mini_tph256_nap4_fc_learnedperm) | 1.8475 |

**BPE data, before RoPE.** The dual-stream models added a learned per-layer vector (width 64) to the input of the joint Q and K table; this was the position encoding of the best model until 2026-05-14 ([`exp303`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp303_resid_tph2048_only), 1.6509, 602M parameters).

**RoPE.** On 2026-05-14, two comparisons, both at 8,000 steps, context 512, vocabulary 32768:

- dense [`MinimalGPT`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp001_minimal_gpt/train.py#L121-L157), 23M parameters, learned absolute position replaced by RoPE with nothing else changed: 1.6256 to 1.5468 ([`exp001`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp001_minimal_gpt), [`exp319`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp319_minimal_gpt_rope)), a gain of 0.0788;
- the dual-stream LUT model, 602M parameters, the per-layer learned additive position replaced by RoPE on $q$ and $k$: 1.6509 to 1.5933 ([`exp303`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp303_resid_tph2048_only), [`exp321`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp321_dense_rope)), a gain of 0.0576, visible from step 200 on and stable to the end.

([exp001](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp001_minimal_gpt), [exp319](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp319_minimal_gpt_rope), [exp303](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp303_resid_tph2048_only), [exp321](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp321_dense_rope), [notes](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/claude-memory/main-h100/memory-bank/project_rope_vs_learned_pos_emb.md).) The note on the first comparison observes that RoPE costs no parameters and is therefore better at this scale. The gain is 0.0788 for the dense and 0.0576 for the LUT model, and the dense model with RoPE still beat the 602M LUT model by 0.0465. Every later model of both lines uses RoPE, and the FFN line uses it by default without an ablation. What is not measured: RoPE against sinusoidal, ALiBi or no encoding on BPE data, anything at GPT-2 size, and the speed claim (it was asserted, not timed); RoPE on the sign signatures of ranking attention was never run.

## Dead ends

- **Position from a permutation of the positions** (the encoding native to rankings): no better than none in the 5M byte-level runs (one model, one seed).

## Open ideas

- A positional encoding for rank-only attention. RoPE rotates coordinate pairs of real vectors and needs a real inner-product score, so the plus-or-minus-one operands of [`BitAttention`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_attention.py#L324-L361) would not survive it; an analogue may exist that we did not find.
- A clean comparison of RoPE against other encodings on the BPE data, and at GPT-2 scale.

## Where it lives

All links are pinned to commits of [`anatoli-starostin/spiky`](https://github.com/anatoli-starostin/spiky).

| What | Link |
| --- | --- |
| relative-position buckets in pairwise LUT attention | [lut_attention.py](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/lut_attention.py#L51-L109) |
| [`PositionalPermutation`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L427-L449), [`LearnedSoftPermutations`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L452-L480) | [ranking_tools.py](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py) |
| [`RotaryEmbedding`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/examples/lutgpt/train.py#L168-L180), [`apply_rope`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/examples/lutgpt/train.py#L188-L192) (published form) | [examples/lutgpt/train.py](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/examples/lutgpt/train.py) |
| byte-level series (all encodings of the first table) | [transformer_exps](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps) |
