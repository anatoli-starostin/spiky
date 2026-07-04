---
name: tinymhl-hybrid-smooth-hard-eval-bug
description: "TinyMultiHeadLut(hybrid_smooth) and TAPL (the hard STE path) pack lookup bits in OPPOSITE orders. Naive hard-eval of a hybrid_smooth checkpoint reads the wrong LUT rows -> catastrophic gap (1.73 bpb on exp720). Fix: bit-reverse the weight tensor along axis 1."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# TinyMHLut hybrid_smooth → hard eval needs bit-reverse perm

## The bug

`TinyMultiHeadLut` has two forward paths that index into the same `self.weights` tensor
(shape `(n_heads * tph, 2^NAP, n_outputs)`, row axis = dim 1):

- **`backward_mode = 'hybrid_smooth'`**: computes `lookup_index = Σ bits[i] * 2^(NAP-1-i)`
  using `self.soft_powers = _msb_powers(NAP)` (`src/spiky/lutorch/tiny_multi_head_lut.py:4040`).
  This is **MSB-first** packing.

- **`backward_mode = 'ste'`** (and the hard STE path that ships forward via TAPL): TAPL's
  `forward` computes `lookup_index = Σ bits[i] * 2^i` via
  `powers = arange(0, NAP)` and `(bits << powers).sum`
  (`src/spiky/lutorch/tiny_anchor_pairs_lookup.py:317, 82`). This is **LSB-first** packing.

For the same input, **`msb_pack(bits) ≠ lsb_pack(bits)` except on bit-palindromic indices** —
so when a model is trained with `backward_mode='hybrid_smooth'` and you simply flip
`backward_mode = 'ste'` at eval time, every LUT picks the wrong physical row. The model
collapses.

## Empirical proof

Eval'd the exp720 checkpoint (`exp720_pure_soft_bs96_16k/checkpoint.pt`, trained with
hybrid_smooth, soft val = 1.2052 at step 16000):

| Eval setup | val_bpb |
|---|---|
| `backward_mode='hybrid_smooth'` (matches training) | 1.2053 (sanity ✓) |
| `backward_mode='ste'`, weights **untouched** | **2.9343 (catastrophic +1.73 bpb gap)** |
| `backward_mode='ste'`, weights bit-reverse-permuted on axis 1 | **1.2909 (proper hard-eval)** |

The +85.7 mb soft→hard gap with the permutation matches the +65-73 mb gaps seen elsewhere on
FastMHL (exp754/755), which is a sanity check that the fix is correct.

## The fix

Before flipping `backward_mode` away from `hybrid_smooth`, permute the LUT weight tensor
along axis 1 (the row axis) by the bit-reverse permutation:

```python
def _bit_reverse_perm(nap):
    K = 1 << nap
    out = [0] * K
    for k in range(K):
        r = 0
        for i in range(nap):
            if k & (1 << i):
                r |= 1 << (nap - 1 - i)
        out[k] = r
    return torch.tensor(out, dtype=torch.long)

with torch.no_grad():
    for m in luts:
        perm = _bit_reverse_perm(m.n_anchor_pairs).to(m.weights.device)
        m.weights.data = m.weights.data.index_select(1, perm).contiguous()
        m.backward_mode = 'ste'
```

Working eval script: `eval_exp720_hard_mode.py` (in repo root).

## Scope

- **Affects**: ANY TinyMultiHeadLut model trained with `backward_mode='hybrid_smooth'` (or
  presumably `'soft'`, `'soft_topk'`, `'prob'`, `'soft_winner'` — all use `soft_powers` MSB-first).
- **Does NOT affect**: FastMultiHeadLUT (different code path; its `forward_mode='hybrid_smooth'`
  → `'hard'` switch uses consistent indexing).
- **Does NOT affect**: TinyMultiHeadLut models that were trained natively in hard/STE mode from
  step 1 — they used LSB-first throughout.

## Implication for the experiment record

- exp720's 1.2052 val_bpb was **soft-only**. The deployable (hard-eval) number is **1.2909**
  after the bit-reverse fix.
- Comparing exp720's 1.2052 against any native-hard run (exp756, exp760, exp764) was
  apples-to-oranges. The fair hard comparison: exp720 = 1.2909, exp764 = 1.2116, exp760 = 1.2048.
- Same caveat applies to **any earlier TinyMHLut(hybrid_smooth) checkpoint** if it gets
  hard-evaluated retrospectively — the bit-reverse permutation must be applied first.

## How to apply

- For any retrospective hard-eval on a hybrid_smooth-trained TinyMHLut checkpoint, use
  `eval_exp720_hard_mode.py` as the template. Don't trust naive `mod.backward_mode = 'ste'`.
- Going forward, **prefer FastMultiHeadLUT over TinyMultiHeadLut for any deployment-targeted
  training** — its forward-mode flip is consistent and doesn't need the workaround.
- If you must use TinyMHLut(hybrid_smooth), document the bit-pack convention in the eval script
  and run the soft sanity check first (training-final should match within 1 mb).

Cross-refs: [[exp764-tph-halved-tiny-pareto]] (used the fix to position exp720 correctly on
the Pareto curve), [[exp735-v-lut-nap7-sota]] (FastMHL — does not need this fix).
