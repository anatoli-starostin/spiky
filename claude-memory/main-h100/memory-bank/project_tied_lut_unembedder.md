---
name: project_tied_lut_unembedder
description: "Tied LUT unembedder (exp514-522): a matmul-free LUT head + identity/inverse reg. Invertibility lever is COVERAGE K=tph*n_sparse/V (not LR): K=4 -> 100% self-decode. But as an LM head it plateaus ~2.12 bpb (sparse logits) vs Linear head 1.498 -- not competitive. Standalone inverse tests in workbooks/."
metadata:
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Tied LUT unembedder exploration (2026-05-24), user's idea.** Goal: replace the 12.6M `Linear(D,V)` unembedder with a small matmul-free LUT head, tied to the embedder via an identity/consistency loss `CE(unemb(emb_table), arange(V))` (each token's embedding must decode to itself). Dual stream collapsed to E so the predicted embedding is in R^E.

## Two head designs tried
1. **Row=token head** (custom, `workbooks/tied_lut_inverse_test.py`): NAP=15 -> 2^15=V rows, row r = token r, tph tables vote. Needs a **winner-take-all ASSIGNMENT STE** (route each token to its closest-Hamming table, push ALL its bits toward the target's binary code) — naive Hamming-1 soft-STE gets stuck at 2% top1, assignment STE -> **99.94%**. Sparse logits (~tph nonzero/token). In the LM it forced the embedder into fixed-hash-cell geometry and bpb got *worse* as the tie improved -> abandoned.
2. **Standard sparse-scatter `TinyMHLut` head** (`workbooks/tied_sparse_lut_inverse_test.py`, exp517+): E->V via `sparse_scatter_n_outputs=V`, `backward_mode='ste'`. The proven mechanism; the LEARNABLE LUT adapts to the embedder instead of forcing it.

## KEY: invertibility lever is COVERAGE K = tph*n_sparse/V, not LR
Isolated inverse test (trainable Embedding(V,64) + sparse head, consistency CE only):
| | lr=1e-3 | lr=1e-2 |
|--|--|--|
| K=1 | 11% | 78% |
| K=4 | **100%** | 100% |
K=4 fully inverts at BOTH LRs; K=1 never does. So earlier "LR was the blocker" was WRONG — coverage is. (exp514 froze because of K=1 + warmup; the LR fix exp515 only partly helped.) K=4 config = nap=6/tph=8192/n_sparse=16 (8.4M, same as K=1 reshaped).

## As an LM head: works but NOT competitive (~2.12 vs Linear 1.498)
With K=4 sparse head + identity reg in the full LM:
- exp518 (dual-stream E, aux weight 1.0): plateau **~2.12 bpb**. aux_top1 -> 100% by ~step 2000.
- exp521 (no aux loss): ~2.12 (same floor; aux SPEEDS mid-run descent but the floor is the same; aux does NOT self-emerge without the reg, stays 0%).
- exp522 (single E-stream, no residual_lut, aux 0.25): ~2.25 -> the dual-stream/residual_lut is worth ~0.12 bpb.
All land **~0.6 bpb above exp428's dense `Linear(D,V)` head (1.498)**. The sparse ~K-vote logits are a fundamentally weaker output distribution than a dense matmul. **Conclusion: elegant + tiny (0.5-8.4M vs 12.6M) + matmul-free, but as a QUALITY head it underperforms badly. Not adopted.**

Relates to [[project_matmul_free_lut_shape_rules]] (the qk/v/residual line kept the standard Linear head). The `RowTokenLUTUnembedder` + assignment-STE math and the sparse-scatter test live in `workbooks/tied_*_inverse_test.py`.
