# exp_n_0054 — soft-forward temperature-annealing LUT, H8/d48/tph64, 16k

> **STATUS: code-before-run (sanity-tested, full 16k run launching).** Tests whether running the soft full-K
> surrogate math ON THE FORWARD PASS (pure autograd, no straight-through estimator) while annealing both
> temperatures soft→sharp beats the hard-forward + surrogate-backward baseline (exp_n_0052 = 1.2285517).

## Idea
Today `FastMultiHeadLut` does a **hard** single-row gather on the forward and only uses the soft full-K
surrogate on the **backward** (STE-style). This experiment does the **opposite**: the forward output is the
**softmax-weighted sum over all K=64 rows** per table, fully differentiable by plain autograd — no STE, no
custom backward. Cheap because K=64 is small. Both temperatures anneal from a genuinely-soft init down to a
small floor over training, so by the end the soft weighted-sum concentrates on ≈one row (≈ discrete lookup).

## Mechanism (faithful to FastMHL's surrogate, run on the forward)
```
d         = x[:, anchor_a] - x[:, anchor_b]        # signed distances per anchor pair
soft_sign = d / (T_soft + |d|)                     # soft ±1;  -> sign(d) as T_soft -> 0
score_k   = <bit_matrix[:,k], soft_sign>           # agreement of input signs with row k's ±1 code
w         = softmax(score / T_sel, dim=K)          # -> one-hot(argmax row) as T_sel -> 0
out_table = Σ_k w_k · weights[table, k, :]         # bag-summed over tables_per_head
```
This matches FastMHL's `_soft_lut_bwd_body` surrogate exactly (for the argmax row the pinned surrogate
`p = d/(T_soft+|d|)` equals `soft_sign`); here it is unpinned over all K rows and used as the forward. As
`(T_soft, T_sel) → 0`, the soft output → the hard FastMHL lookup.

## Annealing
Both temps decay **exponentially** `soft_anneal_temp_start=0.5 → soft_anneal_temp_floor=0.02` over all 16000
steps (`T(step)=start·(floor/start)^(step/n_steps)`), same schedule for T_soft and T_sel, driven by the global
step (NOT learnable). At step 0 the model is genuinely soft (full-K blend); by the final step temp≈0.02 so the
blend is ≈ one-hot ≈ hard lookup. Temps are exposed to the training loop and set on every SoftAnnealLut each step.

## Implementation
`soft_anneal_lut.py` (experiment-local, torch.compile'd) defines **`SoftAnnealLut`** — a drop-in for a per-head
FastMHL that copies its tables/anchors/bit_matrix (so init is **bit-identical to exp_n_0052's forward slot**) and
replaces the forward with the soft weighted-sum above. `train.py` builds the standard CompressionMHL FFN model
(independent per-head loop path) then **swaps each `block.ffn.luts[h]` FastMHL → SoftAnnealLut**. The shared
modules `fast_multi_head_lut.py` / `compression_mhl.py` are **untouched**. LUT tables go to the nodecay optimizer
group as usual; compress/decompress Linears unchanged (decompress zero-init).

## Config
H8/d48/tph64/nap6, device_bs 48, grad_accum 1, 16000 steps, warmup 1600, seed 1, clean val 245,760 — the same
rung as **exp_n_0052 (1.2285517, hard-forward batched control)**, so `0054 − 0052` isolates soft-anneal-forward
vs hard-forward+STE. vs dense: 1.196646.
