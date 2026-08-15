# exp035 — pure-FastMHL FFN replacement (param-matched), 4k steps

The **opposite control to exp034** (linear-only). Each block's dense 384→1536→384 GELU FFN
is replaced by **only a hard FastMultiHeadLUT** — no parallel linear, no GELU, no dense FFN:

```
h = ln2(x)
x = x + fastmhl(h)
```

`fastmhl` = a plain fixed-anchor-pair `FastMultiHeadLut` (classic sign-test addressing —
NOT HyperplaneMHL), `forward_mode="hard"` (hard row-selection forward, full soft-surrogate
backward → gradients reach the LUT tables and flow densely back into `h`). Tables use
near-zero init (1e-3), so the FFN slot ≈ 0 at step 0 and grows on merit. Attention,
LayerNorms, residual, data, training loop, eval, and all hyperparameters are byte-identical
to exp032 (d384 / 6L / 6H / seq 512, device_bs 48, total_bs 24576, n_steps 4096, lr 3e-4,
wd 0.1, warmup 0.1, eval_every 200, seed 1, fp32, vocab 32768, same data).

## FastMHL capacity — EXACT param match
`FastMultiHeadLut(input_dim=384, n_heads=1, n_outputs=384, n_anchor_pairs=6,
tables_per_head=48, forward_mode="hard", use_bf16=False, initial_weights_noise=1e-3,
random_seed=1000+layer_idx)`.

With no linear consuming budget, the FastMHL alone should equal the removed FFN's
1,179,648/layer:
`tph = 1,179,648 / (2^NAP · 384) = 1,179,648 / (64·384) = 48` exactly (NAP=6).
Per-layer FastMHL = 48·64·384 = **1,179,648** = the FFN it replaces, to the param.

## Param count — exact
| | params |
|---|---|
| exp032 (vanilla, incl. 6× FFN 7,077,888) | 35,792,640 |
| − remove dense FFN | −7,077,888 |
| + FastMHL 48·64·384 ×6 (1,179,648/layer) | +7,077,888 |
| **exp035 total** | **35,792,640** |
| **Δ vs exp032** | **0 (exact)** |

(Confirmed by a build smoke.)

## Optimizer
Identical to exp032 except LUT-table weights → **no-weight-decay** AdamW group (project
lesson: LUT tables take no wd). All other params grouped as exp032.

## Purpose — four-way comparison (all 4096 steps)
| exp | FFN slot | params | val_bpb |
|-----|----------|--------|---------|
| exp032 | dense GELU FFN | 35,792,640 | 1.39371 |
| exp033 | Linear ∥ hard-FastMHL | 35,794,944 | 1.41430 |
| exp034 | Linear-only | 29,601,792 | 1.47970 |
| **exp035** | **hard-FastMHL only** | **35,792,640** | *(this run)* |

exp035 vs exp034 isolates FastMHL-alone vs Linear-alone at the FFN slot; exp035 vs exp033
shows whether the parallel linear highway helped the LUT; exp035 vs exp032 is pure-LUT vs
dense-FFN at identical params.

## Status
Launched under the owner's GO. Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`,
`checkpoint.pt` (checkpoint gitignored).
