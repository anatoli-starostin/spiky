# exp034 — linear-only control (no FastMHL), 4k steps

The **linear-only ablation of exp033**. Each block's dense 384→1536→384 GELU FFN is
replaced by **only a single plain `nn.Linear(384 → 384)` with bias** — no GELU, no hidden
expansion, **no FastMHL**:

```
h = ln2(x)
x = x + linear(h)
```

The Linear weight is zero-init'd (residual-identity start, as vanilla did for the FFN
output projection). Attention, LayerNorms, residual structure, data, training loop, eval,
and all hyperparameters are byte-identical to exp032/exp033 (d384 / 6L / 6H / seq 512,
device_bs 48, total_bs 24576, n_steps 4096, lr 3e-4, wd 0.1, warmup 0.1, eval_every 200,
seed 1, fp32, vocab 32768, same data).

## Purpose — three-way comparison
This control isolates the contribution of the FastMHL path in exp033:

| exp | FFN slot | params | val_bpb |
|-----|----------|--------|---------|
| exp032 | dense 384→1536→384 GELU FFN | 35,792,640 | **1.39371** |
| **exp034** | **Linear 384→384 only** | **29,601,792** | *(this run)* |
| exp033 | Linear ∥ hard-FastMHL (param-matched) | 35,794,944 | 1.41430 |

exp034 − exp033 tells us what the FastMHL path bought (or cost) on top of the bare Linear;
exp034 vs exp032 shows how far a single linear FFN-slot is from the dense GELU MLP.

## Param count — intentionally NOT param-matched
| | params |
|---|---|
| exp032 (vanilla, incl. 6× FFN 7,077,888) | 35,792,640 |
| − remove dense FFN | −7,077,888 |
| + add Linear 384→384 + bias (6× 147,840) | +887,040 |
| **exp034 total** | **29,601,792** |
| Δ vs exp032 | **−6,190,848 (−17.3%)** |

This is smaller than exp032/exp033 by design (FFN removed, no LUT) — expected and fine for
a control. (Confirmed by a build smoke.)

## Optimizer
Identical to exp032: AdamW + warmup/cosine; Linear **weight → wd 0.1** group, **bias →
no-wd** group; 2-D params decay, 1-D no-decay.

## Status
Launched under the owner's GO. Est. runtime ≈ exp032's ~8 min (a hair faster — smaller
model). Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt`
(checkpoint gitignored).
