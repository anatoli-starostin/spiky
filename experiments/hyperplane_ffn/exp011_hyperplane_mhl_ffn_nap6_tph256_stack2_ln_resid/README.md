# exp011 — stacked (×2) HyperplaneMHL FFN with intermediate pre-LayerNorm + internal residual

Continues the **exp007** line (`research/hyperplane_ffn_next`). exp007 replaced each block's
dense MLP FFN with a **single** `HyperplaneMultiHeadLUT` (n_heads=1, n_outputs=384=d_model,
NAP6, tph256, hard forward). exp011 changes **only the FFN internals**: it stacks **two**
HyperplaneMHLs with an intermediate pre-LayerNorm and an **internal residual around the second
sublayer**.

## The one architectural change vs exp007

The FFN forward becomes:

```
h = lut1(x)                     # first HyperplaneMHL, 384 -> 384
h = h + lut2(LayerNorm(h))      # residual around the second, LayerNorm as its pre-norm
```

- Both `lut1` and `lut2` are `HyperplaneMultiHeadLUT` with **identical geometry to exp007**:
  `n_heads=1`, `n_outputs=384=d_model`, `n_anchor_pairs=6`, `tables_per_head=256`, `hard`
  forward, LUT weights bf16 / hyperplanes fp32. Each maps `d_model -> d_model` with no
  reshape/projection.
- The intermediate norm is a plain `nn.LayerNorm(384)` with elementwise affine (matching the
  block's `ln1`/`ln2`), used as the **pre-norm of the second sublayer**.
- The **internal residual** wraps only the second HyperplaneMHL: `h + lut2(LN(h))`. The first
  HyperplaneMHL has no internal residual (it produces the intermediate `h`).
- The **outer block residual is unchanged** from exp007: the block still computes
  `x = x + attn(ln1(x))` then `x = x + mlp(ln2(x))`, where `mlp` is this two-sublayer FFN.

Everything else — data, tokenizer, depth 6 / d_model 384 / 6 heads / seq 512, batch, steps,
schedule, seed 1, dtypes — is exactly exp007.

## Per-layer init seeds

Each LUT gets a distinct, reproducible seed so no two LUTs in the model share an init:

```
lut1 seed = random_seed + 2*layer_idx + 1
lut2 seed = random_seed + 2*layer_idx + 2
```

(exp007 used `random_seed + layer_idx + 1` for its single LUT.)

## Optimizer routing

The hybrid optimizer (routed by parameter **identity**, not ndim) now routes **both** LUTs:

- `lut1.weights`, `lut2.weights` (LUT table weights) → **Lion** (lr `lut_lr`, wd 0).
- `lut{1,2}.hyperplane_weight`, `.hyperplane_bias`, and learnable temps → **Adam, no wd**.
- Everything else → **AdamW** (exp001 rule: ndim≥2 → wd 0.1, ndim<2 → no wd). The intermediate
  LayerNorm's affine params are 1-D, so they land in AdamW-**nodecay** — consistent with the
  block's other LayerNorms.

Missing `lut2` in the routing would silently drop its table weights into AdamW-with-wd, which
would be wrong; the loop explicitly iterates `(blk.mlp.lut1, blk.mlp.lut2)`.

## Status

Smoke-tested only (build + a few fwd/bwd steps + one eval batch, grad-flow confirmed). **No
full training run has been launched** — that awaits a separate GO, per the agree→commit→go
protocol.
