# CANCELLED — never started; too large for gpustar, deferred to nebius-h100

Stopped before it ever ran. This run was queued to launch automatically after
`exp_n_0162` finished; that chain was killed together with 0162 on 2026-09-05 when Anatoly
called the nap9 CompressionMHL models too big for the RTX 5090. The large configs move to
nebius-h100; gpustar switched to the smallest cell
(`exp_n_0164_compmhl_H4_tph256_nap8_dc32`).

**Step reached: 0 of 16,000.** No `train.log`, no `metrics.csv`, no checkpoint — the process
was never launched, so there is nothing partial to interpret. `config.json` and `train.py`
are complete and ready: launching this on an H100 needs no edit beyond raising
`device_batch_size` back to 12 / `grad_accum` 4 (see below).

## The config, verified before it was queued

Built once with `SMOKE=1` (which constructs the model and exits) to check the param count
against the brief:

* **180,154,956 params**, *not* exp_n_0118's 180,597,900 as the brief expected.
* The LUT table tensor **is** bit-identical to 0118's: `6·2·512·512·48 = 150,994,944`.
* The 442,944 gap is entirely compress/decompress projections — halving the heads halves
  them (445,248 vs 888,192 across 6 layers, `compress 384→H·d_c` and `decompress H·d_c→384`).
* No configuration of H=2 / tph=512 / nap=9 / d_c=48 reaches 180,597,900. **The pair is
  iso-TABLE, not iso-total**, 0.25% apart. `n_heads=2` and `tables_per_head=512` are exactly
  as specified in the built module, so this is arithmetic in the brief, not a misconfiguration.

Projection FLOPs `H*384*d_c` = 36,864 vs 0118's 73,728 and vanilla's `384*384*4` = 589,824 —
**0.0625x, half of 0118's projection cost at the same table budget**. That was the question
this run was meant to answer: are 4 narrow routing streams better than 2 wider ones when the
table budget is held fixed? It is still open.

## Memory setting

`device_batch_size: 6` / `grad_accum: 8` here is a 5090 workaround, not a property of the
experiment. The nap9 soft-backward buffer is `[tokens, H*tph=1024, 2^nap=512]` fp32 —
identical in shape to 0118 and 0162 — which is 12.9 GiB at bs12 and OOMs on 31.35 GiB.
**On nebius's H100, restore `device_batch_size: 12` / `grad_accum: 4`** to match 0118 exactly.
The effective batch is 24,576 either way, and eval is decoupled from `device_batch_size`
(fixed bs48 × 100, skip 12), so neither choice moves the metric.
