# exp_n_0086_vlut_5x_tph88 — routed V (fix B), stopped at step 1600 for the overnight sweep

Routed V (CompressionMHL v_lut replacing dense Linear V), v_lut = H4/in24/out24/nap6/tph88,
with init fix (B): table-cell init noise 0.363 + decompress bias zeroed → step-0 V RMS ≈0.387,
input-dependent (matches dense Linear V). See config.json `_arch_note` / `_init_fix_note`.

## Partial result (stopped @step 1600 to start the overnight v_lut sweep)
Even with fix (B) landing the step-0 V scale right (early Δ vs exp_n_0084 = +0.0046 @step 200),
the Δ WIDENS over training:

| step | v_lut bpb | exp_n_0084 bpb | Δ |
|---|---|---|---|
| 200 | 2.6235 | 2.6189 | +0.0046 |
| 400 | 2.1415 | 2.1394 | +0.0021 |
| 800 | 1.8681 | 1.8516 | +0.0165 |
| 1200 | 1.7386 | 1.7132 | +0.0254 |
| 1600 | 1.6418 | 1.6133 | +0.0286 |

So the init mismatch was real (fixing it removed the ~+0.03 early gap the pre-fix run had at step 400),
but the routed V still loses ~0.03 bpb to the dense Linear V by step 1600 — a training/optimization or
capacity issue, not init. This motivated the overnight v_lut sweep (n_heads / tph / inner-dim / init /
optimizer). Note: this FFN-slot trainer runs **AdamW on everything** (incl. all LUT tables, in the wd=0
group) — there is no Lion split here.
