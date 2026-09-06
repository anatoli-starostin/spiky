# exp_g_0191 — CRASHED at step 15,100 of 16,000. No final score, no checkpoint.

**Read this before using any number from this folder.**

This run did **not** finish. It was killed at step ~15,100 by a driver-level GPU watchdog,
after the step-15,000 evaluation had been written. There is therefore:

* **no `summary.json`** — no `final_val_bpb`, no `best_val_bpb`;
* **no `checkpoint.pt`** — the run **cannot be re-scored and cannot be resumed**;
* **no `loss.png`**.

What survives is `config.json`, `metrics.csv`, `train.py` (and `train.log`, which is not
committed: `*.log` is gitignored repo-wide and no other run commits one — the relevant
excerpt is reproduced below instead).

## The configuration

Light (`lut_impl=light`) with `lut_confidence_form="margin"` — which is, to machine
precision, the LookupFFN score (arXiv:2403.07221); see `doc/lutorch/lut_mechanisms.pdf`.
Forked from `exp_g_0190` with the confidence form as the **only** functional change, so
`z_norm` is on, `tables_no_decay` is on, `inner_residual` is off, tph=128, NAP=8, seed 1,
16,000 steps, `eval_every=500`. 67,352,256 parameters, identical to exp_g_0190.

## The failure

```
torch.AcceleratorError: CUDA error: the launch timed out and was terminated
Search for `cudaErrorLaunchTimeout' ...

kernel log, 2026-09-06 20:34:03:
  NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7
  NVRM: GPU at PCI:0000:02:00: GPU-7d40ba2b-2f9a-107e-e8d9-a671b8da8089
  NVRM: Xid (PCI:0000:02:00): 8, pid=3280448, name=python, channel 0x00000013
```

Xid 8 with the RC watchdog firing on a 7-second notify timeout: the driver decided the
channel was hung and tore down the context. `pid=3280448` is this trainer. The RTX 5090 on
this host also drives the desktop, which is the usual setting for this class of kill. The
GPU reported healthy immediately afterwards (0% utilisation, 36 °C, no throttle reasons, no
retired pages or remapping failures), and exp_g_0189 and exp_g_0190 had run the identical
code path to completion earlier the same day. Treat it as environmental rather than a
property of this configuration — but note it is unexplained and may recur.

## What the surviving numbers do and do not mean

`metrics.csv` holds 30 evaluations, at every 500 steps up to 15,000. **These are the
corrected protocol** (`evaluate_bpb_fixed`, bs48 × 100 batches, leading 12 rows skipped,
2,451,456 tokens of the held-out `shard_06542.parquet`) — the in-run eval and the final
scorer are the same measurement, so they are directly comparable to any other run's curve
at the same step.

| | exp_g_0191 @15,000 | exp_g_0190 @15,000 | exp_g_0190 final @16,000 |
|---|---|---|---|
| val bpb | **1.178680** | 1.205148 | 1.203936 |

* gap at the matched step 15,000: **−0.026468**, which is 7.90× the 0.00335 seed spread;
* exp_g_0191 was **ahead at 30/30** shared evaluations, and the gap settled at ≈ −0.025 by
  step 5,000 and stayed flat for the following 10,000 steps;
* at step 15,000 it already beat exp_g_0190's **final** number by −0.025256, with 1,000
  steps still to run.

For scale: Fast with the gate off (exp_n_0129) is 1.170961 and vanilla dense is
1.165147 / 1.161798.

**`1.178680` is a step-15,000 value, not a final one.** It is a legitimate comparison
against exp_g_0190 at step 15,000 — same protocol, same step — and it is **not**
comparable to the 16,000-step endpoints the rest of the ladder is quoted at. Any headline
number for this configuration requires the run to be repeated.

The mechanism held as well: `ln2_norm_L0` reads **19.616581** at step 15,000, i.e. layer 0's
LayerNorm gain never left its √384 = 19.5959 initialisation. exp_g_0190 was at 16.35 by that
point and exp_n_0184 had collapsed to 0.00386.

## Status

Not relaunched. Rerunning this configuration unchanged is the obvious next step and would
cost about 50 minutes; it is the only way to obtain the endpoint and the checkpoint.
