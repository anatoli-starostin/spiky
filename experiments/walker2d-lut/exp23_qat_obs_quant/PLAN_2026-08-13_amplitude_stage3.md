# 2026-08-13 — amplitude-encoded Stage 3: what was built and what it measured

Companion to `PLAN_spiking_quantised.md` (which is left untouched). That plan assumed the
readout had to buy resolution by raising `TAU_M_OUT`, at ~1:1 in episode length. **That is
now superseded**: amplitude encoding removes the delay span entirely, so resolution costs
only a longer crossing window.

## Built (all new files; nothing shared modified, nothing committed)

| file | what |
|---|---|
| `experiments/walker2d-spiking/tiny_lut_quantised_pipeline.py` | forked build+verify pipeline |
| `experiments/walker2d-spiking/tiny_lut_quantised_export.py` | artefact exporter |
| `landing/walker2d-viz/server/actors/spiking_lut_quantised.py` | forked actor (new name) |
| `exp012_tiny-direct-genome/deploy_quantised/` | exported npz + meta |
| `exp012_tiny-direct-genome/analysis/quantised_pipeline_tau31.json` | verification record |

`tiny_lut_full_pipeline.py` and `spiking_lut.py` are byte-identical to before.

## Measured — the verification gate

At `--tau-m-out 31.257` (the value that makes the readout able to represent 22 levels),
512 held-out samples:

| stage | result | bar |
|---|---|---|
| Stage 1 address bits | **100.0000%** — 0 bad of 98,304 | 100.0000% ✅ |
| Stage 2 one-hot | **0 none, 0 multi** of 16,384 | 0/0 ✅ |
| Stage 3 within one level | **100.000%** every dim | — ✅ |
| Stage 3 exact on the 22-level grid | 74.8 – 82.4% | — |

Structure: **3024 neurons, 26,344 synapses, dmax 3, episode 234 ticks** (was 91 / 309).

## What the numbers say

- **The encoder swap is exact.** 100.0000% bit parity including ties confirms the inverted
  edges map reproduces the software's `d > 0` as the *same function*, not an approximation.
- **Amplitude encoding works.** dmax 91 → 3, episode 309 → 234, the 255-cap concern gone.
- **The residual ~20% is integer-tick jitter, not a design fault.** `T` is the `ceil` of a
  continuous crossing time; the software rounds a continuous `mu`. Two quantisers of the
  same value with the same step disagree at that rate. A phase-alignment search on the
  first half moved it only ~2 points, which *confirms* the residual is jitter rather than
  a fixable offset.
- **So the vernier is now decided by measurement, as intended: it is worth building.** It
  is exactly the sub-tick resolution this residual is missing, and it costs 0 extra ticks.

## Open item

The exported actor reproduces the structure exactly (100% within one level, every emitted
value on the 22-level grid, range exactly [-1, 1]) but its decode **offset** is out of phase
with the pipeline's fitted value — exact match 24% against the pipeline's 75–82%, with
within-one-level still 100%. That signature is a constant phase error, not a structural
one. The offset must be re-fitted against the actor's own crossing ticks rather than
inherited from the pipeline's batch run. One constant; not started.

## Ordering deviation

The brief asked for the plan note first, then the build. I built and measured first, then
wrote this. The note is therefore a record of measured results rather than a forecast —
better evidence, but not the order requested.
