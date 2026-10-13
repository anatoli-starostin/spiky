# 06_conf_n1_oldbias — RESULT (NEGATIVE)

**The decompress-bias init is NOT the cause of the ~0.021 bpb gap.**

## Hypothesis tested
That the NEW tree's zero-init of `ProjectionMHL.decompress.bias` (commit 955762e9, 2026-10-08)
caused the fused conf_n1's ~0.021 bpb improvement over the abl47 reference-class conf_n1 (1.117706).

## Setup
Fork of `06_conf_n1` (seed 1). `train.py` is byte-identical to `06_conf_n1/train.py` plus ONE inserted
block that, after model construction and before the optimizer, re-initialises every block's
`ffn.proj.decompress.bias` to the OLD default `U(±1/sqrt(384)) = U(±0.051031)` drawn from a **dedicated**
`torch.Generator(seed=777)` — so the global RNG (hence the dropout mask stream) is byte-identical to
`06_conf_n1` seed 1. The bias **values** are the only difference from that run. `config.json` differs only
in `exp_name`. `src/spiky/lutorch_ex` was not modified. Launched from commit d84a1ab5.

Per-block decompress.bias at startup (logged): L2 = [0.61713, 0.58615, 0.54062, 0.55779, 0.58342, 0.57479],
all within ±0.051031, 0/384 zeros per block (re-init landed on all 6; not a no-op). Global RNG (cpu+cuda)
verified UNCHANGED by the re-init.

## Outcome — STOPPED EARLY at step 36000
Deliberately stopped (SIGTERM) at step 36000/48000 because the result was already unambiguous: past the
~16k crossover the oldbias run tracks the NEW zero-bias seeds, NOT abl47.

| step  | oldbias | mean(3 new seeds) | abl47 |
|------:|--------:|------------------:|------:|
| 2500  | 1.59708 | 1.59759 | 1.59626 |
| 5000  | 1.34914 | 1.35817 | 1.34460 |
| 7500  | 1.26319 | 1.26709 | 1.26027 |
| 10000 | 1.22845 | 1.23156 | 1.22725 |
| 12500 | 1.20770 | 1.20979 | 1.20754 |
| 15000 | 1.19151 | 1.19372 | 1.19425 |
| 17500 | 1.17915 | 1.18043 | 1.18315 |
| 20000 | 1.16832 | 1.16937 | 1.17379 |
| 25000 | 1.14985 | 1.15102 | 1.15818 |
| 30000 | 1.13377 | 1.13395 | 1.14510 |
| 35000 | 1.11862 | 1.11883 | 1.13298 |
| 36000 | 1.11569 | 1.11645 | 1.13111 |

At step 36000 the oldbias run (1.11569) sits essentially ON the new-seed mean (1.11645, Δ≈−0.0008) and
~0.0154 BELOW abl47 (1.13111) — and that gap to abl47 was widening, exactly like the new zero-bias seeds.

## Conclusion
Restoring the OLD random decompress bias did **not** reproduce abl47's deficit; the run behaves like the
new (zero-bias) runs. **The decompress-bias init is not the cause of the gap.**

## Caveat
This run was stopped deliberately at step 36000 and is therefore **not** a valid final number — it is a
trajectory comparison only.
