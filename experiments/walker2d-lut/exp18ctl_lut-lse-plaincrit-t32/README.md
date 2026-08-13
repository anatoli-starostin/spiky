# exp18ctl_lut-lse-plaincrit-t32 — the control for exp18

**This is not a standalone experiment — it is exp18's control**, and it only means anything
next to it. See `../exp18_lut-lse-lutcrit-t32/README.md` for the full write-up.

exp17's sum-scaled log-sum-exp actor (tph=32, one trainable τ) paired with an anchor-pair
LUT critic using the **plain sum over tables** — no exponential on the value head.

Everything else is identical to exp18: same actor, same critic topology, same critic tph=32,
same additive init, same bench7 recipe (8192 envs, 768 updates, 3 seeds). The *only*
difference is the critic's readout, which is the point: exp13–15 already showed a LUT critic
costs a lot against an MLP critic, so this arm isolates the **exponential readout** from
"LUT critic vs MLP critic".

14,343 params = exp13's 14,342 + one τ (the actor's).

## Result

| seed | best | final | final/best | actor τ |
|---:|---:|---:|---:|---:|
| 0 | 3690.9 | **3396.2** | 0.920 | 0.0713 |
| 1 | 1131.5 | **1053.2** | 0.931 | 0.0707 |
| 2 | 1346.9 | **1317.3** | 0.978 | 0.0697 |

**1922.2 ± 1047.8, collapse 0/3**, 14.7 min/seed.

Against the treatment (exp18, 2752.5 ± 1127.6): **Δ +830.3 in the exponential critic's
favour, Welch se 1088.4, |t| 0.76 — not significant.** With a pooled sd of 1333 at n=3 the
smallest detectable gap is ~2177 points, so this pair of arms cannot resolve the effect;
~41 seeds per arm would be needed.

Its actor τ rises 0.05 → 0.0706, essentially identical to the treatment's 0.0701 — i.e. the
actor's τ behaviour tracks *the critic being a LUT*, not *the critic being exponential*.

## Verification of the isolation

At init the two arms were confirmed to differ in exactly one place:

- identical actor — action mean +0.000360, std 3.276537e-03 in both;
- this arm's critic is numerically identical to `fastlut2`'s plain-sum LUT critic
  (value mean +0.000212, std 3.170988e-03), while the treatment's differs only by the
  Jensen gap of its exponential readout (+0.000315, same std).

## Files

`config.json` / `metrics.csv` (τ per row; `tau_critic` is empty by construction) /
`summary.json` are written by `../exp18_lut-lse-lutcrit-t32/collect.py`, together with the
treatment's. The run script and plot live in the treatment folder.

Arch: `fastlut_lse_sum2_plaincrit`. Not committed or pushed.
