# PLAN — adapting the handcrafted spiking LUT to the new quantised policy

Preparation only. Nothing here has been built or run; no existing file was modified.

Target: make the exp012 spiking network reproduce
`deploy/quantised/walker2d_fastlut_lse_exp19_quantised.npz` (the L2 w=0.3 QAT policy)
instead of the older un-quantised `deploy_matched` one.

---

## 0. Summary of what I found

**The input side is nearly free.** The new encoder is the same *kind* of object the SNN
already consumes — a single shared monotone value→tick map — so the swap is a table
substitution plus one sign flip. Stage 1 should reach **exact** parity with the software
actor, not approximate.

**The readout side is not free, and this is the decision that governs the whole job.** The
existing Stage-3 structure resolves **7 levels** inside [-1, 1]; the new policy emits **22**.
The only lever that changes readout resolution is `TAU_M_OUT`, and it buys resolution at
exactly 1:1 in episode length: **22 levels needs `TAU_M_OUT` 10 → 31.26 (3.13×), which takes
the episode from ~296 to ~609 ticks and `dmax` from 78 to 236 against a hard engine cap of
255.**

Before paying that, there is a 5-minute measurement that could remove the need entirely —
see §4, step 0.

---

## 1. Input encoder — what changes

### How it works today

Two different encoders exist, both shared across all 17 coordinates, both monotone:

| where | map |
|---|---|
| build pipeline `tiny_lut_order_detect.encode()` | linear on percentiles: `lo, hi = pct(x, 0.5), pct(x, 99.5)`; `tick = (1-u)·127` |
| deployed actor `spiking_lut.py::_encode` | quantile table: `tick = 127 - searchsorted(qtable, x)` |

Both use the convention **larger value → earlier tick**, and the comparator reads
`bit = 1[tick[a] < tick[b]]` (`tiny_lut_full_pipeline.py:175`).

### What the new policy uses

`tick = searchsorted(in_quant_edges, x)` on the *normalised* observation, with 127 edges at
`σ·Φ⁻¹((k+0.5)/127)`. **Larger value → LARGER tick** — the opposite direction.

### The change

```python
# new latency encoder for the SNN — the ONLY input-side change
g   = np.searchsorted(Q["in_quant_edges"], x_norm, side="left")   # 0..127, larger x = larger g
tick = (T_IN - 1) - g                                             # invert: larger x = earlier
```

Three things worth stating because they make this cheaper than it looks:

1. **The SNN never needs `in_quant_dequant`.** It consumes only the *ordering* of ticks, and
   the dequantised value exists solely so the software LUT can compute a float difference.
   Only `in_quant_edges` crosses into the spiking model.
2. **Order is preserved exactly.** `tick[a] < tick[b] ⟺ g[a] > g[b] ⟺ x̂[a] > x̂[b]`, and
   equality maps to equality. So the SNN's address bits are *the same function* as the
   software actor's `d > 0`, not an approximation of it.
3. **Ties behave identically** — same bucket, same tick, tie → bit 0 via the existing tie
   detectors. ⚠️ Those detectors remain mandatory: measured on 100k observations the new
   Gaussian map produces ties on **0.950%** of address slots and **77.12%** of samples have
   at least one. (Task 533b00ea has the full argument.)

`t_in` stays 128, so `GATE_TICK` and the whole Stage-1/Stage-2 timing are untouched.

---

## 2. Readout — the real work

### Confirmed: a uniform tick grid *is* uniform in action space

The brief's intuition is right, and it is stronger than "near-linear" — it is **exactly
affine**. Arrival times are `a_t = emit + C_o − (τ_eff/τ)·w_sel[t,o]`, and the anti-leak
neuron crosses at `−τ_eff·log(Σ_t e^{−a_t/τ_eff})`, so the two logsumexps cancel:

```
crossing = const − (τ_eff / (32·τ)) · teacher_out
```

The tick↔action scale is therefore a **constant** `32·τ/τ_eff` action-units per tick —
identical near 0, near ±1 and in the tails. A uniform tick grid maps to a uniform action
grid with no companding. (Verified numerically earlier: round-trip error `std 0.08077`
against the textbook uniform prediction `step/√12 = 0.08085`.)

### Does the existing structure already support 22 levels? No — it supports 7

With this checkpoint's `τ = 0.09376808`:

| `TAU_M_OUT` | `τ_eff` | action-units/tick | levels in [-1,1] |
|---|---|---|---|
| **10.0 (current)** | 10.248 | 0.2928 | **7** |
| 20.0 | 20.249 | 0.1482 | 14 |
| 30.0 | 30.249 | 0.0992 | 21 |
| **31.257 (needed)** | 31.506 | 0.09524 | **22** |

**`TAU_M_OUT` is the only lever.** The step is `32·τ/τ_eff`; `τ` is the policy's learned
temperature (fixed by the checkpoint) and the weights do not appear — they set the delay
*span*, not the resolution. So resolution cannot be bought by re-scaling weights, by
re-fitting the decode affine, or by the L2 penalty.

### What 22 levels costs

`scale = τ_eff/τ` goes 109.29 → 336.00 ticks per unit of weight, and every delay scales with
it:

| | `dmax` | per-dim spans | settle | episode |
|---|---|---|---|---|
| now (`TAU_M_OUT` 10) | 78 | 78, 60, 70, 55, 64, 59 | ~75 | **~296** |
| 22 levels (`TAU_M_OUT` 31.26) | **236** | 236, 180, 210, 164, 191, 175 | ~231 | **~609** |

🔴 **`dmax` 236 against the engine's hard synapse-delay cap of 255** (`spnet.py:88`) — 19
ticks of headroom. A checkpoint with a slightly wider weight range would not fit at all. Any
build at this `TAU_M_OUT` must assert `dmax ≤ 255` and fail loudly, and it is worth checking
the w=0.1 seeds too (their spans are wider than w=0.3's).

Also note the episode roughly **doubles**, which lands on the demo's inference budget: the
current actor is ~12.9 ms/`act()` against a 33 ms frame, so ~2× puts it near ~26 ms with
`MAX_SESSIONS` already flagged as untested.

---

## 3. Scripts to build (all new, self-contained)

| file | reads | emits |
|---|---|---|
| `src/tiny_lut_quantised_pipeline.py` | the quantised `.npz` (`in_quant_edges`, `weights`, `anchor_a/b`, `tau_actor`, `obs_mean/var`, `out_quant_levels`) | the 3-stage SNN with the Gaussian latency encoder and a configurable `--tau-m-out`; per-stage accuracy JSON |
| `src/tiny_lut_quantised_export.py` | a validated build | `spiking_lut_quantised_actor.npz` in the deployed actor's schema |
| `landing/.../actors/spiking_lut_quantised.py` | that npz | the deployable actor (only after parity passes) |

A fork of `tiny_lut_full_pipeline.py`, per the standing fork-don't-flag rule — the original
stays byte-identical. Changes confined to: `encode()` → the edges-based inverted map;
`TAU_M_OUT` → a flag; a `dmax ≤ 255` assertion; the decode affine fitted against the
**22-level quantised** target rather than the continuous one.

---

## 4. Verification — the order matters

**Step 0 (do this FIRST — ~5 min, and it may cancel §2 entirely).** Evaluate the *already
trained* w=0.3 checkpoint with the output quantiser set to **8 levels** instead of 22, at
1024 envs × 2000 steps, matched physics:

```sh
python eval_qat_ckpt.py --ckpt <w0.3.pt> --envs 1024 --steps 2000 \
    --quant-ticks 128 --quant-sigma 1.0 --out-quant-levels 8
```

Rationale: on the *parent* policy, N=8 and N=22 were **statistically indistinguishable**
(6175.4 vs 6236.1, inside a ±16 noise band). If that still holds for a policy *trained* at
22 levels, then the spiking model can keep `TAU_M_OUT = 10`, stay at ~296 ticks, and the
whole 2× episode cost disappears. If it does not hold, we have quantified what the 2× buys.
**This is the cheapest decision-relevant measurement available and it gates everything else.**

**Step 1 — Stage 1 parity, exact.** Address bits from the SNN vs
`searchsorted`-derived bits from the npz, over the 4000-sample held-out tail. Expect
**100.0000%**, ties included; anything less means the encoder inversion or the tie path is
wrong. (The current model achieves 0 bad bits in 768,000 comparisons, so this is the right
bar.)

**Step 2 — Stage 2 one-hot.** 0 none / 0 multi per table. Same bar as today.

**Step 3 — Stage 3 level agreement.** Not R² — **exact-match rate against the software
actor's 22-level output**, per dim. R² would hide the thing we care about (landing on the
adjacent level is a miss, and R² will look excellent regardless).

**Step 4 — end-to-end.** Run the spiking actor in the gymnasium env for 30 episodes and
compare against the software quantised actor's ~6291. Parity here is the ship gate.

---

## 5. Risks, in the order I'd worry about them

1. **The 2× episode cost may buy nothing measurable** — see step 0. Resolving this first is
   worth more than any implementation work.
2. **`dmax` 236 vs the 255 cap** — 19 ticks of headroom, and it is checkpoint-dependent.
3. **`mem_margin` shrinks with a longer episode.** Recorded at 0.0335 today; the memory
   holds the rail decision with `TAU_MEM = 1200`, and while `GATE_TICK` itself does not move
   (Stage 1 timing is unchanged), any later gate would eat this margin directly.
4. **Settle is estimated, not measured.** The ~231 above is 75 scaled by `τ_eff`; the
   pipeline measures the real value (`measured_settle`), and the last time this was assumed
   rather than measured the true figure was ~half the assumed one. Measure it.
5. **Tie detectors stay.** 77% of samples have a tie under the new map; removing them is a
   separate piece of work with its own ablation.

---

## Backups taken before any of this

Untracked `.pt` checkpoints copied outside the git tree (`git clean` cannot reach them):

```
~/projects/ckpt_backups/exp19_walker2d_lut/{rerun_ckpt,deploy_matched}/actor_s{0,1,2}.pt
~/projects/ckpt_backups/exp23_qat/{qat_s0,qat_s1,qat_s2,l2w0p1_s0,l2w0p1_s1,l2w0p1_s2,l2w0p3_s0}.pt
```
