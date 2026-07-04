# exp374_grow_loop — handoff notes

## What this experiment is

Signal-guided **iterative grow-loop** training for the BitPermLUT transformer
family (FullBitPermRankAttn_ctx128 in `transformer_exps/exp362_front2_graded_v_short`
etc.). The idea:

1. Start with a **tiny uniform model** (per-layer `tph` small and equal).
2. Train `steps_per_round` steps.
3. Measure **per-LUT-module `grad_out_norm`** averaged over the last
   `signal_window` steps (raw loss pressure on each module).
4. For each role in `{Q, K, V, Out}` independently, allocate a budget
   increment across the 6 layers proportional to signal. The total budget
   per role ramps linearly from the initial tiny value to `target_totals[role]`
   over `n_rounds` grow rounds.
5. Call `grow_lut(old, new_tph, seed)` on each LUT that needs growth. The
   new module has `new_tph` tables per head; the **first `old_tph` tables
   per head are warm-started** from the trained small module (latents +
   bit_weights + input & output anchor pairs). Remaining slots get fresh
   CFC-sampled anchors and random latents.
6. Rebuild `BitPermutationLUTOptimizer`. Adam-side state (`m`, `v`) resets
   to zero — acceptable trade-off for MVP.
7. Repeat for `n_rounds`.

Target totals are matched to exp362's best short-scale config (Q/K=1152,
V=1536, Out=6144) so the final grown model is directly comparable.

## Why we started it

The gradient-norm diagnostic in **exp371** (graded topology) and **exp372**
(uniform topology, the cleaner baseline) showed that on the uniform baseline,
L0 sees **50–75× more loss pressure than L5** for Q/K/V, and ~4× for Out:

| Role | L0/L5 `grad_out/latent` ratio (uniform) |
|------|-----------------------------------------|
| Q    | 56×                                     |
| K    | 76×                                     |
| V    | 58×                                     |
| Out  | 4.3×                                    |

This suggested a principled auto-sizing knob: allocate tph proportional to
this signal instead of hand-designing per-layer `tph` patterns. exp374 is
the validation: does signal-guided growth end up at a similar (or better)
final topology and val loss than hand-graded exp362 (val 1.3390 at 25k×bs=8)?

See `project_transformer_exp_summary.md` / `project_permutational_architecture.md`
in claude memory for broader context on the BitPermLUT / graded-tph sweep
(exp347–exp373).

## Current state (when stopped on this VM)

Stopped at **step ~8500 of ~25000** (inside round 1). GPU contention on this
VM required freeing resources for exp370 (full-scale run). The exp was
producing healthy output, no bugs observed.

**Round 0 (uniform 32/32/32/128)** val trajectory:
- step 1000: 2.5122
- step 3000: 1.8969
- step 5000: **1.8278**

**Round 1 allocation** (target_now: Q=432, K=432, V=528, Out=2112):
- Q: [253, 51, 32, 32, 32, 32] signals=[1.85e-4, 2.63e-5, 9.70e-6, 8.60e-7, 2.68e-7, 2.86e-7]
- K: [255, 49, 32, 32, 32, 32] signals≈same as Q
- V: [368, 32, 32, 32, 32, 32] (all extra went to L0 — hit the floor on L1..L5)
- Out: [1355, 245, 128, 128, 128, 128]

**Key observation from round 1**: the gradient signal is **extremely skewed
toward L0** (2-3 orders of magnitude higher than L5). Current allocation
logic honored that proportionally, giving L0 almost the entire budget
increment. This is likely too aggressive — final topology at this rate will
basically be "L0 enormous, L1–L5 stay at initial-tiny for 4 more rounds,
then last round forced to bloat them".

Round 1 training progressed reasonably: val at step 6000 was 1.9741 (a bit
worse than end-of-round-0's 1.8278 — grow adds noise from new random
tables). By step 8000 it was back to 1.8331, recovering.

## Files in this directory

- `config.json` — hyperparameters + `initial_tph`, `target_totals`,
  `n_rounds`, `steps_per_round`, `signal_window`. **Don't change the
  target_totals** if you want direct comparison with exp362.
- `grow_lut.py` — the core primitive. `grow_lut(old, new_tph, seed) -> new_lut`.
  Verified to exactly preserve `latent`, `bit_weights`, `anchor_pairs_a/b`,
  `idx_a/idx_b` in the first `n_heads*old_tph` slots; fresh CFC for new
  slots; rescales `scale = 0.5 / sqrt(n_votes_per_pair)` automatically.
  Includes a self-test at the bottom of the file.
- `train.py` — the driver. Implements `allocate_and_grow(round_idx)` which
  reads `signal_accum` (the per-module grad_out_norm window), computes
  per-role allocations, calls `grow_lut` on each module, and rebuilds the
  optimizer.
- `stdout.log` — training log from this VM's partial run.
- `metrics.csv` — (step, train_loss, val_loss, phase) — partial.
- `topology.csv` — full topology snapshots at each grow event.
- `signals.csv` — allocation inputs & outputs per round.

## Known issues / design choices to revisit

1. **Allocation too aggressive at L0.** The proportional allocation
   `delta_layer = budget * signal_layer / sum(signal)` amplifies the L0
   outlier. Options:
   - Use `log(signal)` or `sqrt(signal)` instead of raw proportion.
   - Cap per-layer allocation at, say, 50% of round budget.
   - Normalize signals differently (e.g., divide by current `tph` so the
     signal is "per-table" pressure rather than total).
   - Use `grad_out_norm / sqrt(tph)` as the signal — rough correction for
     the fact that bigger modules naturally have higher summed grad_out.
   Likely the "per-table" normalization is what we want: if a module has
   10× more tables, its total grad_out_norm is naturally ~√10× higher
   (assuming independent noise), so dividing by `√tph` normalizes.

2. **Adam state reset on grow.** `bit_opt.close()` + re-create causes Adam
   `m`, `v` to reset to zero for all modules (not just grown ones). Could
   preserve state by copying the un-grown modules' m/v forward. Acceptable
   for MVP since m/v re-warms within ~100 steps.

3. **Warmup cosine LR** is computed against `TOTAL_STEPS = (N_ROUNDS+1) * STEPS_PER_ROUND`
   so the LR schedule treats rounds as one continuous training run. After
   a grow, the LR doesn't reset — this matches what we want (continuous
   training with growing architecture).

4. **No CFC coordination across old+new.** New tables get their own
   independent CFC sampling. They may duplicate pair coverage that's
   already present in old tables (benign — just uniform redundancy).
   Option B in the design discussion was to do a "CFC fill-in" that
   targets currently thin pairs; skipped for simplicity. Probably not
   worth implementing unless signal-guided growth otherwise looks
   promising.

5. **Random init noise for new latents** is hard-coded to 0.001 in
   `grow_lut`. Matches `bit_lut_latent_init_std=0.001` from config.

## IMPORTANT: Scale only V and out_proj; keep Q/K flat at tph=192

**Empirical finding (end-of-session)**: the gradient-norm signal is a
**reliable guide for V and Out, but misleading for Q/K**.

Evidence:
- **exp372** (uniform diagnostic) showed L0 Q/K grad pressure **56-76×
  higher than L5** — strong signal suggesting L0 Q/K under-capacity.
- **exp373** tested the prediction directly: `qk_tph=[256,192,192,192,192,192]`
  (bump L0 only). Result: **val 1.3478 vs exp362's 1.3390 — +0.009 worse**.
- Combined with exp363–exp368 (all Q/K grading/reduction variants hurt),
  the picture is consistent: **Q/K has a narrow sweet spot at flat tph=192**.

Interpretation: Q/K is a **discriminator** task (binary sign decisions that
route attention), not a capacity task like V/Out (graded representational
encoding). High L0 Q/K gradient norm reflects **active learning at the
right capacity**, not "wants more tables". Adding more Q/K votes creates
redundant decisions that don't improve the underlying attention routing.

**Rule for grow-loop runs**: allocate budget across V and out_proj only.
Q/K should be held fixed at its sweet-spot tph (192 for the ctx128
architecture) throughout all rounds. The grow allocation loop should skip
roles `Q` and `K` entirely.

Code change needed: in `allocate_and_grow()` in `train.py`, either:
- Set `target_totals['Q'] == target_totals['K'] == initial_total` (so
  allocation is a no-op), **or**
- Short-circuit the loop for `r in ['V', 'Out']` only.

Config suggestion:
```json
"initial_tph": {"Q": 192, "K": 192, "V": 32, "Out": 128},
"target_totals": {"Q": 1152, "K": 1152, "V": 1536, "Out": 6144}
```
(Q/K start at their final value and stay there; only V and Out grow.)

## Suggested next steps (updated)

1. **Run to completion with Q/K pinned at 192**. Compare final val to
   exp362 (1.3390). Since exp362 already hand-graded V and Out correctly,
   the grow-loop's value is showing that a generic signal-guided
   procedure arrives at a similar (or better) allocation automatically.

2. **Try "per-table" normalized signal** (`grad_out_norm / sqrt(tph)`)
   for V and Out. This corrects for the fact that bigger modules
   naturally have higher summed grad_out (√tph scaling under
   independence).

3. **Experiment with round schedule**: instead of N=4 equal rounds, try
   geometric (e.g., each round doubles one role's total). Or fewer,
   bigger rounds (2 × 12500 steps?).

4. **Measure evolution of signals across rounds**: does L0 stay the
   clear outlier as it grows, or does the signal equalize? If it
   equalizes after growing L0, subsequent rounds should redistribute
   naturally.

5. **Compare to hand-tuned exp362 at matched budget**. exp362 topology
   (out graded [2048,2048,1024,512,256,256], V graded [512,512,256,128,64,64],
   Q/K flat 192) at val 1.3390 is the bar. If signal-guided arrives at
   val ≤ 1.3390 **or** at similar val with meaningfully different
   topology, that's a paper-worthy finding.

## Environment

- Repo: `/home/starost/spiky`
- Python: `.venv/bin/python`
- GPU: single H100 80GB on the original VM. grow_loop needs ~20GB after
  final round so any H100 / A100 / L40S works.
- Run: `.venv/bin/python -u transformer_exps/exp374_grow_loop/train.py > transformer_exps/exp374_grow_loop/stdout.log 2>&1 &`
- ETA: ~1.5–2h at 25k total steps × bs=8 on a dedicated GPU (longer if
  contended).

## Related experiments / reference

- **exp362_front2_graded_v_short** — hand-tuned best short-scale config
  (val 1.3390). `out_tph_per_layer=[2048,2048,1024,512,256,256]`,
  `v_tph_per_layer=[512,512,256,128,64,64]`, flat Q/K=192.
- **exp371_grad_diagnostic_short** — graded topology with grad-norm logging.
- **exp372_grad_diagnostic_uniform_short** — uniform topology with grad-norm
  logging (the cleaner baseline for L0/L5 ratio extraction).
- **exp373_graded_v_qk_L0up_short** — L0-only Q/K bump (192→256) motivated
  by the L0 outlier in diagnostics.
- **exp354_graded_out_tph_upper** — full-scale best so far (1.2196).
