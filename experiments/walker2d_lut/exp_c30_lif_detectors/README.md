# exp_c30 — LIF detectors as the Walker2d SAC actor's index front-end

Can the combined LIF detectors of `spiky.lutorch.lif_detectors_mhl.LIFDetectorsMHL`
(branch `exp/lif-detectors-mhl`, built with nucstar) replace the hyperplane sign-tests
that address our LUT actor — trained from scratch, inside SAC, not distilled from an
oracle?

> ## ⚠️ CORRECTION (2026-08-03): "1.8× the parameters" was wrong. It is 3.1×.
>
> This README, and every file in this directory, gave the hyperplane baseline as 49,152.
> **That is exp_c29's *table-only* count for its nap6/tph64 cells** (`tph * 2**nap * 12`);
> exp_c29's own totals were 56,064–70,912. The anchor actually used here — exp_c18,
> 4308.0 ± 500.1 — is **nap6/tph32**: table 24,576 + hyperplane `w`/`b` 3,456 = **28,032
> total**. So 87,361 is **3.12×** the baseline, not 1.8×.
>
> The *returns* are unaffected and were independently re-verified; only the parameter
> framing was. And since every model here carries the **same 24,576-entry table**, the
> honest comparison is the index front-end: **62,785 vs 3,456 — 18.2×.** That makes this
> experiment's per-parameter claim substantially weaker than originally written, which is
> what motivated exp_c30b and exp_c31.

**Yes, within noise, at 3.1× the parameters (18.2× the addressing cost).** 3 seeds,
100-episode deterministic CPU reference: **3931.3 ± 585.8** against exp_c18's hyperplane
cell at the same nap6/tph32 shape, **4308.0 ± 500.1** (6 seeds). Difference −376.7,
unpaired Welch se 395.1 — under one standard error.

## Why a port and not a drop-in

`LIFDetectorsMHL` is a `torch.nn.Module`. Our Walker2d SAC is JAX end to end and the
*environment itself* is MJX, so a torch actor would need a host round-trip per step,
breaking `jit` and the determinism this chapter has relied on since exp_c17. The module is
therefore reimplemented in `jax_lif_mhl.py` and held to the torch reference by an explicit
parity test rather than by inspection.

**The port needs no custom VJP.** `jax_lut_grad` needs one because its surrogate is a
*different function* from its forward — a pinned softmax standing in for a hard index.
Here the surrogate is already inside the forward as an additive, value-cancelling term:

    y = y_hard + y_addr - stop_gradient(y_addr)

with `prow(hard_bits)` a constant one-hot (the bits come from a `>`, which carries no
gradient) and the table detached inside `y_addr`. Transcribing that and letting JAX
differentiate it reproduces the decoupling term for term.

## Parity — 13/13, first run

`./run_parity.sh` at `input_dim=17, n_heads=1, n_outputs=12, nap=6, tph=32`. The two venvs
are disjoint (spiky has torch and no jax; walker2d_mjx has jax and no torch), so the
reference is dumped to an npz by one interpreter and asserted by the other.

| check | relative error |
|---|---|
| forward `hard` / `soft` / `st` | 1.07e-07 / 4.47e-08 / 1.07e-07 |
| `st == hard` identity, JAX side | 1.19e-07 |
| grad `table` | **0.0** (bit-identical) |
| grad `P` / `theta` / `d` / `w` / `r` / `tau_s_raw` / `tau_p_raw` / `log_temp_bit` | ≤ 5.31e-08 |

Structural check too: the table gradient touched **477 of 2048 rows** over 24 samples ×
32 tables — a hard scatter, not the full-table smear the torch module's docstring reports
hitting when the detach is missing.

The reference is dumped on **CPU**: `Tensor.prod(dim=)` fails to compile on this box's
RTX 5090 under torch 2.9.1+cu130, and `_prow` uses it. That is a torch/Blackwell issue,
not a model bug, and it never touches the JAX side (`jnp.prod` is fine). The module is
left unmodified; a one-line workaround exists (`term.log().sum(-1).exp()`, exact for hard
bits) if it is ever wanted on GPU under torch.

## Result

| seed | CPU-ref 100 ep | ep-sd | full-length | velocity |
|---|---:|---:|---|---:|
| 0 | 3268.8 | 958.8 | 57/100 | 2.903 |
| 1 | 4381.2 | 697.5 | 97/100 | 3.467 |
| 2 | 4143.8 | 1173.3 | 66/100 | 3.724 |
| **mean** | **3931.3 ± 585.8** | | | |

**Not param-matched, and it matters.** 87,361 actor params against 49,152 for the LUT
actors — 1.8×. The ordered-pair channel `P` alone is 55,488 of it (detector bank 62,785 +
table 24,576). So this says LIF detectors *can drive the actor*; it says nothing about
whether they are better per parameter.

### Two things the trace shows that the mean does not

**Addressing is not the bottleneck here.** Row coverage reaches ~100% of all 2,048 rows by
iteration 2,500 in every seed. The LUT actors of exp_c29 reached 24–58 of 64 rows per
table. Whatever limits this actor, it is not an under-addressed table.

**Every seed peaks before the end and gives return back as `eps` sharpens** — best MJX
3490 / 4454 / 4656 against final 2916 / 4302 / 3900. The checkpoint is the *final* actor
(`_actor.npz` is rewritten at every eval), so the quoted numbers pay that cost. The last
stretch of the anneal, roughly eps 0.6 → 0.3, is mildly harmful in all three runs. An
obvious follow-up is to stop the anneal at ~0.6, or to hold eps once the return plateaus.

`temp_bit` self-sharpens hard and early — 1.000 → ~0.10 by iteration 2,500 — then drifts
back up to ~0.19. Left trainable and unscheduled, per the module's design.

## Files

| file | what |
|---|---|
| `jax_lif_mhl.py` | the JAX port: membrane, `_prow`, `apply(mode=st/hard/soft)`, `address`, `init` |
| `torch_ref_dump.py` / `parity_check.py` / `run_parity.sh` | the two-venv parity test |
| `lif_sac.py` | exp_c09's LUT-SAC with the index front-end swapped; nothing else changed |
| `eval_lif_cpu.py` | 100-episode deterministic CPU reference — **the only number quoted** |
| `run_sweep_lif.sh` | the 3-seed sweep, determinism on |
| `collect.py` / `plot_c30.py` / `slack_bar_lif.py` | results table, figure, live Slack bar |

## Reproduce

```bash
./run_parity.sh                      # must print PARITY OK before anything else
nohup ./run_sweep_lif.sh > run_sweep_lif.log 2>&1 &
python collect.py                    # mjx venv
MPLCONFIGDIR=/tmp/mplcfg python plot_c30.py    # spiky venv (matplotlib)
```

`eps` anneals 2.0 → 0.3 over the gradient steps actually taken,
`(iters − warmup) × updates = 304,000` — proportional to the distillation recipe, which
annealed over its own 6,000 Adam steps. Annealing over *iterations* would finish the
schedule inside the first 2% of the run.

Training evals score at the **current** eps, not a pinned one: `eps` enters the membrane,
so it changes the bits and therefore which row each table addresses. A policy trained at
eps 2.0 and scored at 0.3 is a different function. The anneal ends at `--eval-eps`, so the
final proxy and the CPU reference are taken in the deployment regime, and
`eval_lif_cpu.py` reads that eps from the checkpoint rather than from a flag.
