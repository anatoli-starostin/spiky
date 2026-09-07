# LightMultiHeadLUT: every run, 16K and 48K

Survey of **all** LightMHL runs on branch `research/ffn_replacement_fix` (issue
[#112](https://github.com/anatoli-starostin/spiky/issues/112)), as of **2026-09-07 15:20 IDT**,
remote head `59c7f669`. Sibling of [`LOOKUPFFN_LINE.md`](LOOKUPFFN_LINE.md), which covers the
earlier 4,000-step confidence-gate arms; this file covers the 16K/48K line that grew out of it.

Every number here is on the corrected protocol (`evaluate_bpb_fixed`, bs48 × 100, leading 12
rows skipped, 2,451,456 held-out tokens of `shard_06542.parquet`). See
[`../FIXED_EVAL.md`](../FIXED_EVAL.md).

---

## 0. The bug that started the branch, and the fix

The historical per-run trainers built the val loader at the **training** `device_batch_size`
and scored a fixed `eval_steps=10`. The loader walks the held-out shard deterministically from
token 0, so the number of validation tokens scored was `device_batch_size × 512 × 10` — a
function of the training batch size. LUT runs (`device_batch_size` 12) were scored on the first
**61,440** tokens; vanilla baselines (`device_batch_size` 48) on the first **245,760**. Those
are different, nested slices of the val stream: the comparison that produced the earlier "LUT
matches vanilla" reads was **comparing models on different data**.

The fix (`tools/fixed_eval.py`):

- **Always bs48 × 100 eval steps**, decoupled from training entirely.
- **Skip the leading 12 rows** — rows 0 and 8 are the two easiest 512-token spans in the shard.
  Scored set is rows `[12, 4800)` = 2,451,456 target tokens, identical for every model.
- **One eval set, two call sites** — the training-loop curve and the standalone final scorer
  call the same function, so a run's curve and its final number are the identical measurement.
- **Clone-then-score.** The nanochat loader yields views into one GPU buffer that the next
  yield overwrites via a `non_blocking` HtoD copy; interleaving forwards with further `next()`
  calls raced the copy and silently corrupted later batches (repeated calls drifted
  1.147→1.153). `fixed_eval` drains all eval batches with `.clone()` first.

**Size of the correction:** −0.036 for the vanilla baselines (bs48, wider window already),
−0.008 to −0.011 for the LUT runs (bs12). Because the two classes were corrected by *different*
amounts, several signs flipped. The clearest casualty: `exp_n_0129` was reported as −0.020
**below** vanilla and is actually **+0.006 above** it.

| anchor | originally reported | corrected |
|---|---|---|
| `exp_n_0135` vanilla @16K seed 1 | 1.20144 | **1.165147** |
| `exp_n_0151` vanilla @48K | 1.151444 | **1.115420** |
| `exp_n_0157` vanilla @144K | 1.147985 | **1.111369** |
| `exp_n_0127` Fast nap7/tph128 @16K | 1.194711 | **1.187011** |
| `exp_n_0129` Fast nap8/tph256 @16K | 1.181484 | **1.170961** |

Runs from `exp_n_0176` onward use the corrected trainer, so their `metrics.csv` curve is
already the corrected metric (`correction: 0.0` in their `corrected_score.json`). Older runs
were re-scored from checkpoints; **use `corrected_score.json`, not `summary.json`, for those.**

### Baselines used throughout

| | bpb | run |
|---|---|---|
| vanilla dense @16K, seed 1 | **1.165147** | `exp_n_0135` (re-scored) |
| vanilla dense @16K, seed 2 | **1.161798** | `exp_n_0176` |
| vanilla dense @48K | **1.115420** | `exp_n_0177` (re-run) / `exp_n_0151` (re-scored) — agree to 6 dp |
| vanilla dense @144K | **1.111369** | `exp_n_0157` (re-scored) |
| **vanilla 16K seed spread** | **0.00335** | 0135 vs 0176 — the noise floor for every claim below |

---

## 1. What LightMHL is

`src/spiky/lutorch/light_multi_head_lut.py`. Same tables, same anchor-pair sign addressing and
same `embedding_bag` gather as `FastMultiHeadLut`; the difference is **the backward**:

- **Fast** builds a full-`K` softmax *surrogate* over all `2^nap` cells of each table, so `x`
  receives gradient through a differentiable relaxation of the routing decision.
- **Light** addresses with `pack(sign(d).detach())` — no STE, no temperature surrogate. Gradient
  reaches `x` **only through the confidence score**. If the score is flat, the routing side gets
  essentially nothing.

That is why the confidence form is not cosmetic for Light: it *is* the input-side gradient path.
Light has no learnable temperatures (12 fewer parameters than the matched Fast model) and no
`K`-sized backward buffer, which is why it can run nap=9 (K=512) at bs12 where Fast OOMs.

Wrapped by `CompressionMultiHeadLUT` (`src/spiky/lutorch/compression_mhl.py`): `compress`
384→48, per-head LUT, `decompress` 48→384, `decompress.weight` zero-initialised.

---

## 2. The confidence forms

With `m_j = |d_j|`, `d = z[a_j] − z[b_j]` and `P = Π_j σ(2 m_j)`:

| form | score | property |
|---|---|---|
| `bounded` | `P` | collapses as `nap` grows (0.054 at nap=8 — an 18.4× forward attenuation) |
| `bounded_norm` | `P^(1/nap)` = `exp(mean_j logsigmoid(2 m_j))` | nap-invariant, but **saturates**: `d/dm` falls 0.0619 → 0.0006 over m = 0.1 → 3.0 |
| `margin` | `(Σ_j m_j) · P` | **exactly the LookupFFN kernel** (arXiv 2403.07221); `Σ m_j` holds `d/dm` near 1 indefinitely |

`margin` was verified against nucstar's reference implementation (`research/lookupffn`,
`lookup_ffn.py`) at |z| = 0.01/0.1/1/3/10: **max relative difference 1.9e-15**. The paper writes
`score = (Σ_j m_j) / Π_j (1 + e^{−2 m_j})`, and `σ(2m) = 1/(1 + e^{−2m})`, so it is our `margin`
term for term. The bit-packed sign address is the same function too — ours packs MSB-first and
theirs LSB-first, which relabels cells within a table but induces the identical partition. **No
new code was needed to run the LookupFFN score.**

---

## 3. All LightMHL runs at 16K

`H=4`, `d_in=d_out=48`, depth 6, `d_model` 384, `seq_len` 512, effective batch 24,576 tokens,
lr 3e-4, cosine + 10% warmup, wd 0.1, seed 1, bf16, `device_batch` 12 × `grad_accum` 4 (except
the vanilla baselines at bs48). All differences from that are in the table.

`exp_g_*` ran on **gpustar** (RTX 5090); `exp_n_*` on **nebius** (H100 80GB).

| run | host | form | z_norm | no-decay | nap (K) | tph | anchor policy | params | **bpb @16K** | vs vanilla@16K | ×sd | h | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `exp_n_0181` | n | bounded_norm | – | – | 8 (256) | 256 | full_cov | 104,952,576 | 1.477708 *(4K proxy)* | – | – | 0.33 | 0.293 |
| `exp_n_0186` | n | bounded_norm | – | – | 7 (128) | 128 | full_cov | 48,477,312 | 1.208987 | +0.043840 | 13.1 | 0.90 | 0.202 |
| `exp_g_0189` | **g** | bounded_norm | – | ✔ | 8 (256) | 128 | full_cov | 67,351,680 | 1.207493 | +0.042346 | 12.6 | 0.89 | 0.200 |
| `exp_n_0185` | n | bounded_norm | – | – | 8 (256) | 128 | full_cov | 67,351,680 | 1.206222 | +0.041075 | 12.3 | 0.96 | 0.215 |
| `exp_g_0190` | **g** | bounded_norm | ✔ | ✔ | 8 (256) | 128 | full_cov | 67,352,256 | 1.203936 | +0.038789 | 11.6 | 0.91 | 0.205 |
| `exp_n_0184` | n | bounded_norm | – | – | 8 (256) | 256 | full_cov | 105,100,416 | 1.201075 | +0.035928 | 10.7 | 1.50 | 0.338 |
| `exp_n_0194` | n | **margin** | ✔ | ✔ | 7 (128) | 128 | full_cov | 48,477,888 | 1.183957 | +0.018810 | 5.6 | 0.97 | 0.218 |
| `exp_g_0191` | **g** | **margin** | ✔ | ✔ | 8 (256) | 128 | full_cov | 67,352,256 | *1.178680 @15K, **crashed*** | +0.013533 | 4.0 | – | – |
| `exp_n_0197` | n | **margin** | ✔ | ✔ | 8 (256) | 128 | **distinct** | 67,352,256 | 1.177476 | +0.012329 | 3.7 | 1.03 | 0.233 |
| `exp_n_0192` | n | **margin** | ✔ | ✔ | 8 (256) | 128 | full_cov | 67,352,256 | 1.177081 | +0.011934 | 3.6 | 1.04 | 0.233 |
| `exp_n_0198` | n | **margin** | ✔ | ✔ | 8 (256) | 128 | **disjoint** | 67,352,256 | 1.174424 | +0.009277 | 2.8 | 1.03 | 0.232 |
| `exp_n_0201` | n | **margin** | ✔ | ✔ | 7 (128) | 256 | full_cov | 67,352,256 | 1.171830 | +0.006683 | 2.0 | 1.42 | 0.320 |
| `exp_n_0195` | n | **margin** | ✔ | ✔ | 9 (512) | 128 | full_cov | 105,100,992 | 1.168194 | +0.003047 | 0.9 | 1.08 | 0.243 |
| `exp_n_0196` | n | **margin** | ✔ | ✔ | 8 (256) | **256** | full_cov | 105,100,992 | **1.163912** | **−0.001234** | −0.4 | 1.56 | 0.351 |

*(×sd = multiples of the 0.00335 vanilla seed spread. `exp_n_0181` is the 4,000-step proxy from
the earlier arm-B line and is not comparable to the 16K column.)*

**Reference points at the same budgets, all 16K, corrected:**

| run | what | params | bpb @16K |
|---|---|---|---|
| `exp_n_0127` | Fast gate-off, nap7/tph128 | 48,477,324 | 1.187011 |
| `exp_n_0188` | Fast + soft_topk (2-cell) backward, nap7/tph128 | 48,477,324 | 1.185226 |
| `exp_n_0129` | Fast gate-off, nap8/tph256 | 105,100,428 | 1.170961 |
| `exp_n_0135` | **vanilla dense** | 35,792,640 | **1.165147** |

Note `exp_n_0194` (Light+margin, 48.48M) = 1.183957 **beats** `exp_n_0127` (Fast, same
geometry, 48.48M) = 1.187011 by −0.003054 — Light with the LookupFFN score is no longer paying
the +0.043 penalty that arm B measured at 4K.

### Which runs nearly match vanilla at 16K

Two, and only in the top param class:

- **`exp_n_0196` (105.1M) = 1.163912.** That is **−0.001234 below** vanilla seed 1 and
  **+0.002114 above** vanilla seed 2 — i.e. it lands *inside* the two-seed vanilla band. On one
  seed this is parity, not a win.
- **`exp_n_0195` (105.1M) = 1.168194**, +0.003047 — 0.9× the seed spread, also parity.

At 67.35M the best is `exp_n_0201` at +0.006683 (2.0× sd) — outside the band but close. At
48.48M the best is `exp_n_0194` at +0.018810.

---

## 4. All LightMHL runs at 48K — and the gap does **not** hold

| run | what | params | bpb @48K | vs vanilla@48K | h | s/step |
|---|---|---|---|---|---|---|
| `exp_n_0177` | **vanilla dense** | 35,792,640 | **1.115420** | — | 1.72 | 0.129 |
| `exp_n_0200` | Light+margin, nap8/tph256 | 105,100,992 | **1.134538** | **+0.019118** | 4.69 | 0.351 |
| `exp_n_0171` | Fast gate-off, nap9/tph256 | 180,597,900 | 1.134511 | +0.019091 | 16.25 | 1.219 |
| `exp_n_0199` | Light+margin, nap8/tph128 disjoint | 67,352,256 | 1.136588 | +0.021168 | 3.10 | 0.232 |
| `exp_n_0155` | Fast gate-off, nap7/tph128 | 48,477,324 | 1.145471 | +0.030051 | 3.82 | 0.286 |
| `exp_n_0158` | Fast gate-off, nap7/tph128 **@144K** | 48,477,324 | 1.137558 | +0.026189 *(vs vanilla@144K 1.111369)* | 11.51 | 0.288 |

**This is the central finding of the survey.** `exp_n_0196` reaches vanilla parity at 16K
(−0.0012). Forked to 48K it becomes `exp_n_0200` and lands at **+0.019118**. The reason is that
both improve over the 3× horizon but vanilla improves **1.69× more**:

```
vanilla   1.165147 -> 1.115420   =  -0.049727
0196/0200 1.163912 -> 1.134538   =  -0.029374
```

And it is not a Light quirk. `exp_n_0171` (Fast, 180.6M, gate off, a completely different
backward) did exactly the same thing: parity at 16K (−0.0002) → **+0.019091** at 48K. Two
architectures, two backward paths, two parameter counts — **the same +0.019 at 48K, agreeing to
3e-5.** Whatever the LUT FFN is missing, it is not something the confidence form or the routing
gradient reaches.

**Do not read the step-16,000 row of a 48K run as a 16K result.** It sits on the 48K cosine
schedule and is systematically worse: vanilla 1.178467 there vs 1.165147 as a finished 16K run.
Step-aligned on the *same* schedule the gap is already **+0.007966** at step 16,000
(0200 1.186433 vs vanilla 1.178467) and roughly doubles by 48,000.

Trajectories, all on the 48K schedule:

| step | vanilla | `0200` L/105M | `0199` L/67M | `0171` F/180M | `0155` F/48M |
|---|---|---|---|---|---|
| 4,000 | 1.435410 | 1.401593 | 1.423623 | 1.421090 | 1.469477 |
| 8,000 | 1.249850 | 1.250391 | 1.260585 | 1.254139 | 1.282913 |
| 16,000 | 1.178467 | 1.186433 | 1.192137 | 1.183450 | 1.210289 |
| 24,000 | 1.152844 | 1.164719 | 1.168283 | 1.161929 | 1.186517 |
| 32,000 | 1.133479 | 1.148078 | 1.151254 | 1.146470 | 1.171519 |
| 40,000 | 1.120560 | 1.139221 | 1.141021 | 1.137995 | 1.161071 |
| 48,000 | **1.115420** | 1.134538 | 1.136588 | 1.134511 | 1.156018 |

The LUT runs are *ahead* at step 4,000 and lose steadily from step 8,000 on.

---

## 5. What each knob is worth (16K, one seed each, sd = 0.00335)

| change | runs | Δ bpb | ×sd |
|---|---|---|---|
| **`bounded_norm` → `margin`** | 0190 → 0192 | **−0.026855** | **−8.0** |
| tables over cells @105M | 0195 → 0196 (tph 128→256 vs nap 9→8) | −0.004281 | −1.3 |
| tables over cells @67M | 0192 → 0201 (tph 128→256 vs nap 8→7) | −0.005251 | −1.6 |
| `z_norm` on the code | 0189 → 0190 | −0.003556 | −1.1 |
| anchor policy `disjoint` | 0192 → 0198 | −0.002657 | −0.8 |
| anchor policy `distinct` | 0192 → 0197 | +0.000395 | +0.1 |
| exempt tables from weight decay | 0185 → 0189 | +0.001270 | +0.4 |
| doubling the table budget | 0192 → 0196 | −0.013168 | −3.9 |
| doubling the table budget | 0194 → 0201 | −0.012127 | −3.6 |

**Only one of these is a result at one seed: the confidence form.** `margin` is worth
**−0.0269 bpb, 8× the seed spread**, at fixed geometry, fixed everything. It is the single
largest lever found on this line and it costs nothing — no parameters, no new code, no
throughput (0190 0.205 s/step → 0192 0.233 s/step, and that 14% is the z_norm/host difference,
not the score).

Everything else is inside or barely outside the noise floor on one seed:

- **Tables beat cells at matched budget, twice.** −0.0043 at 105M and −0.0053 at 67M, both from
  spending the budget on more tables per head rather than more cells per table. Individually
  1.3× and 1.6× sd — neither is decisive, but they are two independent points pointing the same
  way at different budgets, which is worth more than either alone.
- **Anchor policy is nearly inert.** `canonical_disjoint` (each table's pairs use 16 distinct
  coordinates of 48 — a partial matching) is the best of the three at −0.0027, 0.8× sd.
  `canonical_distinct` is a wash. Enum in `src/spiky/lutorch/lut_helpers.py`, selected via
  `lut_anchor_policy` in `config.json` and mapped in `tools/model_build.py::_anchor_policy`.
- **Exempting tables from weight decay is not a win.** `exp_g_0189` (no-decay) is +0.001270
  *worse* than its matched control `exp_n_0185`. It was retained because it is harmless and was
  in the config lineage, not because it helped.
- **Table budget still buys ~0.013 per doubling**, measured twice. That is nearly double the
  −0.007455/doubling law fitted on the earlier Fast grid, i.e. the Light+margin arm is still on
  a steeper part of its curve — but note both doublings are +0.019 apart from closing the 48K
  gap, and the 48K evidence (§4) says the budget is not where the remaining deficit lives.

---

## 6. In progress

**`exp_n_0202_light_margin_znorm_nap9_48k_seed1`** — nebius, running now.

- **Config:** fork of `exp_n_0195` (Light + `margin` + `z_norm` + `tables_no_decay`, H4, nap=9
  (K=512), tph=128, d48, 105,100,992 params) with **only** `n_steps` 16000 → 48000 (cosine +
  warmup auto-stretch to 4,800). Seed 1, `device_batch` 12 × `grad_accum` 4, eval every 500.
- **Purpose:** completes the 48K three-way at ~105M — `0202` (more **cells**: nap9/tph128) vs
  `0200` (more **tables**: nap8/tph256, done, 1.134538) — the 48K version of the tables-vs-cells
  A/B that `0196` vs `0195` ran at 16K.
- **Progress at the last committed artefact (`59c7f669`, 15:11 IDT):** step **11,000 / 48,000**,
  val bpb **1.217808** (8500 → 1.247415, 9000 → 1.240118, 9500 → 1.233944, 10000 → 1.227703,
  10500 → 1.222260, 11000 → 1.217808).
- **ETA:** its 16K sibling `exp_n_0195` ran 16,000 steps in 1.078 h = **0.2426 s/step**, so 48K
  ≈ **3.2 h total** and the remaining 37,000 steps ≈ **2.5 h**, i.e. **~17:40 IDT today**. This
  is extrapolated from the sibling's throughput, not read off the live process; it slips if the
  H100 is shared.
- **Prediction:** at 16K, more tables beat more cells by −0.0043 (`0196` 1.163912 vs `0195`
  1.168194). If that carries, `0202` lands near **1.139** — behind `0200`'s 1.134538 and around
  +0.024 from vanilla@48K. A materially better result would be the first evidence that the
  cells axis behaves differently at long horizon.

### Incomplete / abandoned

| run | what | stopped at | last bpb | why |
|---|---|---|---|---|
| `exp_g_0191` | gpustar, Light+margin nap8/tph128, 16K | 15,100 / 16,000 | 1.178680 @15,000 | `torch.AcceleratorError` cudaErrorLaunchTimeout / **Xid 8 RC watchdog** on the desktop-driving RTX 5090. Environmental. Re-run verbatim on the H100 as `exp_n_0192` → 1.177081, no crash. See `CRASHED.md`. |
| `exp_g_0192` | gpustar, **BH4** addressing (not Light), 16K | 10,500 / 16,000 | 1.230552 | stopped by request; no checkpoint. See `STOPPED.md` and §7. |
| `exp_n_0193` | 48K fork of `exp_n_0192` (full_coverage) | 6,000 / 48,000 | 1.310791 | superseded by `exp_n_0199` (the 48K of the better `disjoint` policy) and `exp_n_0200`. |
| `exp_n_0187` | Light `bounded_norm` nap7/tph128 @48K | 29,500 / 48,000 | 1.193214 | overtaken by the `margin` line before it finished. |
| `exp_n_0188_…_innerres` | Light + `lut_inner_residual` | 9,000 / 16,000 | 1.256461 | clearly diverging from control; `ABANDONED.md`. |

---

## 7. The BH4 side-branch (not Light, recorded for completeness)

`exp_g_0192` (gpustar) replaced anchor-pair addressing *and* the compression projection with
LookupFFN's **BH4** block-Hadamard transform — coordinate-sign addressing, no anchor pairs at
all, nap=7, 4 heads, 128 tables/head, `decompress` on top, 48,427,008 params (−28.1% vs the
matched Light run). Stopped by request at step 10,500/16,000:

| step | `exp_g_0192` BH4 | vs `exp_g_0191` (Light+margin, matched score) | vs `exp_g_0190` |
|---|---|---|---|
| 10,500 | 1.230552 | **+0.026201** | +0.001501 |

The gap to `0191` was flat from step 4,000, so it was not still resolving. Adjusting for the
table-halving handicap (≈ +0.0075 from the budget law) still leaves ≈ **+0.019**. On this
evidence, at a matched score, **coordinate-sign addressing is behind anchor-pair addressing** —
one seed, stopped early, handicap extrapolated rather than measured. Implementation in
`src/spiky/lutorch/bh4_multi_head_lut.py`; verification battery in `diag_bh4_verify.py`.

A separate hash-quality diagnostic (`diag_hash_quality.py`) settled the "is it the hash or the
gradient?" question for Light: every run addresses **83–91%** of its cells at **0.86–0.90** of
maximum address entropy, and **Fast — the best performer — has the worst address distribution
of all of them.** Routing quality is not the bottleneck; the input-side gradient path is, which
is exactly what `margin` improves.

---

## 8. Throughput, parameters, memory

At the 16K sizing (`device_batch` 12 × `grad_accum` 4), s/step on the host that ran it:

| config | params | s/step (nebius H100) |
|---|---|---|
| vanilla dense | 35,792,640 | 0.129 (bs48) |
| Light nap7/tph128 | 48,477,888 | 0.218 |
| Light nap8/tph128 | 67,352,256 | 0.233 |
| Light nap9/tph128 | 105,100,992 | 0.243 |
| Light nap8/tph256 | 105,100,992 | 0.351 |
| Light nap7/tph256 | 67,352,256 | 0.320 |
| **Fast** nap9/tph256 | 180,597,900 | **1.219** |
| Fast nap7/tph128 | 48,477,324 | 0.286 |

Two things worth keeping:

1. **`nap` is nearly free for Light, `tph` is not.** nap 7→8→9 at fixed tph128 costs
   0.218 → 0.233 → 0.243 s/step (+11% total) while the table budget quadruples; doubling tph
   costs ~+45%. Light has no `K`-sized softmax buffer, so cells are cheap in both time and
   memory — this is why nap=9 (K=512) runs at bs12 where the Fast soft-backward path OOMs
   (`exp_n_0162/0163/0164` were cancelled for exactly that).
2. **Light is 3–5× faster than Fast at comparable quality.** `exp_n_0200` matches
   `exp_n_0171`'s 48K bpb (1.134538 vs 1.134511) with **58% of the parameters and 29% of the
   wall clock** (4.69 h vs 16.25 h). If the LUT line has a live advantage right now, it is this,
   not the bpb.

Projection FLOPs for the compression wrapper are 147,456 per token pair (compress + decompress),
**0.125×** a vanilla FFN's — recorded in each run's `corrected_score.json`.

---

## 9. Conclusions

1. **The LookupFFN `margin` score is the real result of this line.** −0.0269 bpb on Light at
   fixed geometry, 8× the seed spread, for zero parameters and zero new code. It works because
   Light's only input-side gradient path is the score, and `bounded_norm` saturates
   (`d/dm` → 0.0006) exactly where Light's margins end up, while `margin`'s `Σ m_j` factor keeps
   `d/dm` near 1. Measured at init on our own module, swapping the string alone gives **8.9×**
   more gradient into `compress`/`x` with table gradients within 6%.
2. **Light with `margin` has erased arm B's +0.043 penalty and now beats Fast at matched
   geometry** (`0194` 1.183957 vs `0127` 1.187011 at 48.48M), at lower cost.
3. **16K parity with vanilla is reachable** — `exp_n_0196` at 105.1M is 1.163912 against a
   vanilla two-seed band of 1.161798–1.165147.
4. **16K parity does not survive to 48K.** `0200` = +0.019118. `exp_n_0171` (Fast, 180.6M, a
   different backward) = +0.019091. The agreement of those two numbers to 3e-5 is the strongest
   signal in the survey: the residual deficit is **architecture- and gradient-path-independent**,
   and neither the confidence form nor the parameter budget touches it. Vanilla simply extracts
   1.69× more from the extra 32K steps.
5. **Confounds, stated.** Every 16K arm here stacks `margin` on top of `z_norm` and
   `tables_no_decay`; `margin`-without-`z_norm` has never been run, so the −0.0269 is the
   *combination*'s credit assigned to the one variable that changed between `0190` and `0192`
   (`z_norm` and `no_decay` were already on in both). Every comparison is one seed against a
   0.00335 spread measured on **vanilla**, not on this architecture — the LUT runs' own paired
   noise floor has never been measured.

### What would move the needle next

- **The 48K +0.019 is the whole problem now.** It is not a routing, score, or budget issue —
  three independent knobs have been shown not to touch it. The informative experiment is one
  that changes *how the LUT FFN uses its extra steps*: an LR/schedule sweep at 48K, or a
  longer-horizon run to see whether the gap is a constant offset or still opening
  (`exp_n_0158` at 144K sits at +0.026 vs vanilla@144K, which suggests still opening).
- **A paired seed floor for this architecture** (two runs differing only by something provably
  irrelevant). Every "suggestive, not decisive" row in §5 is limited by borrowing vanilla's
  0.00335.
- **Deconfound `margin`:** `margin` without `z_norm` at nap8/tph128, one hour, tells us whether
  the −0.0269 is the score or the pair.
- **Tables over cells, confirmed or retired:** it is 1.3× and 1.6× sd at two budgets. `0202`
  will give the 48K version of that A/B for free when it lands.

---

## 10. Where things live

| what | path |
|---|---|
| LightMHL implementation | `src/spiky/lutorch/light_multi_head_lut.py` |
| Fast (surrogate backward) | `src/spiky/lutorch/fast_multi_head_lut.py` |
| BH4 addressing | `src/spiky/lutorch/bh4_multi_head_lut.py` |
| CompressionMHL wrapper (`lut_impl`, `forward_confidence`, `confidence_form`, `z_norm`) | `src/spiky/lutorch/compression_mhl.py` |
| Anchor-pair generation + `AnchorSamplingPolicy` | `src/spiky/lutorch/lut_helpers.py`, `src/spiky/lutorch/anchor_pairs_lookup.py` |
| The eval set | `experiments/ffn_replacement/tools/fixed_eval.py` |
| Config → model (incl. `_anchor_policy`) | `experiments/ffn_replacement/tools/model_build.py` |
| Trainer template | `experiments/ffn_replacement/train_fixed.py` |
| Standalone scorer | `experiments/ffn_replacement/tools/score_checkpoint.py` |
| Protocol write-up | `experiments/ffn_replacement/FIXED_EVAL.md` |
| Runs (one folder each: `config.json`, `metrics.csv`, `summary.json`, `loss.png`, `train.py`) | `experiments/ffn_replacement/runs_corrected/exp_{g,n}_NNNN_*/` |
| Corrected re-scores of pre-fix runs | `…/exp_*/corrected_score.json` |
| The 4K confidence-gate arms | `experiments/ffn_replacement/runs_corrected/LOOKUPFFN_LINE.md` |
| Diagnostics | `…/runs_corrected/diag_*.py`, `dump_margins.py`, `bench_*.py` |

Naming: `exp_g_NNNN` = gpustar (RTX 5090), `exp_n_NNNN` = nebius (H100). The index is shared —
check both disk **and** `git log --all` before claiming one. Note the collision hazard:
`exp_g_0192` (BH4, gpustar) and `exp_n_0192` (Light+margin repro, nebius) are unrelated runs.
