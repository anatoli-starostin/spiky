# exp_g_0017 — q/k/v ATTENTION-side sweep (complement of exp101_sweep_outproj)

Tracking issue: **#108**. Run on **gpustar (RTX 5090)**.

The mirror image of nebius's `exp101_sweep_outproj`: that sweep varied **out_proj** and held q/k/v
fixed; this one **sweeps q/k/v** and pins out_proj.

```
SWEPT   q/k/v : inner_in ∈ {24, 48, 96}  ×  tph ∈ {32, 48, 64}   = 9 runs
        nap=6, inner_out=-1 (multihead → [N,6,64]), sep q/k, forward_mode="hard", use_bf16=True
HELD  out_proj: inner_in=48, tph=64, nap=6, n_heads=8, inner_out=48
        4,000 steps per run
```

Both the swept and held geometries are printed at startup into every run log, so the out_proj pin
is auditable per-run rather than assumed.

## Relationship to nebius's harness

`mh_compression.py` is copied **byte-identical** (`cmp` clean) — the `CompressionMultiHeadLUTMH`
subclass that adds `multihead_output`, extending rather than editing
`src/spiky/lutorch/compression_mhl.py`. `train_sweep.py` differs from nebius's in **only the
sweep-axis swap**: `QKV_IN`/`QKV_TPH` now drive `_C_IN`/`_C_TPH` (q/k/v), while `OUT_IN`/`OUT_TPH`
default to the pinned 48/64.

Per-run subdirs are **`qkvin{IN}_tph{TPH}/`**, deliberately distinct from nebius's
`in{IN}_tph{TPH}/` so the two sweeps can never be confused when both land on the branch.

## Two operational differences from nebius's `run_sweep.sh`

**1. 2-wide, not 3-wide.** Measured peak GPU for the largest point of this grid (96/64):
**10,304 MiB**, flat across 40 one-second samples — the caching allocator holds a steady
reservation, so there are no transient spikes.

```
3 × 10,304 = 30,912 MiB  ->  would fit, ~1.7 GB spare (5%)
2 × 10,304 = 20,608 MiB  ->  ~12.0 GB spare (37%)
```

3-wide is arithmetically viable; **2-wide was chosen explicitly**. `WIDE=3` is one env var away.
(The earlier 12-point grid reached **14,240 MiB** at 96/128, where 3-wide genuinely did *not* fit —
the ~7.7 GB/run figure in the original brief was about half the real usage, because q/k/v are
*three* LUT sites where nebius swept one out_proj.)

**2. Absolute paths.** nebius's script hardcodes `~/projects/spiky` and a relative
`../../../.venv` python. On gpustar this branch lives in the **worktree**
`~/projects/spiky-fmhl-next` (the primary checkout is on `live/walker2d-viz` and has no `.venv`
sibling at that depth), so both are absolute here.

## Smoke check — all 9 points build and forward-pass

`q_lut/v_lut → (4, 6, 64)`, `out_proj → (4, 384)`, `logits → (2, 512, 32768)`, "forward OK",
optimizer grouping OK on every point. `tph=48` is not a power of two and builds fine (`tph` is a
table *count*; only `2^nap` must be a power of two).

| inner_in | tph 32 | tph 48 | tph 64 |
|---|--:|--:|--:|
| **24** | 51,537,744 | 58,615,632 | 65,693,520 |
| **48** | 52,535,664 | 59,613,552 | 66,691,440 |
| **96** | 54,531,504 | 61,609,392 | 68,687,280 |

## ⚠ The two axes are not equally weighted — read results accordingly

```
tph 32 → 64 (2×)        : +14.2M params    <- effectively a CAPACITY axis
inner_in 24 → 96 (4×)   :  +3.0M params    <- nearly capacity-NEUTRAL
```

Spread is 51.5M → 68.7M (**1.33×**), and ~83% of it is the `tph` axis. LUT tables are
`H · tph · 2^nap · d_out`, so `tph` scales them directly, while `inner_in` only widens the compress
`Linear(384 → H·inner_in)`.

**Practical consequence:** the `tph` ranking will likely track parameter count, so a raw
"best val_bpb" winner is close to "biggest model wins". The genuinely interesting comparison is
**within each tph column, across inner_in** — that is the near-capacity-neutral geometry question
this sweep can actually answer. Worth deciding up front whether the headline is *best bpb* or
*best bpb per param*; they will probably disagree.

## Waves (2-wide, 9 runs → 5 waves)

```
wave 1: qkvin24_tph32 + qkvin24_tph48
wave 2: qkvin24_tph64 + qkvin48_tph32
wave 3: qkvin48_tph48 + qkvin48_tph64
wave 4: qkvin96_tph32 + qkvin96_tph48
wave 5: qkvin96_tph64
```

## Reproducing

```
sbox bash run_sweep.sh                    # WIDE=2 default; N_STEPS=4000 default
BAR_HANDLE=<handle> python3 sweep_progress.py &   # consolidated Slack bar, 1 message
```

`sweep_progress.py` posts a single consolidated bar: done-count out of 9, current best, fractional
fill by *total steps across all runs* (so it moves during a run, not only at completion), and a
top-3 ranking on finalize.
