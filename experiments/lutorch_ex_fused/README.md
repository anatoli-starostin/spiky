# lutorch_ex_fused — fused-cartridge abl47 reproduction (refactored lutorch_ex)

> **Corpus note:** all runs in this experiment train on the **full 170-shard ClimbMix corpus**
> (`~/.cache/nanochat/base_data_climbmix`, expanded 2026-10-07). The old abl47 sweep (2026-10-04)
> trained on only a 4-shard looped subset, so **abl47-family numbers are NOT comparable** to anything
> here. See [`aux_runs/README.md`](aux_runs/README.md) for the full corpus finding and the seed-replicate
> / init-ablation runs (`06_conf_n1_s2`, `06_conf_n1_s3`, `06_conf_n1_oldbias`) moved there.

Six fused-cartridge ablations at the abl47 **champion topology** (h8/tph64/nap8/d48, 6-block
MinimalGPT, ~68M params), reproducing the old abl47 sweep on the current `main` lutorch_ex API.
One folder per ablation; each has a **self-contained `train.py` with the model defined inline**
(RoPE + SDPA attention, prenorm blocks, untied unembedder), a `config.json`, and a `README.md`.

| folder | cartridge (current main class) | old abl47 run | old val_bpb |
|---|---|---|---|
| 01_fmh_rational | FusedManifestoHardLUT(weight="rational") | fmh        | 1.15152 |
| 02_fmh_softsign | FusedManifestoHardLUT(weight="softsign") | fss_hard   | 1.15647 |
| 03_fms_rational | FusedManifestoSoftLUT(weight="rational") | fms        | 1.13949 |
| 04_fms_softsign | FusedManifestoSoftLUT(weight="softsign") | fss_smooth | 1.12941 |
| 05_conf_n2      | FusedConfidenceLUT(read_top_n=2)         | conf (ref n=2)   | 1.10606 |
| 06_conf_n1      | FusedConfidenceLUT(read_top_n=1)         | conf_n1 (ref n=1)| 1.11737 |

Deferred (NOT included): the 2 **quantised** fused confidence ablations. `FusedQuantisedConfidenceLUT`
n=1 hard-raises (`_require`: two-cell read only); n=2 has a fused forward but is the deferred,
not-yet-equivalence-blessed path. Run those from the reference `QuantisedConfidenceLUT`, or wait for
the quantised follow-up PR.

## Shared geometry / settings (identical in all 6 config.json)
LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs"); 6 blocks, n_embd 384,
n_head 6, seq_len 512, vocab 32768, untied unembedder. 48,000 steps, lr 3e-4 cosine with a 0.1*lr
floor, warmup_frac 0.1, wd 0.1, grad_accum 4, device_batch 12 (24,576 tok/step), seed 1, eval_every 500.
AdamW betas (0.9, 0.95) eps 1e-8; grad clip 1.0; no-WD group = LUT tables + every ndim<2 param.
cell_tv_penalty * 10 as a SEPARATE backward. table_dropout_rate 0.2. Faithful OLD-matching init
(tables Uniform[±1e-3] per-head CPU-seeded, decompress zeroed, compress Normal(0,0.02)).

## Corpus + eval
Corpus: nanochat **ClimbMix** via `tokenizing_distributed_data_loader_bos_bestfit` (deterministic from
token 0, BOS best-fit packing); tokenizer `RustBPETokenizer.from_directory(get_base_dir()/tokenizer)`,
vocab 32768. Eval: vendored `tools/fixed_eval.evaluate_bpb_fixed` — fixed bs 48 × 100 steps, skip 12
rows -> rows [12, 4800) of the val shard; `val_bpb = total_nats / (ln2 * total_bytes)` (bits/byte;
bit-for-bit = nanochat `evaluate_bpb` at skip_rows=0). abl47-comparable.

## Precision
**FP32 throughout** (PyTorch default — no mixed precision, no explicit dtype knob); the LUT FFN is an
fp32 island (`ProjectionMHL(cart, 384).float()` + an `x.float()` cast in forward). This matches
abl47's *actual* numerics: abl47's config carried a `compute_dtype="bf16"` field that its trainer
never applied (it ran fp32), so that dead field has been removed here rather than carried forward.

## wandb (flagged)
Minimal inline wandb (project "Spiky", group "lutorch_ex_fused", tags [lutorch_ex, fused, <cartridge>]),
fresh unique run name per launch. **ONLINE-or-die:** `wandb.init` runs FIRST — before any
data/model/GPU work — and there is NO offline fallback, so a launch with no network (e.g. inside
the sbox cage) or not logged in to wandb HARD-FAILS in seconds with a clear error. The old
`wandb_tracking.Tracker` was NOT vendored (it pulls spiky.util.wandb_integration + metric_glossary);
swap it in if you want the exact old tags/glossary.

## Launch
Per folder, on a CUDA box with the lutorch_ex CUDA extension built (fused cartridges require it).
**Must be launched as a BARE `python train.py`** on a box with OUTBOUND NETWORK and wandb logged in
— NOT via `sbox` (the cage has no network), or `wandb.init` hard-fails immediately (online-or-die):
    cd <NN_folder> && setsid nohup python train.py > stdout.log 2>&1 &
`SMOKE_STEPS=40 python train.py` for a quick sanity. Single-run checkpoint/resume via ckpt.pt (the
dataloader resume is approximate — restarts from token 0). NOT launched yet.


## Root-cause investigation: conf_n1 fused (~1.096) vs abl47 reference conf_n1 (1.1177)

The fused conf_n1 reproduces ~0.021 bpb LOWER than the abl47 reference-class run, reproducibly across 3 seeds
(1.096980 / 1.096822 / 1.094599; spread 0.0024, ~9x smaller than the gap -> real, not seed noise). Extensive
read-only + numerical investigation has RULED OUT, each with evidence:

- cell-TV penalty: bit-identical value AND gradient on the real checkpoint tensors (mean TV 0.004048587 both; grad norm identical).
- dtype / precision: fp32 throughout in both; TF32 off by default and unset; the fused forward produces ZERO anchor-bit flips vs the reference (fwd agree 1.5e-8, no discrete jumps).
- table dropout: identical semantics -- [B,G,tph] whole-table mask, inverted 1/keep, applied to the per-table score before the tph-sum, same gate, same global-RNG torch.rand, same effective rate 0.2, i.i.d. Bernoulli(0.8) with no reuse/axis-correlation; compiled path had no cudagraph RNG-freeze.
- backbone: strict=True load of the OLD trained weights into the NEW model (160/160 tensors) reproduces abl47's own final bpb to 1e-6 -> forward is mathematically identical.
- fused backward: equals autograd-through-the-reference to fp32 round-off for input/weights/log_beta/log_gamma (norm ratio 1.0, no dropped/detached term).
- torch.compile'd reference (abl47's ACTUAL path) vs eager: confidence score s_t BIT-IDENTICAL (zero mean-signed bias), output 1e-8 unbiased; Inductor used accurate libdevice exp/log1p (no fast-approx bias).
- optimizer / LR / schedule / grad-accum / clipping: all math-equivalent; both 48,000 OPTIMIZER steps and 1.18B tokens; lr stepped once per optimizer step (no 4x warmup compression); TV once per step; clip 1.0 after the TV backward.
- data pipeline: identical loader (tokenizing_distributed_data_loader_bos_bestfit, bs12 x seq512, split train), deterministic from token 0, same tokens in the same order.
- beta/gamma init + learnability: identical -- log(2)/0.0 learnable scalars in both; kwarg names/defaults unchanged.
- execution environment: same host and bit-identical torch 2.13.0 / cuda 13.0.96 / cudnn 9.20 / triton 3.7.1 / python 3.12.3 on the 10-04 box; fp32 (TF32 off) is GPU-robust. (The abl47 conf_n1 run's own 'vm' host metadata is not on this box.)
- decompress.bias init (the one non-math-equivalent code diff: NEW zero-inits it, OLD left it random U(+/-0.051)): ABLATED in 06_conf_n1_oldbias (random bias restored, dropout stream held identical). It tracked the NEW zero-bias seeds, NOT abl47 -> NOT the cause (see 06_conf_n1_oldbias/RESULT.md).

STILL OPEN: the realised random-init VALUES of the non-table, non-re-inited params (compress.weight/bias) may differ because the two runs build the model via different code paths (abl47: build_model skeleton + ProjectionMHL swap; new: direct build + explicit faithful re-init) -- a seed-realisation axis; and a from-scratch reproduction of the abl47 run itself.
