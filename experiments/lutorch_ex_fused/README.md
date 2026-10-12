# lutorch_ex_fused — fused-cartridge abl47 reproduction (refactored lutorch_ex)

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
