# experiment/lx-ternin-5k16s: ternary input matrix T before the FFN, on the lutorch_ex control (seed 2)

Fork of the lutorch_ex control `lutorch_ex_conf_n1_single_noproj_5k16s_seed2` (val bpb 1.2961 @5000; seed 1: 1.2997).
Code base: this repo (anatoli-starostin/spiky), branch `feature/lutorch_ex_conf_n1_single_noproj_5k` @ `b61555bf` (`experiments/lutorch_ex/train_multi.py`, tools from
`n1_5k_setup/research/experiments/ffn_replacement/tools`). Control seed-2 config/run script: `config.json` here is byte-identical to
deepspike-ml `experiment/p384-wmap-seed2-5k16s` @ `2ccc4643` `research/tally_transformer/results/series_T_s0_seed2/control_seed2/config.json` (that run was made on the same H100).

Control config unchanged: 8 heads x 64 tables x 256 cells (nap 8, single probes) x 48-wide dense rows, no compress, learned zero-init decompress 384x384, seed 2 (`lut_base_seed` 20000),
n_steps 16000 / warm-up 1600 / cosine to 0.1x, stop at 5000, eval every 500, 24,576 tokens/step, lr 3e-4, 100 val batches (as in the control).

**Only change:** per layer, the FFN reads `T * ln2(h)` instead of `ln2(h)`; `T` is a 384x384 ternary matrix, {-1,0,+1} in the forward, real latents in the backward
(`ternary_in.py`, a self-contained port of `research/tally_transformer/src/tally_transformer/ternary_in.py` of DeepSpike-AI/deepspike-ml `experiment/p384-ternin-5k16s` @ `b7b7ca7f`;
same init, scaling, STE, no weight decay on the latents, same arm definitions). Opt-in via env `TERN_IN=topk|free`, `TERN_K` in `train_multi.py` (unset = the original model, unchanged).

| arm | T |
|---|---|
| tern4 | 2 x (+1) and 2 x (-1) per output row (top-k of the standardised latents) |
| tern16 | 8 x (+1) and 8 x (-1) per output row |
| ternfree | sign with learned dead zone (threshold init 0.5 standardised units), scale 1/sqrt(mean nonzeros) |

Per-layer nonzero fraction / flip rate (since the previous eval) / plus, minus fractions / temp (/ thr): `tern/*_l{layer}` in W&B at each eval, `[TERN]` lines in train.log, `tern_stats.json`.
Zero-init decompress means T gets no gradient until decompress leaves zero (same as the FFN tables).

Run: `bash chain.sh` (the 3 arms sequentially: train, W&B curve verification, ckpt upload to NEW `s3://deepspike-data/anatoly/tally_transformer/lutorch_ex/<run>/ckpt.pt`);
`SMOKE_STEPS=20 bash run.sh ARM` = smoke. Needs the n1_5k_setup tree, nanochat data, torch-2.9.1 env (see ~/work/ENVIRONMENTS.md of the worker).

Results: see RESULTS.md (added after the runs).
