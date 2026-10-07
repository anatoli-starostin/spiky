# nanochat baseline: decision brief

**Contents:** 1. Which architectures are best · 2. Cost to reproduce each · 3. What to cite as our paper's baseline, and the plan

Companion to the long report *nanochat: state of the project, October 2026* (`nanochat-state-2026-10.pdf`), referred to below as **LR**. Every number here comes from the LR. "Project" means [karpathy/nanochat](https://github.com/karpathy/nanochat) at HEAD [`92d63d4`](https://github.com/karpathy/nanochat/commit/92d63d4) (2026-07-03), plus its leaderboard and discussions. Prices assume ~$24/h for an 8×H100 node, the README's implied rate.

# 1. Which architectures are best

### (a) Canonical dense recipe at HEAD: d24 + fp8, which is what `runs/speedrun.sh` trains
- **Recipe:** `--depth=24 --target-param-data-ratio=8 --fp8`, 1,384,122,122 params (of which 604.0M are value-embedding lookups; see the confirmed table below), ClimbMix-400B data, a 32k tokenizer trained in-run. It is base → SFT; RL is not part of the speedrun.
- **Delta vs the launch d20:**
  - deeper (d24 vs d20) and trained on ClimbMix instead of FineWeb-EDU;
  - attention: FA3 with a "SSSL" sliding window;
  - value embeddings, per-layer resid/x0 scalars, smear/backout, padded vocab;
  - a single MuonAdamW optimiser, fp8 matmuls, and no midtraining stage.
  - The MLP is still ReLU² (4× expansion, two matrices, no gate, no bias).
- **Score:** the record (leaderboard Run 6, [`a825e63`](https://github.com/karpathy/nanochat/commit/a825e63), Mar 14 2026) is CORE **0.2626**, val_bpb 0.71800, with 1.65 h of pretraining. The GPT-2 bar is 0.256525.

### (b) Larger depth tiers that exist as reported runs
- **d26.** The README says only that "GPT-2 capability ≈ depth 26". Two FineWeb-era leaderboard entries used d26:
  - Run 2 (d26 + fp8, Feb 2): CORE 0.2578;
  - Run 3 (d26, 1M batch, Feb 5): CORE 0.2602.
  - There is no standalone d26 tier or price (the "$300 d26" tier was not found, per LR Revalidation).
- **d32**, Oct 2025, official, released as SFT only. Same launch architecture as d20, just scaled: 1.88 B params, 37.6 B tokens. Base CORE **0.3168**; GSM8K 0.1994 after RL; ChatCORE 0.2734.
- **d34**, Nov 2025, official base. Launch-era architecture, 2.22 B params, trained at ratio 40 ("2× longtrained" vs Chinchilla). CORE **0.3382**, val_bpb 0.7045.

### (c) Leaderboard and community variants that match or beat the baseline
- **#481 "$73 / 3.04 h GPT-2"** (= leaderboard Run 1, d24, Jan 29 2026). This is Karpathy's own entry, not a community one. It used the pre-ClimbMix FineWeb-EDU data and predates smear/backout. CORE 0.2585. It has since been superseded by Runs 2–6 at the same task.
- **Run 5** (d24, ratio 8.7, 1.80 h): CORE **0.2690**, the highest d24 CORE on the board. It was superseded only because the leaderboard ranks time-to-0.2565, not CORE.
- **decoderstack-d24** ([ChrisMcCormick](https://huggingface.co/ChrisMcCormick/decoderstack-d24), community):
  - the same d24 shape, but trained with a different, non-nanochat training stack;
  - 5.84 B ClimbMix tokens, its own 32k tokenizer (it does not match other d24 tokenizers), CORE **0.2517**.
- **`varlen-gsm8k`** is the nanochat side branch, at `d9dffe9`, into which decoderstack-d24 is verified to load. It is not an architecture of its own. Its name suggests variable-length attention plus GSM8K changes (UNCERTAIN: the branch contents were not read).

### (d) Not reusable today
- **d34 and d32 official checkpoints.** They fail `strict=True` loading on master: they lack value embeddings, `ve_gate`, smear and backout, and their tokenizers were trained on FineWeb-EDU. They run only at "an equally old commit" (@svlandeg, [#481](https://github.com/karpathy/nanochat/discussions/481)).
- **RTX-4070-fork checkpoints** (`Marcolini/nanochat-d24-base-*`):
  - 910 M params, 9 heads, FineWeb-EDU, weights only, CORE 0.149–0.151;
  - they need that fork's GPT class.
- **Launch d20, Sofie d14, Nekochu d24, jkminder d26:** never released by Karpathy (d20), or pre-smear/backout, non-upstream format, or HF ports with the speedrun mechanisms ablated (LR §A.2).

### Judgement
- **By CORE per dollar (our criterion), the best is the canonical d24 speedrun at HEAD.** It reaches CORE 0.2626 for ~$48, about 0.0055 CORE/$. For comparison, d32 gives ~0.0004 CORE/$ (0.3168 / $800) and d34 ~0.00014 CORE/$ (0.3382 / $2,500). It is also the only top entry that is runnable at HEAD.
- **By absolute CORE, the best is d34 (0.3382).** But it is frozen at a Nov-2025 commit on different data, so it is a reference point, not a usable architecture.
- **Among the d24 variants, the differences are within noise.** Run 5 (0.2690), Run 6 (0.2626) and decoderstack (0.2517) sit inside CORE noise of about ±0.008 per run, with a 7-run spread of 0.0165 (LR §2.3). None of them beats the canonical recipe in a way that matters.

*Sources:* LR Executive summary, §A.2, §B.1, §B.3, §1.4, §2.3, Revalidation.

### Confirmed d24 architecture and parameter budget (read from code at `92d63d4`)
| Item | Value at d24 | Code |
|---|---|---|
| Width | `n_embd` = 24 × 64 = **1536** (`depth × aspect_ratio`, rounded up to a multiple of `head_dim`) | [base_train.py L129–143](https://github.com/karpathy/nanochat/blob/92d63d4/scripts/base_train.py#L129-L143) |
| Heads | 12 heads × `head_dim` 128; `n_kv_head` = 12 (GQA supported but unused) | [base_train.py L133–138](https://github.com/karpathy/nanochat/blob/92d63d4/scripts/base_train.py#L133-L138) |
| Vocab | 32,768 (`tok_train` default); pad-to-64 is a no-op here; wte and lm_head **untied** | [tok_train.py L19](https://github.com/karpathy/nanochat/blob/92d63d4/scripts/tok_train.py#L19), [gpt.py L170–177](https://github.com/karpathy/nanochat/blob/92d63d4/nanochat/gpt.py#L170-L177) |
| MLP | `c_fc` 1536→6144 (**4 × n_embd**), `relu(x).square()`, `c_proj` 6144→1536. **Two matrices, no gate, no bias.** 18,874,368 params per block (8d²). RMSNorm is parameter-free | [gpt.py L131–141](https://github.com/karpathy/nanochat/blob/92d63d4/nanochat/gpt.py#L131-L141) |
| Attention | q/k/v/o, no bias: 9,437,184 per block (4d²). Window pattern SSSL, **short window = 512** (the inline comment's "2048 -> 768" is stale), layers 3,7,…,23 full 2048 | [gpt.py L77–80](https://github.com/karpathy/nanochat/blob/92d63d4/nanochat/gpt.py#L77-L80), [gpt.py L298–313](https://github.com/karpathy/nanochat/blob/92d63d4/nanochat/gpt.py#L298-L313) |

| Param group | Count |
|---|---|
| FFN, 24 blocks | 452,984,832 |
| Attention q/k/v/o, 24 blocks | 226,492,416 |
| `ve_gate`, 12 × (12→12) | 1,728 |
| wte | 50,331,648 |
| lm_head | 50,331,648 |
| Value embeddings, 12 odd layers × 32,768 × 1536 | 603,979,776 |
| Scalars (resid/x0 48, smear 25, backout 1) | 74 |
| **Total** (asserted by `num_scaling_params`, [gpt.py L390–409](https://github.com/karpathy/nanochat/blob/92d63d4/nanochat/gpt.py#L390-L409)) | **1,384,122,122** |

- **The ratio-8 denominator** is `transformer_matrices + lm_head` = **729,810,624**; wte and the value embeddings are excluded ([base_train.py L263–269](https://github.com/karpathy/nanochat/blob/92d63d4/scripts/base_train.py#L263-L269)). 8 × that → batch 2^20 → **5,568 steps** = 5,838,471,168 tokens ([L276–351](https://github.com/karpathy/nanochat/blob/92d63d4/scripts/base_train.py#L276-L351)), i.e. 4.22 tokens per *total* param.
- **FLOPs** ([`estimate_flops`](https://github.com/karpathy/nanochat/blob/92d63d4/nanochat/gpt.py#L319-L339)). Projections are exactly ⅔ FFN. With attention scores counted, the FFN share is 63.2% in the 512-window layers, 54.5% in the full layers, and 56.9% of the 4.775e9 training FLOPs per token. Attention projections are 28.5%, scores 8.3%, lm_head 6.3%.

# 2. Cost to reproduce each

| Variant | Depth / params | Tokens | Wall-clock | $ | CORE | Hardware | Loads into HEAD? |
|---|---|---|---|---|---|---|---|
| **Speedrun, current record** ([Run 6](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md)) | d24 / 1.384 B | 5.84 B (5,568 steps × 2^20; computed from HEAD code) | 1.65 h pretrain; ~2 h full pipeline (2–2.5 h with setup, UNCERTAIN) | **~$48**; ~$15 on spot | **0.2626** (averaging not stated) | 8×H100, fp8, FA3 | **Yes**; it *is* HEAD |
| Run 5 | d24 / ~1.38 B | ratio 8.7 | 1.80 h | ~$43 *(derived)* | 0.2690 | 8×H100 | Run with HEAD code (not a checkpoint) |
| [#481](https://github.com/karpathy/nanochat/discussions/481) / Run 1 (FineWeb-EDU) | d24 / 1,384,124,976 | not stated (4.33e19 FLOPs) | 3.04 h | ~$73 | 0.2585 | 8×H100 | No; Jan-2026 commit and data |
| Runs 2–3 (FineWeb-EDU) | d26 / — | — | 2.91 h / 2.76 h | ~$70 / ~$66 *(derived)* | 0.2578 / 0.2602 | 8×H100 | No; Feb-2026 commits |
| Launch d20 ([#1](https://github.com/karpathy/nanochat/discussions/1), Oct 2025) | d20 / 560,988,160 | 11.22 B | 3 h 51 min | $92.40 | 0.2219 | 8×H100 | No; never released, and old architecture |
| [d32](https://huggingface.co/karpathy/nanochat-d32) (official, SFT only; [#8](https://github.com/karpathy/nanochat/discussions/8)) | d32 / 1,879,048,192 | 37.58 B | ~31.3 h | ~$800 | 0.3168 (base) | 8×H100 | **No**; needs an Oct-2025 commit |
| [d34](https://huggingface.co/karpathy/nanochat-d34) (official base) | d34 / 2,217,082,880 | 88.7 B (ratio 40) | ~100 h | ~$2,500 | **0.3382** | 8×H100 | **No**; "equally old commit" |
| [decoderstack-d24](https://huggingface.co/ChrisMcCormick/decoderstack-d24) (community) | d24 / not stated | 5.84 B | not stated in LR | not stated in LR | 0.2517 | training hardware not stated; load verified on an H100 | **UNCERTAIN**: verified only on `varlen-gsm8k` @ `d9dffe9`, not on master |
| [Marcolini d24](https://huggingface.co/Marcolini/nanochat-d24-base-champion) (RTX-4070 fork) | "d24" / 910 M | — | — | — | 0.149–0.151 | RTX 4070 (fork) | No; fork's GPT class, no tokenizer |

**Notes on the table:**
- $ marked *derived* is wall-clock × $24/h, our arithmetic, not quoted by upstream.
- CORE noise is about ±0.008 per run, so differences under ~0.01 are not real.
- val_bpb is **not comparable** across the Mar 4 ClimbMix switch, nor across the Jul 3 `token_bytes` fix (`2ce972a`). CORE is the cross-era metric.
- No current-speedrun report card (chat metrics) is published (UNCERTAIN).

*Sources:* LR §B.1 (tiers and leaderboard row), §B.2 (launch d20), §A.2 (checkpoints, decoderstack, Marcolini), §2.3 (noise, baselines), Revalidation.

# 3. Which baseline our paper should cite

**Recommendation: cite and re-run the canonical nanochat d24 speedrun, pinned to commit `92d63d4`. Report it as the baseline both as Karpathy's published number (CORE 0.2626, Run 6) and as our own reproduction, mean ± sd over 3 seeds. Cite d32/d34 only as reported numbers, and use no community variant as a baseline.**

### Why this one
- **Reproducible by reviewers:** one script, standard 8×H100 at ~$48 (~$15 on spot), or one H100 (see below). It is the only top configuration a third party can re-run at HEAD; every alternative needs an old commit, another training stack or a fork.
- **Canonical:** the README and leaderboard define it as "the" nanochat model, so "nanochat d24 speedrun @ `92d63d4`" is an unambiguous citation.
- **Cheap enough to seed:** CORE noise (±0.008 per run) is about the size of our effects, so we need our own multi-seed mean.

### The baseline is a moving target: what the paper must pin
The repo has been quiet since 2026-07-03, but it switched datasets on Mar 4 2026 and changed the val_bpb byte accounting on Jul 3 2026, so a "nanochat baseline" without these pins is meaningless:
- **Code:** commit `92d63d4`, the exact `speedrun.sh` flags (`--depth=24 --target-param-data-ratio=8 --fp8`), plus `torch==2.9.1+cu128` (fp8 is nanochat's own `fp8.py`; no torchao), and the FA3 kernel revision fetched from the Kernels Hub.
- **Data:** `karpathy/climbmix-400b-shuffle`, the 170 downloaded shards, with val = the last shard (`shard_06542`). Also the sha256 of our trained tokenizer, since it is trained in-run.
- **Token budget:** ratio 8 gives 5,568 steps × 1,048,576 = 5.84 B tokens at HEAD (LR Part 3). Record it from our log anyway.
- **Eval:**
  - CORE = the 22-task DCLM bundle from S3; store its sha256 (size unverified).
  - val_bpb as computed after `2ce972a`.
  - ChatCORE is 5 tasks at HEAD (it was 6 at earlier commits); say which.
- **Hardware:** 8×H100 with fp8 and FA3. Wall-clock claims hold only on this setup. Our 5090 runs fall back to SDPA, with no FA3.

### Should we run d32/d34 as an upper reference?
**No; cite them.**
- Re-running costs ~$800 for d32 or ~$2,500 for d34, and at HEAD it would produce a *new* model, not Karpathy's. Running the published checkpoints would mean the old commit and FineWeb-EDU data, not comparable with our ClimbMix arm.
- Their reported CORE values (0.3168, 0.3382) are enough to show where the scaling curve goes, framed as "launch-era architecture, FineWeb-EDU, as reported".
- Only buy a stronger tier if a reviewer asks whether LUT gains hold at scale. Even then, a HEAD d26 or d32 run on our recipe is the right experiment, not the old checkpoints.

### The risk of citing community and leaderboard variants
- decoderstack-d24 has a different training framework and tokenizer, loads only on a side branch, and scores 0.2517, below the GPT-2 bar and within noise. Runs 1–3 are FineWeb-era and superseded; the RTX-4070 fork is a different model class. They are fine as related-work mentions, but never as the baseline row.

### One caveat on the comparison itself
LUT layers are not `nn.Linear`, so the LUT arm cannot use fp8 in its FFN (LR §2.4). The headline baseline row stays the canonical fp8 run, because that is what is citable. But the **matched control** for the LUT claim should be the same d24 at the same commit with `--fp8` off, so the comparison isolates the FFN rather than the precision.

### Can one H100 80GB replace 8×H100? Yes, with caveats (LR Part 3)
- **Numerically comparable, yes.** At HEAD, the global batch (2^20 tokens), the step count (5,568), the LRs and the weight decay all derive from param count and token ratio, never from GPU count. Only grad-accum changes: 4 → 32 at dbs 16.
- `MuonAdamW` runs single-rank natively. Its maths is per matrix, so sharding only decides who computes what.
- The run is not bit-identical: rank-strided data order and fp32 reduction order differ. But the same is true of two 8-GPU runs. Expect CORE within noise (±0.008); the README says "~identical results".
- CORE eval is rank-agnostic. Report the standalone `base_eval` CORE. The in-training CORE is a 500-example-per-task subsample and is not the reported number.
- **Caveats:**
  - edit `--nproc_per_node=8` in `speedrun.sh`;
  - memory is ≈ 70–75 GB at dbs 16 (UNCERTAIN; extrapolated from one 5090 report). Fall back to dbs 8, which has the same maths.
  - It takes ≈ 15–18 h end to end at about the same GPU-hours, so about the same $ (≈ $45–55 at $3/GPU-h).
  - On spot, add `--save-every`. Resume works but is only approximate for the dataloader.
- **Upstream evidence:** [#819](https://github.com/karpathy/nanochat/discussions/819) trained the same d24 (5,568 steps, 256× accumulation) on one RTX 5090 in 41.4 h, reaching CORE 0.2702 with the non-standard `L` window.

### What we should actually do
1. **Run** the canonical speedrun at `92d63d4` on rented 8×H100, **3 seeds**: ~3 × $48 ≈ **$150**. Use spot pricing if available, ~3 × $15.
2. **Run** the matched dense control, d24 at `92d63d4` with `--fp8` off, 3 seeds. Budget **~$170**: fp8 gave 1.17× throughput at d26 (LR Part 3 / LOG), so fp8-off is ~2.3 h ≈ $56 per run. This is our estimate.
3. **Pin and publish:**
   - the commit, flags and package versions;
   - the dataset name and shard range;
   - the tokenizer, eval-bundle and FA3 hashes;
   - the logged token count;
   - the hardware and seeds.
   Release our d24 checkpoint and logs, since no current-architecture base checkpoint is public.
4. **Cite, don't run:**
   - Run 6 (CORE 0.2626) as the canonical reference;
   - the GPT-2 bar (0.256525);
   - d32 (0.3168) and d34 (0.3382) as reported upper reference points;
   - the launch d20 (0.2219) only for history.
5. **Do small-scale LUT iteration at home first.** Run dense d12 at the same commit on the 5090: no rental cost, and no published d12 number exists, so we must make our own. Buy the 8×H100 runs only once the LUT arm works.
6. **Rough total: ~$320 for the d24 baseline package**, about $370 with a contingency rerun. Spot pricing, where available, cuts this roughly threefold.

*Sources:* LR Executive summary (recommendation), §A.2 (checkpoint loadability), §A.3 (FA3 fetch), §B.1, §B.3, §C (software stack), §2.3 (eval harness, noise, missing d12 numbers), §2.4 (fp8 only on `nn.Linear`; fork at `92d63d4`).
