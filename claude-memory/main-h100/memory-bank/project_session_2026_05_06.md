---
name: 2026-05-06 nanochat session — big unembedder & distillation framework
description: SOTA progression exp154→exp174 (1.7105→1.6478, vanilla=1.6256), distill_unembedder framework setup, LUT-vs-MLP distillation findings.
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
# 2026-05-06 nanochat session

## Headline results

**New SOTA: exp174 = val_bpb 1.6478** (481M params).
- Combines exp154's LUT qk_joint + BitAttention with the big unembedder MLP from exp173.
- Vanilla baseline (exp001, 23M params): **1.6256** — exp174 is +0.022 bpb (~1.4%) above vanilla.
- exp154 reference: 1.7105 → exp174 closes 75% of the gap to vanilla.

**The unembedder upgrade is the dominant lever.** Previous LM head was `LayerNorm(384) → Linear(384, vocab=32768)` (12.6M). New is `LN(384) → Linear(384, 3072, bias=False) → GELU → Linear(3072, vocab, bias=False)` (101.8M). Gain: **−0.063 bpb** for +89M params — bigger than any single architectural change in the residual stream.

## Experiment progression (key runs only)

| Exp | val_bpb | params | Δ vs exp154 | Change |
|---|---|---|---|---|
| **exp001 vanilla** | **1.6256** | 23M | — | classic transformer reference |
| exp154 | 1.7105 | 392M | — | LUT qk_joint + BitAttention (reference) |
| exp159 anneal_to_hard | 1.7996 | 392M | +0.089 | linear soft→hard attention |
| exp160 skip_v | 1.7872 | 391M | +0.077 | V skipped (v=x.expand) |
| exp162 h12_dv16 | 1.7402 | 430M | +0.030 | 2× heads, half d_v |
| exp166 mlp_qk_gelu_nobias | 1.7162 | 184M | +0.006 | per-head MLP for Q/K, h=2048, GELU+nobias |
| exp171 grow_v_out | 1.7071 | 335M | −0.003 | exp166 + 2× v_tph + 2× out_tph (first to beat exp154) |
| exp173 big_unembedder | 1.6552 | 424M | −0.055 | exp171 + big-unembedder MLP |
| **exp174 lut_qk_big_unembed** ✨ | **1.6478** | **481M** | **−0.063** | exp154 + big-unembedder MLP |
| exp175 lut_unembedder | killed | 514M | n/a | LUT unembedder; way slower convergence |

## Key findings

1. **Q/K can be replaced with a small per-head MLP** (3.2M/layer, h=2048, GELU+nobias+LN) at < +0.01 bpb cost vs LUT (37.7M/layer). The original LUT qk_joint was massively over-parameterized.
2. **V is load-bearing**: replacing with MLP cost +0.09 bpb (exp168) — V can't be cheaply replaced.
3. **Block output V2D→D2V wrap** (rank-canonicalization) is needed for stability when the unembedder is small Linear-only. With big-MLP unembedder, simpler block-output LayerNorm also works.
4. Halving d_qk (32→16) with H doubled (6→12) keeps params identical (H×n_outputs invariant) and gives same val_bpb within noise.
5. **The LM head was the biggest hidden bottleneck across all architectures.** Both exp154 (LUT qk) and exp166 (MLP qk) gain ~−0.05 bpb from the same big-unembedder swap.

## Distillation framework (`nanochat_exps/distill_unembedder/`)

**Goal**: standalone task to find a LUT-based unembedder that matches MLP — convergence speed matters because transformer training has fixed step budget.

**Files**:
- `extract_pairs.py` — runs frozen exp174 forward, dumps 100K (input[384], logits[32768]) pairs to `pairs.parquet` (~5.5 GB, .gitignored).
- `fit_baseline_mlp.py` — MlpUnembedder (same architecture as exp174's head). KL distillation against teacher logits.
- `fit_lut.py` — TinyMultiHeadLut(384→128 sparse → vocab=32768) + bias.
- `fit_lut_stack.py` — stacked LUT residual blocks. Flags: `--n_blocks`, `--block_tph`, `--block_n_heads`, `--share_weights`, `--dense_concat`, `--expand_dim`, `--lr_schedule`, `--init_std`.

**KL-loss leaderboard at 20 epochs (cosine LR 1e-3)**:

| Variant | params | KL @ 20 |
|---|---|---|
| **MLP** ✨ | 100M | **0.0072** |
| LUT 3×550 (NlogN coverage) — best LUT | 53M | 0.029 |
| LUT 3×550 dense_concat | 91M | 0.028 |
| LUT 3×550 multi-head K=4 | 53M | 0.030 (in progress) |
| LUT 3×550 share_weights ×3 | 26M | 0.037 |
| LUT 3×1100 (2×NlogN) | 94M | 0.031 |
| LUT 3×550 + expand_dim=1024 | 110M | 0.036 |
| LUT 6×275 (deeper, same budget) | 53M | 0.059 (constant LR) |
| LUT single (exp175 setup) | 134M | 0.439 |

**Findings**:
- MLP cosine 1e-3 reaches 0.0072 (well-trained) vs constant 3e-4 0.012.
- LUT cosine 1e-3 reaches 0.029 (vs constant 3e-4 0.050). Same 1.7× boost from cosine.
- Gap **MLP vs LUT held at ~4×** across all schedules tried.
- 3 blocks at NlogN coverage (3300 pairs/block, tph=550 at nap=6) is the sweet spot at 20 ep.
- Adding capacity (deeper / wider blocks / expander) underperforms at 20 ep — bigger models under-train.
- Hard lookup constraints gradient density (only selected entries get gradient per token); MLP gets ~150× more weight updates per token. This is the structural challenge.

## Open: how to make LUT distillation match MLP at 20 ep
- **smooth_mode disallowed** by user — must win with topology only.
- Recurrent (share_weights) showed promise per param-efficiency though slower at fixed budget; user wants to continue exploring tomorrow.
- Untried: parallel mixture per block, bottleneck blocks, custom complementary anchor sampling, gated residuals.

## Telegram bridge note
Hook `/home/starost/.claude/hooks/telegram_permission.py` updated this session:
1. Auto-allow for `ALWAYS_SAFE_TOOLS` (Read/Edit/Write/etc.) — was previously falling through to local prompt.
2. Auto-allow for safe Bash patterns (was previously falling through too).
3. PreToolUse matcher changed from `Bash` to `*` so Edit/Write tools also route through the hook.
