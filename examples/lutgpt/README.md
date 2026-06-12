# LUTGPT

A six-layer transformer where every attention projection (Q / K / V / out) and
every per-layer residual is a `FastMultiHeadLut` table instead of a dense
matmul — those LUT forwards are sign-pack lookups, no dot product. The other
matmuls that any transformer needs are still there: the token-embedding gather,
the scaled-dot-product attention itself (`Q·Kᵀ`, `softmax(...)·V`), and the
unembedder `nn.Linear(D, vocab_size)`. Backward through every LUT uses a soft
surrogate over the full K-row neighborhood; see
[`../../doc/lutorch/lutgpt_research_report.pdf`](../../doc/lutorch/lutgpt_research_report.pdf)
for the math (Section 2) and the rest of the published write-up.

The configuration shipped here is the **narrow-backbone reference** of the
report (`exp755` of Section 6.4): a rank-coded backbone at half-width
(`E = 192`) with a wide Euclidean accumulator (`D = 384`) for the unembedder.
176.2 M parameters total. The full-width variant (`exp754`,
`E = D = 384`, 276.8 M params) is reached by setting `embedding_dim=384` and
`d_v=64` in `config.json`.

## What's in this directory

- `train.py` — single-GPU training script.
- `config.json` — all hyperparameters (architecture, optimizers, schedule).
- `loss.png`, `metrics.csv`, `summary.json` — populated after training.
- `temperatures.csv`, `weight_deltas.csv` — diagnostic time-series.
- `checkpoint.pt` — final model state.

## Architecture

```
tokens -> tok_emb_E [V, E]
          |
          MeanAbsNorm(E) -> emb_resid_lut (NAP=6, tph=256) -> D  ----.
          |                                                          |
          |   per layer x N_LAYERS:                                  |
          |     MeanAbsNorm(E)                                       |
          |     qk_lut       (NAP=4, tph=256, n_outputs=2*d_qk)      |
          |     v_lut        (NAP=6, tph=256, n_outputs=d_v)         |
          |     scaled-dot-product attention (RoPE on q,k)           |
          |     out_proj     (NAP=7, tph=512, n_outputs=E)           |
          |     residual_lut (NAP=6, tph=256, n_outputs=D) ----------+
          |     x_lut += out_proj(attn)                              |
          v                                                          v
       E-stream                                                   D-stream
                                                                  ln_final
                                                                  unembedder -> logits
```

Two independent residual streams: the E-stream (width `E = 192`) is the
rank-coded backbone, mutated only by the per-layer `out_proj`; the D-stream
(width `D = 384`) is a pure Euclidean accumulator, written by `emb_resid_lut`
once at the embedding and by one `residual_lut` per layer, and read by the
linear unembedder. The two streams are at different widths by construction —
that is the asymmetric architecture of Section 3.4 of the report.

## Training recipe

Single-phase: `forward_mode = hybrid_smooth` (top-2 blended forward + soft
backward) held fixed for the full 16 000 steps; cosine LR schedule with 10 %
warmup, decaying to 0.1 × peak; effective batch 48 × 512 = 24 576 tokens per
step (`device_batch_size = 24` with `grad_accum = 2` for memory on a single
GPU). Total compute ≈ 3.93 × 10⁸ training tokens.

Optimizers and precision:

- **AdamW** for the dense (non-LUT) parameters: `lr = 3e-4`, `wd = 0.1`,
  `betas = (0.9, 0.95)`.
- **Lion** for the LUT weight tables: `lr = 2e-4`, `betas = (0.9, 0.95)`,
  with an `fp32` master copy of every parameter alongside its momentum buffer.
- **LUT weights are stored in `bf16`** (`weight_dtype = bf16` in
  `config.json`). The forward gather and the weight gradient stay in `bf16`;
  the Lion master step casts the gradient to `fp32`, applies the update to the
  `fp32` master, and copies it back into the `bf16` parameter. Halves the LUT
  HBM footprint and the per-forward gather bandwidth relative to `fp32`
  storage.
- **`clip_grad_norm` to 1.0** over all trainable parameters at the end of every
  step. Required by the `bf16`-storage recipe to bound the gradient
  accumulation noise.

To train in `hard` mode from scratch (`exp756` of the report), set
`forward_mode = "hard"` in `config.json` and rerun.

## Prerequisites

- A single CUDA GPU.
- `spiky` installed in this checkout (`pip install -e .` from the repo root).
- A [nanochat](https://github.com/karpathy/nanochat) checkout, set up as
  below — the training script imports the tokenizer, the BOS-aligned data
  loader, and the bits-per-byte eval helper from it.

## Setting up nanochat

`train.py` only uses three things from nanochat: the trained BPE tokenizer,
the streaming ClimbMix data loader, and `evaluate_bpb`. So you don't need to
run nanochat's full speedrun; just the data + tokenizer prep.

```bash
# 1) clone nanochat next to your spiky checkout
git clone https://github.com/karpathy/nanochat ~/nanochat
cd ~/nanochat

# 2) install nanochat's deps with uv
#    (nanochat pins torch==2.9.1 + a few rust BPE / dataset packages)
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv isn't installed
uv venv
uv sync --extra gpu
source .venv/bin/activate

# 3) download ~8 shards of ClimbMix (~2B characters; one is the val shard).
python -m nanochat.dataset -n 8

# 4) train the BPE tokenizer (vocab 32768, ~2-3 min)
python -m scripts.tok_train

# 5) optional: report tokenizer compression
python -m scripts.tok_eval
```

That populates `$HOME/.cache/nanochat/{tokenizer,base_data_climbmix}/`,
which is where the data loader and bpb eval look (via
`nanochat.common.get_base_dir()`). Set `NANOCHAT_BASE_DIR` if you want
them in a different cache location.

## Running

From inside the **spiky** repo (with spiky's own venv active and
`spiky` installed editable, and the nanochat venv activated *before this
shell* so its packages are on `PYTHONPATH` — or just run from the same
venv with `uv pip install -e <nanochat>` for convenience):

```bash
export NANOCHAT_ROOT=$HOME/nanochat        # or wherever you cloned it
python examples/lutgpt/train.py
```

Outputs land alongside `train.py` (the script computes `EXP_DIR` from its own
location): `loss.png`, `metrics.csv`, `summary.json`, `temperatures.csv`,
`weight_deltas.csv`, `checkpoint.pt`.

Smoke run before committing to the full run: edit `config.json` and set
`n_steps=20`, `eval_every=20`, `eval_steps=2`, `device_batch_size=4`,
`total_batch_size=4096`. That exercises model construction, the bf16-Lion-master
recipe, and a validation pass in under a minute and confirms the nanochat
setup is wired up.

## Math

The soft surrogate backward derivation — including the asymmetric
weight-gradient (1-row for `hard`, 2-row for `hybrid_smooth`) — is Section 2 of
[`doc/lutorch/lutgpt_research_report.pdf`](../../doc/lutorch/lutgpt_research_report.pdf).
