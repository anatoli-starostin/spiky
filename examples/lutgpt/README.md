# LUTGPT

A small (84.9 M params) transformer where every attention projection
(Q / K / V / out) and every per-layer residual is a `TinyMultiHeadLut`
table instead of a dense matmul — those LUT forwards are sign-pack
lookups, no dot product. The other matmuls that any transformer needs
are still there: the token-embedding gather, the scaled-dot-product
attention itself (`Q·Kᵀ`, `softmax(...)·V`), and the unembedder
`nn.Linear(D, vocab_size)`. Backward through every LUT uses a soft
surrogate over the full K-row neighborhood; see
[`../../paper/tinymhl_hybrid_smooth.tex`](../../paper/tinymhl_hybrid_smooth.tex)
for the math.

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
          MeanAbsNorm(E) -> emb_resid_lut (NAP=5, tph=256) -> D  ----.
          |                                                          |
          |   per layer x N_LAYERS:                                  |
          |     MeanAbsNorm(E)                                       |
          |     qk_lut       (NAP=4, tph=256, n_outputs=2*d_qk)      |
          |     v_lut        (NAP=6, tph=256, n_outputs=d_v)         |
          |     scaled-dot-product attention (RoPE on q,k)           |
          |     out_proj     (NAP=6, tph=512, n_outputs=E)           |
          |     residual_lut (NAP=5, tph=256, n_outputs=D) ----------+
          |     x_lut += out_proj(attn)                              |
          v                                                          v
       E-stream                                                   D-stream
                                                                  ln_final
                                                                  unembedder -> logits
```

Two independent residual streams: the E-stream (width E=96) is mutated only by
attention output projections; the D-stream (width D=384) is a pure accumulator,
fed by `emb_resid_lut` at the embedding and one `residual_lut` per layer.

## Training recipe

Two phases, switching simultaneously at step 8000 (`bs_switch_step ==
hard_switch_step`):

| step range | `forward_mode` | `device_batch_size` | tokens |
|---|---|---:|---:|
| 1–8000 | `hybrid_smooth` (top-2 blend) | 8 | 32.8 M |
| 8001–16000 | `hard` (single row) | 16 | 65.5 M |

(Total ≈ 98 M tokens; the original research recipe ran ~3.3× larger batches
for ~328 M tokens. Bump `device_batch_size{,_b}` and `total_batch_size{,_b}`
proportionally if you have headroom.)

The backward path is the *same* soft surrogate in both phases (see paper). Only
the forward path and the weight-gradient scatter differ:

- Phase A's hybrid_smooth forward keeps gradients dense across the K-row
  neighborhood so the LUT weights move smoothly. Cheaper-per-token at the
  smaller batch size.
- Phase B switches to the hard forward to harden the discrete decision
  boundaries while compensating with a larger batch.

Continuous LR schedule over all 16 000 steps: 10 % warmup, cosine decay to
0.1× peak.

Optimizers: AdamW for the dense (non-LUT) parameters, Lion for the LUT weight
tables (the `lut_lr=2e-4` is lower than `adam_lr=3e-4`).

## Prerequisites

- A single CUDA GPU.
- `spiky` installed in this checkout (`pip install -e .` from the repo root).
- A [nanochat](https://github.com/karpathy/nanochat) checkout, set up as
  below — the training script imports the tokenizer, the
  BOS-aligned data loader, and the bits-per-byte eval helper from it.

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
#    8 is enough for the lighter lutgpt recipe; download more if you scale
#    bs / n_steps up.
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
`weight_deltas.csv`, `stdout.log`, `checkpoint.pt` (~750 MB).

Smoke run before committing to the full 3 h: edit `config.json` and set
`n_steps=8`, `bs_switch_step=4`, `hard_switch_step=4`, `eval_every=4`,
`eval_steps=2`, `device_batch_size=4`, `device_batch_size_b=8`,
`context_size=256`. That exercises both phases in ~70 s and confirms
the nanochat setup is wired up.

## Math

The soft surrogate backward derivation — including the asymmetric
weight-gradient (1-row for `hard`, 2-row for `hybrid_smooth`) — is written up
in [`paper/tinymhl_hybrid_smooth.tex`](../../paper/tinymhl_hybrid_smooth.tex).
