# LUTGPT

A small (84.9 M params) transformer where every attention projection and
per-layer residual is a `TinyMultiHeadLut` table instead of a dense matmul.
The forward pass is a sign-pack lookup (no dot product); the backward pass
uses a soft surrogate over the full K-row neighborhood. See
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
- A [nanochat](https://github.com/karpathy/nanochat) checkout. The script
  pulls its tokenizer, data loader, and bpb evaluator from there.

## Running

```bash
export NANOCHAT_ROOT=/path/to/nanochat
python examples/lutgpt/train.py
```

Outputs land alongside `train.py` (the script computes `EXP_DIR` from its own
location).

## Math

The soft surrogate backward derivation — including the asymmetric
weight-gradient (1-row for `hard`, 2-row for `hybrid_smooth`) — is written up
in [`paper/tinymhl_hybrid_smooth.tex`](../../paper/tinymhl_hybrid_smooth.tex).
