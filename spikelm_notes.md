# SpikeLM Notes

**Paper:** "SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms"
**Venue:** ICML 2024
**Repo:** https://github.com/Xingrun-Xing/SpikeLM/tree/main

---

## What it is

The first fully spiking mechanism for general language tasks (both discriminative GLUE and generative).
Takes BERT-base architecture and replaces all linear layers with spiking equivalents.

---

## Core Problem with Standard SNNs

Binary {0, 1} spikes don't have enough representational capacity for rich NLP semantics.

---

## Solution: Elastic Bi-Spiking

Spikes are extended along two axes:

- **Bi-directional:** values in {-1, 0, +1} (ternary) instead of {0, 1}
- **Elastic amplitude:** learned per-layer step size `alpha` (from LSQ, arXiv:1902.08153), initialized as `2 * |x|.mean() / sqrt(Qp)`
- **Elastic frequency:** T=4 temporal timesteps with leaky membrane integration

---

## Key Classes

### `ElasticBiSpiking` (torch.autograd.Function)

Ternary quantization in forward:
```python
q_w = (input / alpha).round().clamp(-1, 1)
w_q = q_w * alpha
```

Not truly differentiable — uses **surrogate gradients** (STE variant):
- `grad_input`: straight-through inside `[-1, 1]`, zero outside
- `grad_alpha`: real analytic LSQ gradient

### `SpikeLinear` (replaces nn.Linear)

Runs T=4 spiking steps before the linear projection:
```python
for i in range(T):
    mem = mem_old * 0.25 * (clip_val - output[i-1]).detach() + input[i]
    output[i] = ElasticBiSpiking(mem, alpha[i])  # → {-1, 0, +1} * alpha

out = F.linear(output, weight)  # standard matmul at the end
```

The `.detach()` on `mem_old` and `output[i-1]` means **no BPTT** — gradients don't flow through the temporal recurrence.

After quantization, `F.linear` becomes additions/subtractions only (no multiplies) — the hardware efficiency motivation.

### `AlphaInit` (nn.Parameter subclass)

Lazy initialization: `alpha` starts at 1.0, initializes from data statistics on first forward pass.

---

## How SpikeLinear Replaces Standard Linear

Standard `nn.Linear`: `out = input @ weight + bias` — one matrix multiply.

`SpikeLinear`:
1. Input is pre-split into T copies (by `BertEncoder.repeat(T, ...)`)
2. Each copy passes through leaky membrane integration + ternary quantization
3. Same linear projection applied to quantized outputs
4. T outputs averaged back in `BertEncoder`

Expressiveness is recovered via: T timesteps + leaky state + learned alpha scale.

---

## Differentiability Summary

| Component | Differentiable? |
|-----------|----------------|
| Ternary quantization (forward) | No — discrete `round()` |
| `grad_input` (backward) | STE surrogate — not true gradient |
| `grad_alpha` step size | Yes — real LSQ gradient |
| `weight` of linear layer | Yes — standard `F.linear` |
| Temporal recurrence | No — `.detach()` breaks BPTT |

STE is the simplest surrogate gradient: the surrogate is a flat 1 inside the clamp window.
Smooth alternatives (sigmoid, triangle, arctan surrogate) are used elsewhere in SNN literature
and generally give more informative gradients, but STE is sufficient here.

---

## Training Procedure

- **Weights:** randomly initialized (no pretrained BERT weights transferred)
- **Architecture config:** loaded from `bert-base-uncased` (hidden sizes, layer counts, etc.)
- **Distillation:** framework present in code but disabled (`teacher_model = None`)
- **Pretraining:** 500k steps on Wikipedia + BookCorpus, lr=2e-4, 8×A800 GPUs
- **Fine-tuning:** standard GLUE fine-tuning after pretraining

The model is trained entirely from scratch with standard MLM+NSP loss.

---

## Config

```python
weight_bits = 32   # full precision weights
input_bits = 2     # ternary activations {-1, 0, +1}
T = 4              # spiking timesteps
clip_val = 1.0
clip_init_val = 2.5
hidden_act = 'relu'
```
