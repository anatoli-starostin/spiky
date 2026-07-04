# LUTorch Programming Guide

This guide describes how to use **LUTorch** (`spiky.lutorch`), the PyTorch-native LUT (Look-Up Table) stack. LUTorch provides differentiable layers that map inputs to outputs via anchor-pair comparisons and learned table weights. For architectural overview and module relationships, see [High-level architecture](highlevel_architecture.md).

## Table of Contents

1. [MultiHeadLut](#multiheadlut)
2. [ProjectionLUT](#projectionlut)
3. [Conv2DLut](#conv2dlut)
4. [LUTAttention](#lutattention)
5. [Helpers and utilities](#helpers-and-utilities)

---

## MultiHeadLut

`MultiHeadLut` is the main building block for vector inputs. It combines **AnchorPairsLookup** (input → lookup indices) and **LProjection** (indices → output vectors) in a multi-head layout.

### Constructor

```python
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode
import torch

lut = MultiHeadLut(
    input_dim=64,           # Input feature dimension
    n_heads=4,             # Number of heads
    n_outputs=16,           # Output dimension per head
    n_anchor_pairs=6,       # Anchor pairs per table (table size = 2**6)
    tables_per_head=1,      # Tables per head (default 1)
    n_buckets=1,           # Position buckets; >1 enables bucket_indices in forward (e.g. RPE)
    connected_anchors_mode=False,  # If True, anchor pairs form a connected graph
    anchor_candidates=None, # Optional tensor [tables_per_head, n_heads, max_anchors_per_table] with input indices
    cmp_eps=0.0,           # Epsilon for delta > cmp_eps comparison
    random_seed=42,        # For reproducible anchor sampling
    n_alternatives=1,      # Alternative indices per table (for smooth backward)
    smooth_mode=False,     # Smooth interpolation in projection
    device=None,           # Device for buffers
    uncertainty_mode=UncertaintyMode.INVERSE_L1,  # For smooth interpolation
    initial_weights_noise=0.001,  # Std of Gaussian noise on projection weights at init
)
```

- **`n_anchor_pairs`**: Each table has `2**n_anchor_pairs` entries. Anchor pairs are sampled at construction (balanced over `input_dim` or from `anchor_candidates`).
- **`n_buckets`**: When `n_buckets > 1`, `forward()` expects a `bucket_indices` tensor (e.g. from relative position encoding); the effective table size becomes `(2**n_anchor_pairs) * n_buckets`.
- **`smooth_mode`**: If `True`, projection blends the main lookup entry with alternative entries using an uncertainty function, giving differentiable behavior w.r.t. the continuous deltas.

### Forward

```python
# x: [B, input_dim]
out = lut(x)  # [B, n_heads, n_outputs]

# With positional buckets (e.g. for attention):
# bucket_indices: [B] int tensor, required when n_buckets > 1
out = lut(x, bucket_indices=bucket_indices)
```

### Minimal example

```python
import torch
from spiky.lutorch.multi_head_lut import MultiHeadLut

lut = MultiHeadLut(
    input_dim=100,
    n_heads=4,
    n_outputs=32,
    n_anchor_pairs=6,
    random_seed=42,
)
x = torch.randn(8, 100)
y = lut(x)  # [8, 4, 32]
```

---

## ProjectionLUT

`ProjectionLUT` applies a LUT to 2D spatial input `[B, H, W]` using an unfold-style patch grid. Each patch gets its own head; anchor candidates are restricted to that patch’s indices.

### UnfoldConfiguration

Patch layout is defined by `UnfoldConfiguration` (kernel size, stride, padding). Dilation is fixed to 1.

```python
from spiky.lutorch.multi_head_lut import UnfoldConfiguration

# 5x5 patches, stride 2, no padding
unfold = UnfoldConfiguration(H=28, W=28, kernel_size=5, stride=2, padding=0)
H_p, W_p = unfold.output_spatial_shape()  # Number of patches along H and W
```

### Constructor

```python
from spiky.lutorch.multi_head_lut import ProjectionLUT, UnfoldConfiguration

layer = ProjectionLUT(
    unfold_config=UnfoldConfiguration(H=28, W=28, kernel_size=5, stride=2, padding=0),
    n_outputs=64,
    n_anchor_pairs=4,
    tables_per_head=1,
    fold_config=None,  # Optional: scatter patch outputs to a larger grid
    device=None,
    **multi_head_lut_kwargs,  # e.g. random_seed, smooth_mode
)
```

- **`unfold_config`**: Must use `padding=0`. Defines patch grid over `(H, W)`.
- **`fold_config`**: If set, patch outputs are scatter-added onto an output grid of size `(fold_config.H, fold_config.W)`; forward then returns `[B, H_out, W_out]` instead of `[B, H_p, W_p, n_outputs]`.

### Forward

```python
# x: [B, H, W]
out = layer(x)  # [B, H_p, W_p, n_outputs] or [B, H_out, W_out] if fold_config set
```

---

## Conv2DLut

`Conv2DLut` is a 2D convolution-style layer: input `[B, C, H, W]` is unfolded into patches of size `C * kH * kW`, each patch is processed by a shared `MultiHeadLut`, and outputs are reshaped to `[B, out_channels, H_p, W_p]`.

### Constructor

```python
from spiky.lutorch.multi_head_lut import Conv2DLut, UnfoldConfiguration

layer = Conv2DLut(
    unfold_config=UnfoldConfiguration(H=32, W=32, kernel_size=3, stride=1, padding=1),
    in_channels=3,
    out_channels=64,
    n_anchor_pairs=5,
    n_heads=1,
    tables_per_head=1,
    device=None,
    **multi_head_lut_kwargs,
)
```

- **`out_channels`** must be divisible by **`n_heads`**; each head produces `out_channels // n_heads` channels.

### Forward

```python
# x: [B, C, H, W]
out = layer(x)  # [B, out_channels, H_p, W_p]
```

---

## LUTAttention

`LUTAttention` implements cross-attention over two sequences using a `MultiHeadLut` with **`n_outputs=1`** (one score per head per pair). It supports causal masking and optional relative positional encoding via buckets.

### Pair processing

For each query position `i` and key position `j`, the module forms one input vector per pair and runs the LUT to get scores. Two modes (via `PairProcessingConfig`):

- **LINEAR_COMBINATION** (default): `c1 * input1[i] + c2 * input2[j]` (default `c1=1.0`, `c2=-2.0`).
- **CONCATENATION**: `[input1[i], input2[j]]` (so `MultiHeadLut.input_dim` must be twice the per-sequence feature dim).

### Constructor

```python
from spiky.lutorch.lut_attention import LUTAttention, PairProcessingConfig, PairProcessingMode
from spiky.lutorch.multi_head_lut import MultiHeadLut

# MultiHeadLut must have n_outputs=1 and n_buckets == n_positional_buckets
multi_head_lut = MultiHeadLut(
    input_dim=32,   # Per-query/key dim in LINEAR_COMBINATION; 2*dim in CONCATENATION
    n_heads=4,
    n_outputs=1,
    n_anchor_pairs=6,
    n_buckets=8,   # For RPE; must match n_positional_buckets
    random_seed=42,
)

attn = LUTAttention(
    multi_head_lut=multi_head_lut,
    causal=True,             # Causal mask (lower triangular)
    n_positional_buckets=8,  # Must match multi_head_lut.n_buckets
    include_diagonal=True,   # Allow (q,q) in causal mask
    pair_config=None,        # Default: LINEAR_COMBINATION, c1=1.0, c2=-2.0
    do_sanity_checks=False,
    attention_temperature=1.0,
)
```

### Forward

```python
# input1, input2: [B, S, n_inputs]
scores = attn(input1, input2)  # [B, S, S, n_heads]
```

Scores are already softmax-normalized over the key dimension. For causal=True, only valid (q, k) pairs under the mask are computed; the rest are -inf before softmax.

### Relative positional encoding

When `n_positional_buckets > 1`, the LUT’s `n_buckets` is set to the same value. Internally, `LUTAttention` builds bucket indices from (q, k) distances (e.g. via `logarithmic_pe_buckets` and `rpe_matrix` from `lut_helpers`) and passes them as `bucket_indices` to `MultiHeadLut.forward()`.

---

## Helpers and utilities

### get_balanced_anchor_pairs

Used internally to sample anchor pairs with balanced coverage over input dimensions (or over per-table candidate indices):

```python
from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs
import torch

anchor_pairs_a, anchor_pairs_b = get_balanced_anchor_pairs(
    n_tables=10,
    n_anchor_pairs=6,
    input_dim=64,
    device=torch.device("cuda:0"),
    random_seed=42,
    connected_mode=False,  # True: pairs form connected graph
    anchor_candidates=None,  # Optional [n_tables, max_anchors] int tensor
)
# anchor_pairs_a, anchor_pairs_b: [n_tables, n_anchor_pairs] int64
```

### UncertaintyMode

Controls the uncertainty function used in smooth projection (blending main and alternative lookup entries):

```python
from spiky.lutorch.lut_helpers import UncertaintyMode

# INVERSE_L1: 0.5 / (1 + |delta|)
# INVERSE_QUADRATIC: 0.5 / (1 + delta^2)
lut = MultiHeadLut(..., smooth_mode=True, uncertainty_mode=UncertaintyMode.INVERSE_L1)
```

### Positional encoding for attention

For RPE in `LUTAttention` you can use the same bucket allocation as the module:

```python
from spiky.lutorch.lut_helpers import logarithmic_pe_buckets, rpe_matrix
import torch

device = torch.device("cuda:0")
seq_len = 32
num_buckets = 8

buckets = logarithmic_pe_buckets(num_buckets, seq_len, device)  # [seq_len]
rpe = rpe_matrix(buckets, seq_len, device)  # [seq_len, seq_len], RPE[i,j] = buckets[max(0,i-j)]
```

### Lower-level modules

If you need to wire lookup and projection yourself:

- **AnchorPairsLookup** (`spiky.lutorch.anchor_pairs_lookup`): `forward(x, return_alternatives=True)` returns lookup indices and optional alternative indices/deltas and gradient carriers for training.
- **LProjection** (`spiky.lutorch.l_projection`): `forward(lookup_indices, lookup_alt_indices=..., lookup_alt_deltas=..., ...)` returns `[B, n_lookup_tables, n_outputs]`.

`MultiHeadLut` is the standard way to use them together; use the raw classes only for custom pipelines.

---

## Environment and performance

- **SPIKY_LUTORCH_NO_COMPILE=1**: Disable `torch.compile` for LUTorch hot paths (e.g. for debugging or older PyTorch).
- **SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS=1**: Disable custom CUDA kernels and use PyTorch fallbacks.
- **SPIKY_LUTORCH_NO_LPROJECTION_CUSTOM_CUDA_KERNELS=1**: Disable custom CUDA only for `LProjection`.
- **SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK**: Threads per block for CUDA kernels (default 256; must be in [1, 1024]).

Install the optional `lutorch_cuda` extension (see main README) for best GPU performance; LUTorch works without it in pure PyTorch (and optional compile) mode.
