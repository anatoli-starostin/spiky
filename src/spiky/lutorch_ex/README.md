# lutorch_ex — lookup-table layers for PyTorch

`lutorch_ex` is a small PyTorch library of **lookup-table layers**: drop-in replacements for a
feed-forward layer. Instead of multiplying its input by weight matrices, such a layer turns the input
into a handful of sign bits, uses those bits as an address into many small learnable tables, and adds
up the rows it reads. A token touches only a few rows of each table, so the layer can hold many
parameters while doing very little arithmetic per token.

The library separates the parts that never change from the part you experiment with. The geometry, the
addressing and the wrapper are fixed. The *read mathematics* — how an addressed row is weighted, and
what gradient training sees — is a swappable **cartridge**.

The detailed write-up, with the mathematics, the design choices and a seven-configuration
language-model sweep against a dense baseline, is the report
[`doc/lutorch_ex/lutorch_ex_report.pdf`](../../../doc/lutorch_ex/lutorch_ex_report.pdf).

**Contents:** [Install](#install) · [Quickstart](#quickstart) · [How a table reads](#how-a-table-reads) ·
[Geometry](#geometry-lutspec) · [Cartridges](#cartridges) · [Reference and fused cartridges](#reference-and-fused-cartridges) ·
[ProjectionMHL](#the-wrapper-projectionmhl) · [Regularisers](#regularisers) ·
[int8 and deployment](#int8-quantisation-and-deployment) · [Benchmarking](#benchmarking) · [Testing](#testing) ·
[Layout](#package-layout)

## Install

`lutorch_ex` is part of the `spiky` package; follow the [repository README](../../../README.md)
(`pip install -e .` from the repository root). The library itself is pure PyTorch and runs on CPU.

The `Fused…` classes run on hand-written CUDA kernels, compiled on first use with
`torch.utils.cpp_extension` (this needs `nvcc` and `ninja`). They have no fallback: a fused class whose
extension cannot be built raises `NativeUnavailableError` naming the cause, and a case its kernels do not
cover (a CPU input, an fp64 input, ...) raises naming the case. Construct the reference class
(`ManifestoHardLUT`, `ManifestoSoftLUT`, `ConfidenceLUT`) for those: same mathematics and state_dict, plain
PyTorch, runs anywhere.

## Quickstart

```python
import torch
from spiky.lutorch_ex import LUTSpec, ConfidenceLUT, ProjectionMHL

spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)  # 8 heads x 64 tables, 2**8 cells each
cart = ConfidenceLUT(spec, seed=1, read_top_n=2)                    # the read mathematics
ffn  = ProjectionMHL(cart, d_model=384)                             # 384 -> 384, like a feed-forward layer

x = torch.randn(4, 128, 384)                                        # (B, T, D) activations
y = ffn(x.reshape(-1, 384)).reshape(x.shape)                        # the wrapper takes [N, D]
print(y.shape)                                                      # torch.Size([4, 128, 384])
```

`ProjectionMHL` takes 2-D input, `[N, input_dim]`, so a sequence batch is flattened first. Any cartridge
can take `ConfidenceLUT`'s place.

A cartridge can also be used on its own. It maps `[B, h_in, d_in]` to `[B, h_out, d_out]`:

```python
from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT

spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8)
cart = ManifestoHardLUT(spec, seed=0)
z = torch.randn(5, 2, 8)                 # [B, h_in, d_in]
print(cart(z).shape)                     # torch.Size([5, 2, 8])  = [B, h_out, d_out]
```

Every runnable example in this README was executed against this package, on CPU and on CUDA.

## How a table reads

Each **table** reads a `d_in`-wide vector `z` and owns `nap` **probes**. A probe is either an anchor
pair `(a, b)`, giving the margin `u = z[a] - z[b]`, or a single anchor `a`, giving `u = z[a]`. Anchors
are coordinate indices: drawn once from the seed, stored as buffers and never trained.

1. **Address.** Each probe contributes one bit, `[u > 0]`, and the `nap` bits form a cell index
   `c = sum_j bit_j * 2**(nap-1-j)` in `[0, 2**nap)`: probe 0 gives the highest bit, the last probe the
   lowest. For example, with `nap = 3` and bits `(1, 0, 1)` from probes 0, 1, 2, `c = 4 + 0 + 1 = 5`.
   Each cell holds a learnable row of width `d_out`.
2. **Margins.** `|u_j|` says how far each bit is from flipping. The smallest, `u* = min_j |u_j|`, marks the
   least certain bit.
3. **Neighbour.** Flipping that bit gives the **neighbour** cell `c'`, the cell the input would land in
   under the smallest perturbation.

Every cartridge uses `c`, `c'` and the margins, and nothing else from the input.

Tables are grouped into **heads** of `tph` tables each. A head reads its own `d_in` slice of the input,
and its output is the sum of its tables' reads. The head outputs are then placed side by side, giving
`[B, h_out, d_out]`. With fan-in (`h_out == 1`) they are summed instead.

## Geometry: `LUTSpec`

`LUTSpec` is a frozen dataclass describing the geometry. It holds no weights.

| field | meaning |
| --- | --- |
| `h_in`, `h_out` | input and output heads |
| `tph` | tables per head (per group) |
| `nap` | probes per table; each table has `K = 2 ** nap` cells |
| `d_in`, `d_out` | per-head input and output width |
| `anchor_mode` | `"pairs"` (default) or `"single"` |

Allowed head patterns are `h_in == h_out` (per head), `h_in == 1` (fan-out: every group reads the one
shared input) and `h_out == 1` (fan-in: all groups are summed into one output). Anything else raises.

Derived properties: `n_groups = max(h_in, h_out)`, `n_tables = n_groups * tph`, `n_cells = 2 ** nap`,
`in_features = h_in * d_in`, `out_features = h_out * d_out`.

In `"pairs"` mode `d_in >= 2` is required. In `"single"` mode each bit tests one coordinate against zero,
and `d_in >= 1` suffices.

## Cartridges

### One read, three ways to weight it

Every cartridge computes, per head,

```text
y = sum over tables t of   a_t * W_t[c_t]  +  b_t * W_t[c'_t]
```

The cartridges differ only in the weights `a` and `b`, and in what training differentiates. A **soft**
read uses that formula as its value, so both rows learn. A **hard** read returns `W[c]` alone (`a = 1`,
`b = 0`) and uses the blend only as a **surrogate** for the input gradient, so the table update stays on
row `c`.

| class | gen | value (`a`, `b`) | training gradient | learned scalars | fused twin |
| --- | --- | --- | --- | --- | --- |
| `ManifestoHardLUT` | 1 / 2 | `1`, `0` | surrogate `1-w`, `w`; weight grad on row `c` only | `weight="softsign"`: `T_soft`, `T_sel` | `FusedManifestoHardLUT` |
| `ManifestoSoftLUT` | 1 / 2 | `1-w`, `w` | the value itself; both rows | `weight="softsign"`: `T_soft`, `T_sel` | `FusedManifestoSoftLUT` |
| `ConfidenceLUT` (`read_top_n=1`) | 3 | `s`, `0` | the value itself | `β`, `γ` | — |
| `ConfidenceLUT` (`read_top_n=2`) | 3 | `s(1-v)`, `s·v` | the value itself | `β`, `γ`, `τ` | — |
| `QuantisedConfidenceLUT` | 3 | as above, int8 rows and power-of-two weights | straight-through to the float values | `β`, `γ`, `τ` | — |

All classes are importable from `spiky.lutorch_ex`.

### Gen-1 and Gen-2 — Manifesto, `weight="rational" | "softsign"`

`ManifestoHardLUT` returns the hard read and sends the input gradient through the blend, onto the two anchors of the
deciding probe only. `ManifestoSoftLUT` makes the blend the value, at training and at evaluation. The two generations
differ only in the neighbour weight `w`, chosen by `weight=`:

- **`weight="rational"`** (Gen-1, the default): the fixed uncertainty `w = U = 0.5 / (1 + u*)`, in `(0, 0.5]`,
  exported as `rational_uncertainty`. No learnable scalars: the tables are the only parameters.
- **`weight="softsign"`** (Gen-2): a blend weight with two temperatures, learned per layer:

  ```text
  rho = u* / (T_soft + u*)          w = sigmoid(-2 * rho / T_sel)    in (0, 0.5]
  ```

  They are stored as `log_soft_score_temp` and `log_select_temp`, and exist only with `weight="softsign"`.

### Gen-3 — Confidence

Each table is weighted by a **confidence score** computed from all `nap` margins `m_j = |u_j|`:

```text
s = (sum_j m_j) * exp( gamma * sum_j log sigmoid(beta * m_j) )      v = sigmoid(-2 * u* / tau)
```

The second factor of `s` is the product of `sigmoid(beta * m_j) ** gamma`. It is near 1 for a margin far
from its boundary and `2 ** -gamma` for a margin of zero, so every uncertain bit discounts the table.

`β`, `γ` and `τ` are single scalars per layer, shared by all its tables, and stored in log form. They
are trainable: each is an `nn.Parameter` learned with the tables, unless frozen with
`learnable_score=False` (`β`, `γ`) or `read_tau_learnable=False` (`τ`). Because
the address is an integer, the input gradient reaches `z` only through `s` and, for `read_top_n=2`,
through `v`. Gen-3 has no hard form. Its training forward is `torch.compile`d on CUDA, which folds the
score's chain of small kernels.

### Constructor arguments

Every cartridge takes the `LUTSpec` positionally and everything else as keyword arguments. The base
arguments are shared by all of them:

| argument | default | meaning |
| --- | --- | --- |
| `seed` | `0` | anchors (group `g` seeded `seed + g`) and the table init |
| `weight_init_std` | `1e-3` | tables start as `Uniform[-std, +std]`, drawn per head |
| `table_dropout_rate` | `0.0` | whole-table dropout, see [Regularisers](#regularisers) |
| `device` | `None` | where to build parameters and buffers |
| `index_dtype` | `None` | dtype of the **stored** cell-index tensors, produced once in the shared addressing. `None` picks `torch.int32` when the table fits it, else `torch.int64`; an explicit `torch.int32` or `torch.int64` is honoured (see below) |

The cell index is stored as int32 by default. The index the reads use is the flat row index into the
`[n_groups·tph·2^nap, d_out]` table, up to `n_groups·tph·2^nap − 1` (262,143 at h=16, tph=64, nap=8).
That needs at least int32. torch's index ops (`embedding_bag`, `index_select`, `index_add_`, indexing) accept
only int32 and int64, so int16 and int8 are rejected.

- **The rule:** `index_dtype=None` (the default) uses int32 when `n_groups·tph·2^nap − 1` fits it and int64
  otherwise. An explicit `torch.int32` that doesn't fit raises; an explicit `torch.int64` is always accepted.
- **Only storage narrows.** Offset arithmetic, the bit-powers buffer and bag offsets stay int64, and the
  state_dict is unchanged.
- **Large eval calls widen automatically:** in auto mode, a call whose read could exceed 2^31 − 1 index
  entries uses int64, because PyTorch's CUDA `embedding_bag` forward fails above that with int32.

What int32 gives you:
- It halves the stored index tensors, including those saved for the backward.
- It narrows the radix-sort keys in `embedding_bag`'s weight gradient.
- The fused classes' kernels do not read the stored indices at all: they recompute the addressing from the
  input (int16 anchors).
- Training gradients are bit-identical to int64 under `torch.use_deterministic_algorithms(True)`.

Manifesto (`ManifestoHardLUT`, `ManifestoSoftLUT` and their fused twins):

| argument | default | meaning |
| --- | --- | --- |
| `weight` | `"rational"` | the neighbour weight: `"rational"` (Gen-1) or `"softsign"` (Gen-2) |
| `soft_score_temp` | `0.5` | `weight="softsign"` only: initial `T_soft` |
| `select_temp` | `0.5` | `weight="softsign"` only: initial `T_sel` |
| `learnable_temps` | `True` | `weight="softsign"` only: `False` stores both temperatures as frozen buffers |

The three temperature arguments raise with `weight="rational"`.

Confidence (`ConfidenceLUT`, `QuantisedConfidenceLUT`):

| argument | default | meaning |
| --- | --- | --- |
| `read_top_n` | `1` (`2` for `QuantisedConfidenceLUT`) | `1` = scored single cell, `2` = scored two-cell blend |
| `beta_init`, `gamma_init` | `2.0`, `1.0` | initial `β`, `γ` (must be `> 0`) |
| `read_tau_init` | `0.5` | initial `τ`; only valid with `read_top_n=2` (at `read_top_n=1` it raises `ValueError`: no blend, no `τ`) |
| `learnable_score` | `True` | `False` freezes `β`, `γ` as buffers |
| `read_tau_learnable` | `True` | `False` freezes `τ` as a buffer; only valid with `read_top_n=2` (raises at `read_top_n=1`) |

`ConfidenceLUT` has one read, `y = Σ_t s_t · W_t[c_t]` (n=1) or its two-cell blend (n=2), in train and eval, in one
of two forms of the same maths:

- **Compiled** (CUDA, the default): a fused gather × score → sum with its own backward (a re-gather of the same rows
  for the score gradient and an `index_add_` of the score-weighted rows for the table gradient), read straight from
  the fp32 table. Inductor fuses these into a few kernels. Fp32, h16 d48 tph64 nap8, 8,192 vectors, against
  `F.embedding_bag`: train step 6.55 → 3.48 ms (n=1) and 11.83 → 5.35 ms (n=2), eval 0.89 → 0.94 ms (n=1) and
  2.16 → 1.86 ms (n=2).
- **Eager** (CPU, or `LUTORCH_EX_NO_COMPILE=1`): one `F.embedding_bag` with the scores as per-sample weights.
  Uncompiled, the fused form would materialise its gathered rows (CPU n=1 train step 97.6 → 350.7 ms at 1,024
  vectors; CUDA eager n=2 train 20.3 ms / 1.92 GiB → 47.5 ms / 7.27 GiB at 8,192).
- **Agreement:** to fp32 re-association (≤ 7e-7), not bitwise. The compiled table gradient's `index_add_` uses atomic
  adds, so it is not bit-reproducible run to run, like the rest of the CUDA training step.

Quantised (`QuantisedConfidenceLUT`, in addition to the Confidence arguments): `quant_mode="p2_int8"`
(the only preset) and `quant_overrides=None`, see [below](#int8-quantisation-and-deployment). Its
`read_top_n` defaults to `2`. `read_top_n=1` trains (also as `FusedQuantisedConfidenceLUT`), evaluates and exports
too: the export records `meta["read_top_n"]`, and `DeployedQuantisedConfidenceLUT` reads one cell.

### PyTorch's embedding_bag backward size limit

This is a known upstream constraint. lutorch_ex documents it but doesn't enforce it.

- **Symptom:** a training step crashes with an opaque `CUDA error: an illegal memory access was encountered`.
  It is not a readable error, and stray reads could in principle return wrong gradients instead of crashing.
- **Cause:** PyTorch's CUDA `aten::_embedding_bag_dense_backward` (kernel `compute_grad_weight_bags`) indexes
  its launch threads in 32 bits. It fails when
  `(floor(numel/10) + min(numel, num_weights)) * 32 * ceil(d_out/32) > 2^31`, where:
  - `numel` is the index-tensor element count of one call;
  - `num_weights` is the table rows;
  - `d_out` is the row width.
- **Dtype-independent:** it reproduces identically for int32 and int64 indices.
- **Where it bites:** far below 2^31 entries. At h=16, tph=64, nap=8, d_out=48 (262,144 rows) the cliff is at
  ~2^28.3 entries (measured: 332,922,889 OK, 332,922,890 crashes; torch 2.9.1, RTX 5090). That's ~325k
  tokens per micro-batch at `read_top_n=1` and ~163k at `read_top_n=2`, against a live micro-batch of 32,768
  (~10× and ~5× headroom).
- **Exposed routes** (training calls to `F.embedding_bag` or `_embedding_bag_dense_backward`):
  ConfidenceLUT's eager read (CPU, `LUTORCH_EX_NO_COMPILE=1`), QuantisedConfidenceLUT and the Manifesto
  references' train reads.
- **Not exposed:** ConfidenceLUT's compiled read (its table gradient is an `index_add_`; verified correct on the
  compiled n=2 path past the cliff, to 819M entries where memory ran out first), the fused classes'
  kernels, and every eval / `no_grad` path.
- **Workaround:** reduce the micro-batch size and raise gradient accumulation.

## Reference and fused cartridges

Each class has exactly one implementation; there is no `backend=` argument.
The reference cartridges, `ManifestoHardLUT`, `ManifestoSoftLUT` (either weight) and `ConfidenceLUT`, are
readable PyTorch (compiled on CUDA) and run anywhere. They are the **oracle**: `tests/test_fused_vs_reference.py` is the required equivalence
harness: every fused class against its reference class, per dtype, on the value, the input
gradient, the table gradient and every learned scalar, with explicit tolerances justified in its docstring.
`tests/test_fused_manifesto_cuda.py` and `tests/test_fused_confidence.py` add the kernel-specific checks.

`FusedManifestoHardLUT` and `FusedManifestoSoftLUT` run one hand-written forward and one backward kernel
(`cartridges/csrc/fused_manifesto.cu`, the `lutorch_ex_fused_manifesto` extension). The kernels do the
addressing, the read summed over each head's tables, the table-gradient scatter (fp32 vector atomics) and the
margin gradient. Nothing per table goes through global memory, and the backward recomputes the addressing from
the input. They cover a CUDA input with an fp32 or bf16 table, nap ≤ 16, d_in ≤ 32767 and a d_out one CTA can
cover; anything else raises, naming the case and the reference class.
Launch knobs: `knobs=ManifestoCudaKnobs(...)` (from `cartridges._fused_manifesto_cuda`) or the
`LUTORCH_EX_MANI_CUDA_*` variables. The table
gradient is accumulated with atomics, so it is not bit-reproducible run to run (the forward is).

The Gen-3 `ConfidenceLUT` has the hand-written twin `FusedConfidenceLUT` (`cartridges/fused_confidence.py`,
the `lutorch_ex_fused_confidence` extension), with the same coverage and errors; neither inherits the other's
forward (both build on `ConfidenceBase`).

`QuantisedConfidenceLUT` (the reference: the torch definition in `cartridges/_pow2.py`, every case) has the twin
`FusedQuantisedConfidenceLUT`: the `lutorch_ex::p2_scalars` CUDA op (`csrc/pow2_int8_read.cu`, the
`lutorch_ex_pow2_int8_read` extension) computes the per-table power-of-two integers and straight-through cell weights,
and n=2 training runs one flat compiled read. The op is the two-cell read on CUDA fp32, so read_top_n=1, single
anchors, a CPU input or a non-fp32 input / table raise, naming the case; it is enabled only on validated
architectures (sm_120, sm_90), elsewhere it raises with that cause. Despite the file name the extension is not an
int8 read: int8 tables are always read with the torch definition (`_pow2.int_blend_read`).

**Precision.** Every Manifesto and confidence class, reference and fused, runs fp32 and accepts a bf16 / fp16
input and a bf16 table with one contract: addressing runs in fp32, so a lossy margin cannot flip a bit; reads and
reductions accumulate in fp32; and each result is rounded once (the output and the input gradient to the input dtype,
a parameter's gradient to its dtype). The reference classes do it by running their fp32 maths on fp32 views; the
result is bit-identical to the fp32 run rounded once -- measured at test sizes (tests/test_reference_low_precision.py), and on CUDA under torch.use_deterministic_algorithms (otherwise the fp32 run's own atomic scatters are not reproducible). The reference
classes also run fp64; the fused ones do not (and take no fp16 table). The quantised family's fused CUDA cartridges
-- `FusedQuantisedConfidenceLUT`, and `DeployedQuantisedConfidenceLUT` on CUDA -- take fp32 input and write fp32 output
only; any other input dtype raises, naming it (the deployed class's CPU path runs the torch definition, which also
takes fp64). `QuantisedConfidenceLUT` is the pure-PyTorch reference and is not part of that rule.

**Compilation and kernels.**

- The reference cartridges' forward (`cartridges/reference_base.py`) is `torch.compile`d on CUDA in eval,
  and in training too for Gen-3; CPU always runs eager. The fused classes bypass compile: they call their
  kernels directly.
- `cell_tv()` (every cartridge) is `torch.compile`d on CUDA; CPU runs it eager.
- CUDA extensions are built lazily, on first use. The build is cached in `TORCH_EXTENSIONS_DIR`, which
  defaults to `~/.cache/torch_extensions_lutorch_ex`.
- A fused class whose extension cannot be built raises `NativeUnavailableError` with the classified cause
  (setuptools / ninja / toolkit missing, a compile error, ...), every call. A case its kernels do not cover
  raises a `RuntimeError` naming the case and the reference class.

| environment variable | effect |
| --- | --- |
| `LUTORCH_EX_NO_COMPILE=1` | disable `torch.compile` everywhere |

## The wrapper: `ProjectionMHL`

```python
ProjectionMHL(cartridge, d_model=None, *, input_dim=None, output_dim=None,
              compress=True, decompress=True, bias=True, device=None,
              fp8_projections=(), compress_fp8_out_dtype=torch.float32, projection_dtype=None)
```

The wrapper puts a linear map on each side of the cartridge:
`[N, input_dim] → compress → [N, h_in, d_in] → cartridge → [N, h_out·d_out] → decompress → [N, output_dim]`.

- `d_model` is shorthand for `input_dim == output_dim`. On a rectangular wrapper, reading the `d_model`
  property raises an error.
- Both projections are `nn.Linear`. `compress` is initialised from `N(0, 0.02)`. `decompress` has its
  weight **and bias** initialised to zero, so a fresh layer outputs exactly zero for any input: a clean
  drop-in into a pretrained residual stream. The bias stays a trainable parameter.
- Either projection may be switched off (`compress=False` or `decompress=False`, which makes that side
  an identity), provided the widths already match. Switching off both is refused.
- `projection_dtype` (opt-in, default `None`: unchanged) runs the compress and decompress GEMMs in
  `torch.bfloat16` (or `torch.float32`). The weights stay fp32 `nn.Linear` parameters with the same
  state_dict. The compress output is cast back to fp32 before the cartridge, so addressing, scoring and
  the gather stay fp32; the output comes back in the input's dtype. This is the narrowed fp32 island. Do
  not call `.half()` or `.bfloat16()` on the whole wrapper, which would drag the cartridge into low
  precision. At h=16, tph=64, nap=8, bf16 compress flips ~0.6% of the cells (token × table) against an
  exact-fp32 compress. `fp8_projections` takes precedence per projection, so for example
  `projection_dtype=torch.bfloat16, fp8_projections=("decompress",)` runs compress in bf16 and
  decompress in fp8.
- `fp8_projections` (opt-in, default `()`) runs the GEMM of `"compress"` and/or `"decompress"` in fp8
  (`fp8.py`). The weights stay fp32 `nn.Linear` parameters with the same state_dict. Each call casts both
  operands to e4m3 with one amax scale per tensor, multiplies with `torch._scaled_mm` into an fp32
  accumulator, and runs the backward GEMMs with e5m2 gradients. The cartridge itself stays fp32. This
  needs a CUDA GPU that runs `torch._scaled_mm`; anything else raises at forward.
- **fp8 `"compress"` changes the model.** With fp8 compress operands, ~9% of the cells (token × table)
  read a different cell at h=16, tph=64, nap=8, measured against an exact-fp32 compress. These flips come
  from the e4m3 operands, so `compress_fp8_out_dtype` (fp32 or bf16 handed to the addressing) does not
  reduce them. fp8 `"decompress"` alone leaves the addresses bit-identical.

```python
ffn = ProjectionMHL(ConfidenceLUT(spec, seed=1), input_dim=384, output_dim=256)
print(ffn(torch.randn(10, 384)).shape)   # torch.Size([10, 256])
```

## Regularisers

Both regularisers live in the shared base class, so every cartridge has them. Both are off by default.

**Cell total variation.** A table's `2**nap` cells are the corners of an `nap`-dimensional cube.
`cartridge.cell_tv()` is the mean of `||W[c] - W[c']||²` over every pair of cells whose addresses differ
in one bit, averaged over all tables of the layer; `d_out` is summed, not averaged.
`cell_tv_penalty(module)` averages `cell_tv()` over every cartridge inside `module`. Scale it by `λ` and
add it to the loss. The prior it encodes: a one-bit change of address should change the read only a
little. On CUDA the penalty runs through `torch.compile`: at h=16, tph=64, nap=8 in fp32 its forward plus
backward takes 0.36 ms instead of 2.6 ms eager, and it holds no per-axis intermediates for the backward
(eager holds `nap/2` copies of the table's size).

**Table dropout.** `table_dropout_rate=p` drops whole tables in training. One Bernoulli draw per
(sample, head, table) keeps each table with probability `1 - p`, and the survivors are scaled by
`1/(1-p)`. Nothing is dropped at evaluation, or when gradients are disabled.

```python
from spiky.lutorch_ex import LUTSpec, FusedManifestoSoftLUT, ProjectionMHL, cell_tv_penalty

spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)
ffn = ProjectionMHL(FusedManifestoSoftLUT(spec, seed=1, weight="softsign", table_dropout_rate=0.2), d_model=384)
opt = torch.optim.AdamW(ffn.parameters(), lr=3e-4)

x = torch.randn(256, 384)
loss = ffn(x).pow(2).mean() + 10.0 * cell_tv_penalty(ffn)   # lambda = 10
loss.backward()
opt.step()
```

## int8 quantisation and deployment

`QuantisedConfidenceLUT(spec, quant_mode="p2_int8", …)` is Gen-3 trained so that the deployed read needs
only int8 rows, additions and bit shifts. Training is **quant-aware**: every forward already computes
with the rounded tables and weights. Both roundings are straight-through, so the gradient reaches the
float tables and `β`, `γ`, `τ`.

- **int8 cells.** `Ŵ = clamp(round(W / 2**e), -128, 127)`, with one power-of-two scale per group and
  output channel, `e = ceil(log2 max|W|) - 7`.
- **Power-of-two weights.** With `r = v/(1-v) = exp(-2u*/τ)`, the read `s(1-v)·(W[c] + r·W[c'])` rounds
  `r` to `2**-q` and `s(1-v)` to `2**k'`:
  - `q = clamp(floor(2u*/(τ ln 2) + ½), 0, 64)`;
  - `k' = round(log2 s - c_q)`, with `c_q = log2(1 + 2**-q)` for `q < 8`, else `0`.
  
  A table then adds `2**k'·Ŵ[c] + 2**(k'-q)·Ŵ[c']`.
- **Window and drop rule.** `k'` is kept in `[-3, 4]`. Below the window the table is skipped (it keeps
  its gradient); above it, `k'` is clamped. The second cell is dropped when `q > 3`.
- **Integer sum.** Shifts are counted in units of `2**-6`, so every shift is a left shift into an int32
  accumulator. The per-channel `2**(e-6)` is applied once, after the sum.

The `p2_int8` preset is `bits=8, offset=0, Q=3, L=8, kmax=4`, giving the window `[kmax - L + 1, kmax]`.
`quant_overrides` may change only `Q` (in `[0, 3]`), `L` (in `[1, 8]`) and `kmax`, and the window must
stay inside `[-3, 4]`.

**Deployment.** Export changes how the model is stored, not the model:

```python
import torch
import torch.nn as nn
from spiky.lutorch_ex import LUTSpec, QuantisedConfidenceLUT, ProjectionMHL, export_deployment, load_deployment

def make_model():
    spec = LUTSpec(h_in=4, h_out=4, tph=8, nap=6, d_in=12, d_out=12)
    return nn.Sequential(ProjectionMHL(QuantisedConfidenceLUT(spec, seed=1, read_top_n=2), d_model=48))

def make_skeleton():                     # same architecture, no memory allocated
    with torch.device("meta"):
        return make_model()

model = make_model().eval()              # ... trained ...
export_deployment(model, "model.lxq")    # int8 tables + anchors + scalars; scale folded into decompress
dep = load_deployment("model.lxq", make_skeleton, device="cpu").eval()
x = torch.randn(16, 48)
with torch.no_grad():
    print((dep(x) - model(x)).abs().max())   # agrees to quantisation round-off
```

What export and load do:

- `export_deployment` walks the model and asks every `ProjectionMHL`'s cartridge for a compact payload
  (`to_deployment()`). Other cartridges answer "dense", and their parameters go through the normal state
  dict.
- For the quantised cartridge, export stores the int8 tables and folds `2**(e-6)` statically into a copy
  of the layer's `decompress` weight. The fold is exact, because a power of two changes only a float's
  exponent.
- `load_deployment` builds the skeleton on the `meta` device and rebuilds each compacted layer as a
  `DeployedQuantisedConfidenceLUT`, through a format-tag registry. The float tables are never allocated.
- The file is a safetensors container when `safetensors` (and the `numpy` it needs to write) is installed,
  otherwise a torch file. `load_deployment` reads the format the file is in, so either loads anywhere;
  the torch file is loaded with `weights_only=True`.

In the check above (with trained-looking weights), the deployed output matched the float model exactly.

Scope today:

- `read_top_n=2` and `h_out == n_groups` only; fan-in would need a scale applied before routing.
- The wrapper must have a `decompress` layer to fold the scale into.
- Only int8 is implemented; other bit widths have no packer.
- The deployed int8 read is the torch definition (`_pow2.int_blend_read`, compiled on CUDA); no extension is
  involved.

## Benchmarking

`python -m spiky.lutorch_ex.benchmarks.bench_snapshot` measures a performance snapshot of every class (fp32 and
bf16, 8k and 32k vectors, one fresh process per cell) as JSON and Markdown.
`python -m spiky.lutorch_ex.benchmarks.bench_compare A.json B.json --out-dir DIR` compares two snapshots (interleaved
before / after bars and a ratio strip; matplotlib needed).

`spiky.lutorch_ex.benchmarks.bench` times any cartridge's `forward_eval`, `forward_train` and `backward` over a
grid of batch sizes and devices. It reports the median and minimum milliseconds and the throughput:

```python
from spiky.lutorch_ex import LUTSpec, ConfidenceLUT
from spiky.lutorch_ex.benchmarks.bench import benchmark, format_table

spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)
rows = benchmark(lambda: ConfidenceLUT(spec, seed=1, read_top_n=2), name="ConfidenceLUT n=2",
                 batch_sizes=[1024], devices=["cpu"], warmup=2, repeats=3)
print(format_table(rows))
```

How `devices` are interpreted:

- Devices are `"cpu"` or named GPU targets such as `"cuda:H100"` and `"cuda:RTX5090"`.
- A GPU target is timed only with `measure_cuda=True` and only if the present GPU matches the name.
  Otherwise its rows are `placeholder` rows with the reason.
- A failing cell, such as one that runs out of memory, becomes an `error` row rather than aborting the
  grid.
- `rows_to_json` and `rows_to_csv` export the rows.

As a script, `python -m spiky.lutorch_ex.benchmarks.bench [--warmup N] [--repeats N] [--measure-cuda] [--json PATH]
[--csv PATH]` runs a fixed demo: `ManifestoHardLUT` and `ManifestoSoftLUT` at a small geometry. For other
cartridges, call `benchmark()` as above.

## Testing

The tests are plain pytest. From the repository root:

```bash
python -m pytest src/spiky/lutorch_ex/tests -q
```

CUDA cases are parametrised in and run when a GPU is present. On a machine with an RTX 5090 the suite
collects 825 tests, and all pass with none skipped. That needs the `test` extras (`pytest`, `safetensors`,
`numpy`) installed.

**On a GPU machine, the Triton kernel cache must be writable.** The CUDA tests compile kernels with Triton,
which writes them to `TRITON_CACHE_DIR`, or to `~/.triton/cache` if that is unset. If CUDA is available and
that directory is read-only (a sandbox, a read-only home), the suite refuses to start and names the
directory. Point it at a writable directory:

```bash
export TRITON_CACHE_DIR=/tmp/triton-cache
```

On a CPU-only machine nothing is compiled with Triton, so the requirement does not apply: the suite skips the
check, says so in one line, and runs.

## Package layout

```text
lutorch_ex/
├── lut_spec.py          LUTSpec — the geometry
├── lut_base.py          MultiHeadLUT — the cartridge contract ([B, h_in, d_in] -> [B, h_out, d_out])
├── addressing.py        MSB-first sign-bit packing
├── anchors.py           canonical full-coverage anchor sampling (pairs and singles)
├── projection.py        ProjectionMHL
├── fp8.py               opt-in fp8 GEMMs for ProjectionMHL's projections (tensorwise e4m3/e5m2, fp32 master)
├── deploy.py            export_deployment / load_deployment
├── deploy_registry.py   format tag -> rebuilder registry
├── cartridges/
│   ├── manifesto_base.py            ManifestoLUT: tables, anchors, addressing, routing, cell TV, table dropout
│   ├── manifesto_hard.py, manifesto_soft.py           the Manifesto reference cartridges (both weights)
│   ├── fused_manifesto_hard.py, fused_manifesto_soft.py   fused twins
│   ├── confidence.py                Gen-3 ConfidenceLUT
│   ├── quantised_confidence.py      QuantisedConfidenceLUT (and the family's shared QuantisedScaffold)
│   ├── fused_quantised_confidence.py    FusedQuantisedConfidenceLUT
│   ├── deployed_quantised_confidence.py DeployedQuantisedConfidenceLUT
│   ├── _pow2.py                     the power-of-two quantisation maths (the torch definition)
│   ├── _fused_quantised_cuda.py     Fused/DeployedQuantisedConfidenceLUT glue for csrc/pow2_int8_read.cu
│   ├── fused_confidence.py          FusedConfidenceLUT
│   ├── _fused_confidence_cuda.py    FusedConfidenceLUT glue for csrc/fused_confidence.cu
│   ├── reference_base.py            ReferenceForward: the reference classes' (compiled) forward
│   ├── confidence_base.py           ConfidenceBase: ConfidenceLUT / FusedConfidenceLUT parameters and scaffolding
│   ├── _fused_ops.py                the reference reads (embedding_bag)
│   ├── _fused_read.py               ConfidenceLUT's compiled score-weighted read (one autograd Function)
│   ├── _fused_manifesto_cuda.py     FusedManifesto* glue for csrc/fused_manifesto.cu
│   ├── _cuda_ext.py                 the JIT build of every extension, the kernel launch knobs and launch fit
│   ├── _fallback.py                 classified "extension unavailable" errors (NativeUnavailableError)
│   ├── uncertainty.py               rational_uncertainty
│   └── csrc/                        CUDA / C++ kernel sources
├── benchmarks/
│   ├── bench.py                     the benchmark harness
│   ├── bench_snapshot.py            a performance snapshot of every class (one fresh process per cell)
│   ├── bench_compare.py             two snapshot JSONs -> the comparison figures (before / after bars, ratio strip)
│   └── bench_fused_confidence.py    FusedConfidenceLUT vs compiled ConfidenceLUT timings and launch-knob sweeps
└── tests/               pytest suite
```
