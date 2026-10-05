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

On CUDA, some cartridges have native kernels that are compiled on first use with
`torch.utils.cpp_extension`. This needs `nvcc` and `ninja`. Without them, or on CPU, every cartridge
falls back to its pure-PyTorch path, which gives the same results.

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

1. **Address.** Each probe contributes one bit, `[u > cmp_eps]`. The `nap` bits, probe 0 being the most
   significant, form a cell index `c` in `[0, 2**nap)`. Each cell holds a learnable row of width `d_out`.
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
| `ManifestoHardLUT` | 1 | `1`, `0` | surrogate `1-U`, `U`; weight grad on row `c` only | — | `FusedManifestoHardLUT` |
| `ManifestoSoftLUT` | 1 | `1-U`, `U` | the value itself; both rows | — | `FusedManifestoSoftLUT` |
| `SoftSignHardLUT` | 2 | `1`, `0` | surrogate `1-w`, `w`; weight grad on row `c` only | `T_soft`, `T_sel` | `FusedSoftSignHardLUT` |
| `SoftSignSmoothLUT` | 2 | `1-w`, `w` | the value itself; both rows | `T_soft`, `T_sel` | `FusedSoftSignSmoothLUT` |
| `ConfidenceLUT` (`read_top_n=1`) | 3 | `s`, `0` | the value itself | `β`, `γ` | — |
| `ConfidenceLUT` (`read_top_n=2`) | 3 | `s(1-v)`, `s·v` | the value itself | `β`, `γ`, `τ` | — |
| `QuantisedConfidenceLUT` | 3 | as above, int8 rows and power-of-two weights | straight-through to the float values | `β`, `γ`, `τ` | — |

All classes are importable from `spiky.lutorch_ex`.

### Gen-1 — Manifesto

The uncertainty `U = 0.5 / (1 + u*)`, in `(0, 0.5]`, is a fixed function, exported as
`rational_uncertainty`. `ManifestoHardLUT` returns the hard read and sends the input gradient through the
blend, onto the two anchors of the deciding probe only. `ManifestoSoftLUT` makes the blend the value, at
training and at evaluation. Neither has learnable scalars: the tables are the only parameters.

### Gen-2 — SoftSign

The fixed `U` becomes a blend weight with two temperatures, learned per layer:

```text
rho = u* / (T_soft + u*)          w = sigmoid(-2 * rho / T_sel)    in (0, 0.5]
```

They are stored as `log_soft_score_temp` and `log_select_temp`. `SoftSignHardLUT` and
`SoftSignSmoothLUT` relate to each other as Gen-1's hard and soft cartridges do.

### Gen-3 — Confidence

Each table is weighted by a **confidence score** computed from all `nap` margins `m_j = |u_j|`:

```text
s = (sum_j m_j) * exp( gamma * sum_j log sigmoid(beta * m_j) )      v = sigmoid(-2 * u* / tau)
```

The second factor of `s` is the product of `sigmoid(beta * m_j) ** gamma`. It is near 1 for a margin far
from its boundary and `2 ** -gamma` for a margin of zero, so every uncertain bit discounts the table.

`β`, `γ` and `τ` are single scalars per layer, shared by all its tables, and stored in log form. Because
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
| `cmp_eps` | `0.0` | the threshold `eps` in the bit test `u > eps` |
| `table_dropout_rate` | `0.0` | whole-table dropout, see [Regularisers](#regularisers) |
| `device` | `None` | where to build parameters and buffers |

SoftSign (`SoftSignHardLUT`, `SoftSignSmoothLUT` and their fused twins):

| argument | default | meaning |
| --- | --- | --- |
| `soft_score_temp` | `0.5` | initial `T_soft` |
| `select_temp` | `0.5` | initial `T_sel` |
| `learnable_temps` | `True` | `False` stores both temperatures as frozen buffers |

Confidence (`ConfidenceLUT`, `QuantisedConfidenceLUT`):

| argument | default | meaning |
| --- | --- | --- |
| `read_top_n` | `1` | `1` = scored single cell, `2` = scored two-cell blend |
| `beta_init`, `gamma_init` | `2.0`, `1.0` | initial `β`, `γ` (must be `> 0`) |
| `read_tau_init` | `0.5` | initial `τ` (used when `read_top_n=2`) |
| `learnable_score` | `True` | `False` freezes `β`, `γ` as buffers |
| `read_tau_learnable` | `True` | `False` freezes `τ` as a buffer |

Quantised (`QuantisedConfidenceLUT`, in addition to the Confidence arguments): `quant_mode="p2_int8"`
(the only preset) and `quant_overrides=None`, see [below](#int8-quantisation-and-deployment). Note that
`read_top_n` still defaults to `1` here, while deployment needs `2`.

Fused twins (`Fused…`): `backend="auto"`, see the next section.

## Reference and fused cartridges

The plain cartridges, `ManifestoHardLUT`, `ManifestoSoftLUT`, `SoftSignHardLUT` and `SoftSignSmoothLUT`,
are readable PyTorch. They are the **oracle**: `tests/test_fused_equivalence.py` and
`tests/test_fused_softsign.py` check that every fused backend gives the same values and gradients.

The `Fused…` twins have identical mathematics but dispatch each call to the fastest available
implementation. `backend="auto"` picks the path per call; any other value forces one:

| class | `backend` values | `auto` picks |
| --- | --- | --- |
| `FusedManifestoHardLUT` | `auto`, `pure_eval`, `tier1`, `native` | eval: compiled gather; train: `native` if available, else `tier1` |
| `FusedSoftSignHardLUT`, `FusedSoftSignSmoothLUT` | `auto`, `pure_eval`, `tier1`, `native` | as above |
| `FusedManifestoSoftLUT` | `auto`, `pure`, `tier1`, `native` | CUDA batch `>= 4096`: `tier1`; smaller eval: `pure`; smaller train: `native` if available |

- `tier1` is one `F.embedding_bag` over the addressed cells, and runs anywhere.
- `native` uses the `lutorch_ex_lprojection` CUDA extension.

The Gen-3 cartridges have no twin: their `embedding_bag` read with a compiled forward is already the
fast path.

**Precision.** The plain cartridges and Gen-3 run in fp32 or fp64, and raise a `TypeError` on
bf16/fp16. The four fused twins accept bf16/fp16: addressing runs in fp32, so a lossy margin cannot flip
a bit; reads and reductions accumulate in fp32; and the output is cast back once.

**Compilation and kernels.**

- Eval forwards are `torch.compile`d on CUDA; training forwards only for Gen-3. CPU always runs eager.
- Native extensions are built lazily, on first use. The build is cached in `TORCH_EXTENSIONS_DIR`, which
  defaults to `~/.cache/torch_extensions_lutorch_ex`.
- Without the extensions, every cartridge falls back to pure torch with identical results.

| environment variable | effect |
| --- | --- |
| `LUTORCH_EX_NO_COMPILE=1` | disable `torch.compile` everywhere |
| `LUTORCH_EX_NO_CUDA_EXT=1` | skip the `lprojection`, single-anchor and soft-sign surrogate extensions |
| `SPIKY_P2_CUDA_DISABLE=1` | skip the int8 power-of-two read extension |

## The wrapper: `ProjectionMHL`

```python
ProjectionMHL(cartridge, d_model=None, *, input_dim=None, output_dim=None,
              compress=True, decompress=True, bias=True, device=None)
```

The wrapper puts a linear map on each side of the cartridge:
`[N, input_dim] → compress → [N, h_in, d_in] → cartridge → [N, h_out·d_out] → decompress → [N, output_dim]`.

- `d_model` is shorthand for `input_dim == output_dim`. On a rectangular wrapper, reading the `d_model`
  property raises an error.
- Both projections are `nn.Linear`. `compress` is initialised from `N(0, 0.02)` and `decompress` to
  zero, so a fresh layer contributes nothing.
- Either projection may be switched off (`compress=False` or `decompress=False`, which makes that side
  an identity), provided the widths already match. Switching off both is refused.

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
little.

**Table dropout.** `table_dropout_rate=p` drops whole tables in training. One Bernoulli draw per
(sample, head, table) keeps each table with probability `1 - p`, and the survivors are scaled by
`1/(1-p)`. Nothing is dropped at evaluation, or when gradients are disabled.

```python
from spiky.lutorch_ex import LUTSpec, FusedSoftSignSmoothLUT, ProjectionMHL, cell_tv_penalty

spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)
ffn = ProjectionMHL(FusedSoftSignSmoothLUT(spec, seed=1, table_dropout_rate=0.2), d_model=384)
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
- The file is a safetensors container, or a `weights_only` torch file when `safetensors` is missing.

In the check above (with trained-looking weights), the deployed output matched the float model exactly.

Scope today:

- `read_top_n=2` and `h_out == n_groups` only; fan-in would need a scale applied before routing.
- The wrapper must have a `decompress` layer to fold the scale into.
- Only int8 is implemented; other bit widths have no packer.
- The int8 CUDA read is enabled only on validated architectures, sm_120 and sm_90. Elsewhere the
  pure-torch read is used.

## Benchmarking

`spiky.lutorch_ex.bench` times any cartridge's `forward_eval`, `forward_train` and `backward` over a
grid of batch sizes and devices. It reports the median and minimum milliseconds and the throughput:

```python
from spiky.lutorch_ex import LUTSpec, ConfidenceLUT
from spiky.lutorch_ex.bench import benchmark, format_table

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

As a script, `python -m spiky.lutorch_ex.bench [--warmup N] [--repeats N] [--measure-cuda] [--json PATH]
[--csv PATH]` runs a fixed demo: `ManifestoHardLUT` and `ManifestoSoftLUT` at a small geometry. For other
cartridges, call `benchmark()` as above.

## Testing

The tests are plain pytest. From the repository root:

```bash
python -m pytest src/spiky/lutorch_ex/tests -q
```

CUDA cases are parametrised in and run when a GPU is present. On a machine with an RTX 5090 the suite
collects 768 tests, and all pass.

## Package layout

```text
lutorch_ex/
├── lut_spec.py          LUTSpec — the geometry
├── lut_base.py          MultiHeadLUT — the cartridge contract ([B, h_in, d_in] -> [B, h_out, d_out])
├── addressing.py        MSB-first sign-bit packing
├── anchors.py           canonical full-coverage anchor sampling (pairs and singles)
├── projection.py        ProjectionMHL
├── deploy.py            export_deployment / load_deployment
├── deploy_registry.py   format tag -> rebuilder registry
├── bench.py             the benchmark harness
├── cartridges/
│   ├── manifesto_base.py            ManifestoLUT: tables, anchors, addressing, routing, cell TV, table dropout
│   ├── manifesto_hard.py, manifesto_soft.py           Gen-1 reference cartridges
│   ├── softsign_base.py, softsign_hard.py, softsign_smooth.py   Gen-2 reference cartridges
│   ├── fused_manifesto_hard.py, fused_manifesto_soft.py, fused_softsign.py   fused twins
│   ├── confidence.py                Gen-3 ConfidenceLUT
│   ├── quantised_confidence.py      QuantisedConfidenceLUT, DeployedQuantisedConfidenceLUT
│   ├── _pow2.py, _pow2_int8.py      power-of-two quantisation maths and its CUDA read
│   ├── _fused_ops.py, _native_ops.py, _native_softsign.py   tier-1 ops and native-extension glue
│   ├── uncertainty.py               rational_uncertainty
│   └── csrc/                        CUDA / C++ kernel sources
└── tests/               pytest suite
```
