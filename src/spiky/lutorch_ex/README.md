# lutorch_ex — lookup-table layers for PyTorch

**Contents:** [Introduction](#introduction) · [Basic architecture](#basic-architecture) ·
[Usage example](#usage-example) · [Cartridges specification](#cartridges-specification) · [Regularisers](#regularisers)

## Introduction

`lutorch_ex` is a PyTorch library of **lookup-table layers**: replacements for a feed-forward layer. Instead of
multiplying its input by weight matrices, such a layer turns the input into a handful of sign bits, uses those bits as
an address into many small learnable tables, and adds up the rows it reads. A token touches only a few rows of each
table, so the layer can hold many parameters while doing very little arithmetic per token.

The geometry, the addressing and the wrapper are fixed. The *read mathematics* — how an addressed row is weighted, and
what gradient training sees — is a swappable **cartridge**. The mathematics of every cartridge, the design choices and
a language-model comparison against a dense baseline are in the report
[`doc/lutorch_ex/lutorch_ex_report.pdf`](../../../doc/lutorch_ex/lutorch_ex_report.pdf) (source:
`doc/lutorch_ex/lutorch_ex_report.tex`); this README refers to its sections as "report §N". The report calls a
cartridge a *lookup variant*: "cartridge" is the Python API's term only.

**Install.** `lutorch_ex` is part of the `spiky` package: `pip install -e .` from the repository root (see the
[repository README](../../../README.md)).

**What runs where.** Each cartridge is either a *reference* class or a *fused* class:

- The **reference classes** (`ManifestoHardLUT`, `ManifestoSoftLUT`, `ConfidenceLUT`, `QuantisedConfidenceLUT`) are
  PyTorch code and run on CPU and CUDA.
- The **fused classes** (`FusedManifestoHardLUT`, `FusedManifestoSoftLUT`, `FusedConfidenceLUT`,
  `FusedQuantisedConfidenceLUT`) run hand-written CUDA kernels, CUDA input only. Each has the same mathematics and
  the same `state_dict` keys as its reference class.
- **`DeployedQuantisedConfidenceLUT`**, the inference-only int8 form, runs a hand-written CUDA kernel for a CUDA
  input and the PyTorch definition for a CPU input.

The kernels live in three CUDA extensions (`cartridges/csrc/`), built with `torch.utils.cpp_extension` the first time
a class needs one. The build needs `setuptools`, `ninja` and the CUDA toolkit (`nvcc`). The `lutorch_ex_pow2_int8_read` extension, used by `FusedQuantisedConfidenceLUT` and
`DeployedQuantisedConfidenceLUT`, is enabled only on sm_120 and sm_90 GPUs.

**No fallback.** A fused class never falls back to PyTorch, and neither does `DeployedQuantisedConfidenceLUT` on a
CUDA input:

- An input or configuration its kernels do not cover (a CPU input, an unsupported dtype, `nap` or `d_out` beyond the
  kernels' limits, ...) raises a `RuntimeError` naming the case and, where one runs it, the reference class to use.
  `FusedConfidenceLUT` and `FusedQuantisedConfidenceLUT` check `d_in <= 32767` at construction (`ValueError`).
- For an input the kernels cover, an extension that cannot be built or loaded raises `NativeUnavailableError` (a
  `RuntimeError`, from `spiky.lutorch_ex.cartridges._fallback`) naming the cause (no CUDA device, an unvalidated
  architecture, `setuptools` / `ninja` / toolkit missing, a compile error, ...), on every call.

## Basic architecture

### Geometry: `LUTSpec`

`LUTSpec` is a frozen dataclass describing the geometry. It holds no parameters.

| field | meaning |
| --- | --- |
| `h_in`, `h_out` | input and output heads |
| `tph` | tables per group |
| `nap` | probes per table; each table has `K = 2 ** nap` cells |
| `d_in`, `d_out` | per-head input and output width |
| `anchor_mode` | `"pairs"` (default) or `"single"` |

Allowed head patterns are `h_in == h_out`, `h_in == 1` (fan-out) and `h_out == 1` (fan-in); anything else raises.
Derived properties: `n_groups = max(h_in, h_out)`, `n_tables = n_groups * tph`, `n_cells = 2 ** nap`,
`in_features = h_in * d_in`, `out_features = h_out * d_out`.

A trainable cartridge needs `d_in >= 2` and `nap <= d_in * (d_in - 1) / 2` in `"pairs"` mode, and `nap <= d_in`
in `"single"` mode.

### How a table reads

Tables come in `n_groups` groups of `tph`. Group `g` reads one `d_in`-wide input head: head `g` when `h_in ==
n_groups`, the single input head in fan-out. Each **table** owns `nap` **probes**. A probe is either an anchor pair
`(a, b)`, giving the margin `u = z[a] - z[b]`, or a single anchor `a`, giving `u = z[a]`. Anchors are coordinate
indices drawn once from the seed, stored as buffers and never trained.

1. **Address.** Each probe contributes one bit, `[u > 0]`, and the `nap` bits form a cell index
   `c = sum_j bit_j * 2**(nap-1-j)` in `[0, 2**nap)`: probe 0 gives the highest bit. Each cell holds a learnable row of
   width `d_out`.
2. **Deciding probe.** The probe with the smallest margin, `u* = min_j |u_j|`, marks the least certain bit.
3. **Neighbour.** Flipping that bit gives the **neighbour** cell `c'`.

A group's output is the sum of its tables' reads. The group outputs are the output heads, `[B, h_out, d_out]`; in
fan-in (`h_out == 1`) they are summed into the one output head.

### Cartridges

Every cartridge computes, per group, `y = sum over tables t of a_t * W_t[c_t] + b_t * W_t[c'_t]`, and differs from
the others only in the coefficients `a`, `b` and in what training differentiates (report §3). The classes and their
properties are in the [Cartridges specification](#cartridges-specification). All are importable from
`spiky.lutorch_ex`.

**Compilation.** On CUDA the reference classes' eval forward is `torch.compile`d. Their training forward is compiled
for `ConfidenceLUT` and `QuantisedConfidenceLUT`, and runs eager for `ManifestoHardLUT` and `ManifestoSoftLUT`. The
CPU runs eager. The fused classes and `DeployedQuantisedConfidenceLUT` are never compiled: they call their kernels.

**Precision.** The reference classes other than `QuantisedConfidenceLUT`, and the fused Manifesto and Confidence
classes, accept bfloat16 / float16 inputs and low-precision tables (dtypes per class in the specification table): the
addressing runs in float32, so a rounded margin cannot flip a bit; reads and reductions accumulate in float32; and each
result is rounded once (the output and the input gradient to the input's dtype, a parameter's gradient to its dtype).
The quantised classes take no low precision.

### Constructor arguments

Every cartridge except `DeployedQuantisedConfidenceLUT` takes the `LUTSpec` positionally and everything else as keyword
arguments. Shared by all of them:

| argument | default | meaning |
| --- | --- | --- |
| `seed` | `0` | anchors (group `g` seeded `seed + g`) and the table initialisation |
| `weight_init_std` | `1e-3` | tables start as `Uniform[-std, +std]` |
| `table_dropout_rate` | `0.0` | whole-table dropout, see [Regularisers](#regularisers) |
| `index_dtype` | `None` | dtype of the stored cell-index tensors (below) |

`index_dtype=None` stores the cell indices as `torch.int32` when the largest flat row index, `n_groups·tph·2^nap − 1`,
fits it, else as `torch.int64`; a call whose read could exceed `2^31 − 1` index entries then uses `torch.int64`. An
explicit `torch.int32` that does not fit raises; an explicit `torch.int64` is always accepted; other dtypes are
rejected. Only the stored index tensors change: the `state_dict` does not.

Manifesto (`ManifestoHardLUT`, `ManifestoSoftLUT`, `FusedManifestoHardLUT`, `FusedManifestoSoftLUT`):

| argument | default | meaning |
| --- | --- | --- |
| `uncertainty_func` | `"rational"` | the neighbour's share `U` of the blend: `"rational"` (fixed, no parameters) or `"softsign"` (two learned temperatures); report §3.2, §3.3 |
| `soft_score_temp` | `0.5` | `uncertainty_func="softsign"` only: initial `T_soft` |
| `select_temp` | `0.5` | `uncertainty_func="softsign"` only: initial `T_sel` |
| `learnable_temps` | `True` | `uncertainty_func="softsign"` only: `False` stores both temperatures as frozen buffers |
| `knobs` | `None` | fused classes only: a `ManifestoCudaKnobs` launch configuration (from `spiky.lutorch_ex.cartridges._cuda_ext`); `None` takes the [environment](#environment-variables) defaults |

The three temperature arguments raise with `uncertainty_func="rational"`.

Confidence (`ConfidenceLUT`, `FusedConfidenceLUT`, `QuantisedConfidenceLUT`, `FusedQuantisedConfidenceLUT`):

| argument | default | meaning |
| --- | --- | --- |
| `read_top_n` | required | `1` = one scored cell per table, `2` = a scored two-cell blend; omitting it raises `TypeError` |
| `beta_init`, `gamma_init` | `2.0`, `1.0` | initial `β`, `γ` of the confidence score (must be `> 0`) |
| `read_tau_init` | `0.5` | initial blend temperature `τ`; `read_top_n=2` only (raises `ValueError` at `read_top_n=1`) |
| `learnable_score` | `True` | `False` freezes `β`, `γ` as buffers |
| `read_tau_learnable` | `True` | `False` freezes `τ` as a buffer; `read_top_n=2` only (raises at `read_top_n=1`) |
| `knobs` | `None` | `FusedConfidenceLUT` only: a `CudaKnobs` launch configuration (from `spiky.lutorch_ex.cartridges._cuda_ext`) |
| `quant_mode` | `"p2_int8"` | quantised classes only: the quantisation preset (`"p2_int8"` is the only one; report §3.5) |
| `quant_overrides` | `None` | quantised classes only: may set `Q` (in `[0, 3]`), `L` (in `[1, 8]`) and `kmax`; the window `[kmax − L + 1, kmax]` must stay inside `[-3, 4]` |

### The wrapper: `ProjectionMHL`

```python
ProjectionMHL(cartridge, d_model=None, *, input_dim=None, output_dim=None,
              compress=True, decompress=True, bias=True, device=None,
              fp8_projections=(), compress_fp8_out_dtype=torch.float32, projection_dtype=None)
```

The wrapper puts a linear map on each side of the cartridge (report §2):
`[N, input_dim] → compress → [N, h_in, d_in] → cartridge → [N, h_out·d_out] → decompress → [N, output_dim]`.
It takes 2-D input, so a sequence batch is flattened first.

- `d_model` sets `input_dim == output_dim`. On a rectangular wrapper, reading the `d_model` property raises a
  `TypeError` that says to use `input_dim` / `output_dim`.
- Both projections are `nn.Linear`. `compress` is initialised from `N(0, 0.02)`. `decompress` has its weight **and
  bias** initialised to zero, so a fresh layer outputs exactly zero; the bias is a trainable parameter.
- Either projection may be switched off (`compress=False` or `decompress=False` makes that side an identity),
  provided the widths match. Switching off both is refused.
- `projection_dtype` (default `None`: unchanged) runs the compress and decompress GEMMs in `torch.bfloat16` (or
  `torch.float32`). The weights stay float32 parameters with the same `state_dict`; the compress output is cast to
  float32 before the cartridge; the output comes back in the input's dtype.
- `fp8_projections` (default `()`) runs the GEMM of `"compress"` and/or `"decompress"` in fp8 (`fp8.py`): both
  operands cast to e4m3 with one amax scale per tensor, `torch._scaled_mm` into a float32 accumulator, e5m2 gradients
  in the backward. The weights stay float32 parameters with the same `state_dict`. It needs a CUDA GPU that runs
  `torch._scaled_mm`; anything else raises at forward. It takes precedence over `projection_dtype` per projection.
  An fp8 `"compress"` changes which cell some tables read; an fp8 `"decompress"` alone leaves the addresses
  bit-identical.

### Deployment

`export_deployment(model, path)` writes a model whose `ProjectionMHL` layers wrap a `QuantisedConfidenceLUT` (or
`FusedQuantisedConfidenceLUT`) in a compact form, and `load_deployment(path, make_skeleton, device=None)` reads it back
(report §3.5):

- Export asks every `ProjectionMHL`'s cartridge for a compact payload. A quantised cartridge gives its int8 tables,
  anchors and scalars, and its per-channel power-of-two output scale is folded into a copy of the layer's `decompress`
  weight (exact: a power of two changes only a float's exponent). Every other parameter and buffer of the model goes
  through the normal `state_dict`.
- Load calls `make_skeleton()`, which must build the same architecture on the `meta` device, and rebuilds each
  compacted layer as a `DeployedQuantisedConfidenceLUT` through a format-tag registry; the float tables are never
  allocated.
- The file is a safetensors container when `safetensors` and `numpy` are installed, otherwise a torch file. Reading a
  safetensors file needs `safetensors`; a torch file is loaded with `weights_only=True`.

Limits:

- `read_top_n` 1 and 2 are both supported: the export records the read width and the deployed class reads that many
  cells per table.
- `h_out == n_groups` only: exporting a fan-in cartridge (`h_out == 1 < n_groups`) raises `NotImplementedError`.
- The wrapper needs a `decompress` layer to fold the scale into; without one, export raises `ValueError`.
- On a CUDA input the int8 read is the `lutorch_ex_pow2_int8_read` extension's `deploy_fwd` kernel: it needs `nvcc`, an
  sm_120 or sm_90 GPU, a float32 input, `nap <= 8` and `d_in <= 32767`. On a CPU input it is the PyTorch definition
  (`_pow2.int_single_read` at `read_top_n=1`, `_pow2.int_blend_read` at `read_top_n=2`), float32 or float64. Neither
  is compiled.
- The int8 format is the only one (`quant_mode="p2_int8"`).

### Environment variables

| variable | default | effect |
| --- | --- | --- |
| `LUTORCH_EX_NO_COMPILE` | unset | `1` disables `torch.compile` (the reference forwards and `cell_tv()`) |
| `LUTORCH_EX_MANI_CUDA_FWD_THREADS`, `_BWD_THREADS`, `_ROWS_PER_CTA`, `_VEC`, `_VEC_BF16`, `_VEC_ATOMICS` | `64`, `64`, `4`, `4`, `4`, `1` | the fused Manifesto kernels' launch configuration, for a `knobs` field left at `None` |
| `LUTORCH_EX_CONF_CUDA_FWD_THREADS`, `_BWD_THREADS`, `_ROWS_PER_CTA`, `_VEC`, `_VEC_BF16`, `_VEC_ATOMICS` | `64`, `64`, `4`, `4`, `4`, `1` | the same for `FusedConfidenceLUT` |
| `TORCH_EXTENSIONS_DIR` | `~/.cache/torch_extensions_lutorch_ex` | where the CUDA extensions are built and cached (set to the default when unset) |
| `SPIKY_CXX_STD` | `c++20` | the C++ standard the extensions are compiled with |

### Tests and benchmarks

The tests are pytest, run from the repository root:

```bash
python -m pytest src/spiky/lutorch_ex/tests -q
```

With pytest, safetensors and numpy installed, on an RTX 5090 (sm_120, torch 2.9.1+cu130) the suite gives 1689 passed,
0 skipped; with CUDA hidden (`CUDA_VISIBLE_DEVICES=`) it gives 450 passed, 1087 skipped. When CUDA is available, the
suite needs a writable Triton cache (`TRITON_CACHE_DIR`, else `~/.triton/cache`) and refuses to start, naming the
directory, if it is read-only; without CUDA it skips the check and says so.

`benchmarks/`:

- `python -m spiky.lutorch_ex.benchmarks.bench_snapshot --out snapshot.json --md snapshot.md` measures every class on
  a CUDA GPU (float32 and, where a class accepts it, bfloat16; four geometries by default; one fresh process per
  cell).
- `python -m spiky.lutorch_ex.benchmarks.bench_compare before.json after.json --out-dir DIR` compares two snapshots of
  one machine; `python -m spiky.lutorch_ex.benchmarks.bench_devices first.json second.json --out-dir DIR` compares two
  devices.
- `spiky.lutorch_ex.benchmarks.bench` times any cartridge's eval forward, train forward and backward over batch sizes
  and devices (`benchmark()`, `format_table()`, `rows_to_json()`, `rows_to_csv()`).
- `benchmarks/results_2026-10-15/` holds the H100 and RTX 5090 snapshots the
  [specification table](#cartridges-specification) takes its numbers from.

### Package layout

```text
lutorch_ex/
├── README.md                                   this file
├── __init__.py                                 the public API
├── addressing.py                               MSB-first sign-bit packing
├── anchors.py                                  anchor sampling (pairs and singles), full coverage per group
├── deploy.py                                   export_deployment / load_deployment
├── deploy_registry.py                          format tag -> rebuilder registry
├── fp8.py                                      opt-in fp8 GEMMs for ProjectionMHL's projections
├── lut_base.py                                 MultiHeadLUT: the cartridge contract ([B, h_in, d_in] -> [B, h_out, d_out])
├── lut_spec.py                                 LUTSpec: the geometry
├── projection.py                               ProjectionMHL
├── pyproject.toml                              standalone packaging of lutorch_ex (inert inside the spiky repository)
├── benchmarks/
│   ├── __init__.py
│   ├── bench.py                                timing harness for any cartridge (API and CLI)
│   ├── bench_compare.py                        two snapshots of one machine -> comparison figures
│   ├── bench_devices.py                        two snapshots of two devices -> comparison figures
│   ├── bench_snapshot.py                       performance snapshot of every class, one fresh process per cell
│   └── results_2026-10-15/                     H100 and RTX 5090 records, comparison figures (11 files)
├── cartridges/
│   ├── __init__.py                             the cartridge exports
│   ├── _cuda_ext.py                            the JIT build of every extension, the kernel launch knobs
│   ├── _dtypes.py                              the dtype tuples shared by the cartridges
│   ├── _fallback.py                            NativeUnavailableError and its classified causes
│   ├── _fused_confidence_cuda.py               FusedConfidenceLUT glue for csrc/fused_confidence.cu
│   ├── _fused_manifesto_cuda.py                FusedManifesto* glue for csrc/fused_manifesto.cu
│   ├── _fused_ops.py                           the reference reads (embedding_bag)
│   ├── _fused_quantised_cuda.py                Fused/DeployedQuantisedConfidenceLUT glue for csrc/pow2_int8_read.cu
│   ├── _fused_read.py                          ConfidenceLUT's compiled read (one autograd Function)
│   ├── _pow2.py                                the power-of-two int8 maths (the torch definition)
│   ├── confidence.py                           ConfidenceLUT
│   ├── confidence_base.py                      ConfidenceBase: the confidence family's parameters
│   ├── deployed_quantised_confidence.py        DeployedQuantisedConfidenceLUT
│   ├── fused_confidence.py                     FusedConfidenceLUT
│   ├── fused_manifesto_hard.py                 FusedManifestoHardLUT
│   ├── fused_manifesto_soft.py                 FusedManifestoSoftLUT
│   ├── fused_quantised_confidence.py           FusedQuantisedConfidenceLUT
│   ├── manifesto_base.py                       ManifestoLUT: tables, anchors, addressing, routing, cell TV, table dropout
│   ├── manifesto_hard.py                       ManifestoHardLUT
│   ├── manifesto_soft.py                       ManifestoSoftLUT
│   ├── quantised_confidence.py                 QuantisedConfidenceLUT (and the family's QuantisedScaffold)
│   ├── reference_base.py                       ReferenceForward: the reference classes' forward (compile, precision)
│   ├── uncertainty.py                          rational_uncertainty
│   └── csrc/
│       ├── fused_confidence.cu                 FusedConfidenceLUT forward and backward kernels
│       ├── fused_manifesto.cu                  FusedManifesto* forward and backward kernels
│       ├── pow2_int8_read.cu                   FusedQuantisedConfidenceLUT and DeployedQuantisedConfidenceLUT kernels
│       └── pow2_scalars.cuh                    the per-table power-of-two integers (device code)
└── tests/                                      pytest suite (36 files)
```

## Usage example

A forward pass, on CPU:

```python
import torch
from spiky.lutorch_ex import LUTSpec, ConfidenceLUT, ProjectionMHL

spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)    # 8 heads x 64 tables, 2**8 cells each
ffn = ProjectionMHL(ConfidenceLUT(spec, seed=1, read_top_n=2), d_model=384)

x = torch.randn(4, 128, 384)                    # (B, T, D) activations
y = ffn(x.reshape(-1, 384)).reshape(x.shape)    # the wrapper takes [N, D]
print(y.shape)                                  # torch.Size([4, 128, 384])
```

A cartridge can also be used on its own; it maps `[B, h_in, d_in]` to `[B, h_out, d_out]`:

```python
import torch
from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT

cart = ManifestoHardLUT(LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8), seed=0)
print(cart(torch.randn(5, 2, 8)).shape)         # torch.Size([5, 2, 8])
```

A training step with a fused cartridge and both [regularisers](#regularisers) (needs a CUDA GPU and the CUDA toolkit):

```python
import torch
from spiky.lutorch_ex import LUTSpec, FusedManifestoSoftLUT, ProjectionMHL, cell_tv_penalty

spec = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)
cart = FusedManifestoSoftLUT(spec, seed=1, uncertainty_func="softsign", table_dropout_rate=0.2)
ffn = ProjectionMHL(cart, d_model=384).cuda()
opt = torch.optim.AdamW(ffn.parameters(), lr=3e-4)

x = torch.randn(256, 384, device="cuda")
loss = ffn(x).pow(2).mean() + 10.0 * cell_tv_penalty(ffn)   # cell-TV weight 10
loss.backward()
opt.step()
print(loss.shape)                               # torch.Size([])
```

Export and load for deployment, on CPU:

```python
import torch
import torch.nn as nn
from spiky.lutorch_ex import LUTSpec, QuantisedConfidenceLUT, ProjectionMHL, export_deployment, load_deployment

def make_model():
    spec = LUTSpec(h_in=4, h_out=4, tph=8, nap=6, d_in=12, d_out=12)
    return nn.Sequential(ProjectionMHL(QuantisedConfidenceLUT(spec, seed=1, read_top_n=2), d_model=48))

def make_skeleton():                            # same architecture, no memory allocated
    with torch.device("meta"):
        return make_model()

torch.manual_seed(0)
model = make_model().eval()
with torch.no_grad():                           # stand-in for training: move every parameter off its initial value
    for p in model.parameters():
        p.add_(0.05 * torch.randn_like(p))
export_deployment(model, "model.lxq")
dep = load_deployment("model.lxq", make_skeleton, device="cpu").eval()

x = torch.randn(16, 48)
with torch.no_grad():
    y, y_dep = model(x), dep(x)
print(y.abs().max() > 0, (y_dep - y).abs().max())   # tensor(True) tensor(0.)
```

Measured on this example (`LUTSpec(h_in=4, h_out=4, tph=8, nap=6, d_in=12, d_out=12)`, 16 inputs, outputs of
magnitude about 0.2): on CPU the deployed output equals the float model's exactly, at `read_top_n=1` and `2`; with
the float model, the deployment and the input on CUDA (`load_deployment(..., device="cuda")`), the largest
difference is 7.45e-09 at both widths, on an RTX 5090.

## Cartridges specification

The numbers are from `benchmarks/results_2026-10-15/h100.json`, measured with `bench_snapshot` on an NVIDIA H100 80GB
HBM3 (sm_90, torch 2.9.1+cu130), in this configuration:

- **Geometry:** `LUTSpec(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)` (the record's `champion` geometry),
  **8,192 input vectors** (`x` of shape `[8192, 16, 48]`), the cartridge alone (no `ProjectionMHL`).
- **Defaults:** float32 input and table, `seed=1`, every other argument at its default (`uncertainty_func="rational"`,
  `table_dropout_rate=0.0`, `quant_mode="p2_int8"`). `read_top_n` has no default; both widths are given as
  *n=1 · n=2*. `DeployedQuantisedConfidenceLUT` is built from a `QuantisedConfidenceLUT` export at that read width.
- **eval ms:** the forward in eval mode under `no_grad`. **train ms:** forward and backward in train mode. Each is the
  median wall-clock time of 100 timed steps (after 20 warm-up steps, L2 cache flushed before each), the median over 3
  rounds, averaged over the record's 2 passes.
- **train peak MiB:** the train step's peak `torch.cuda.max_memory_allocated` above what is allocated before it
  (inputs, parameters and their gradients excluded).

The record also holds the same cells at 32,768 vectors (its `large` geometry); see `h100.md` next to it.

| name | short description | python class | input dtypes | output dtype | table dtypes | default table dtype | custom CUDA kernels | eval ms (H100) | train ms (H100) | train peak MiB (H100) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Manifesto, hard read | returns the addressed row; trains the input through the blend with the neighbour, the table on the addressed row only (report §3.1–§3.3) | `ManifestoHardLUT` | fp32, fp64, bf16, fp16 | input's | fp32, fp64, bf16, fp16 | fp32 | no | 1.174 | 19.551 † | 1304.0 |
| Manifesto, soft read | returns the blend of the addressed row and its neighbour (report §3.1–§3.3) | `ManifestoSoftLUT` | fp32, fp64, bf16, fp16 | input's | fp32, fp64, bf16, fp16 | fp32 | no | 2.082 | 19.809 † | 1304.0 |
| Manifesto, hard read, fused | `ManifestoHardLUT`'s mathematics in CUDA kernels (the report does not describe the kernels) | `FusedManifestoHardLUT` | fp32, bf16, fp16 (CUDA) | input's | fp32, bf16 | fp32 | yes | 0.447 | 2.072 | 96.0 |
| Manifesto, soft read, fused | `ManifestoSoftLUT`'s mathematics in CUDA kernels (the report does not describe the kernels) | `FusedManifestoSoftLUT` | fp32, bf16, fp16 (CUDA) | input's | fp32, bf16 | fp32 | yes | 0.650 | 2.380 | 96.0 |
| Confidence | each table weighted by a confidence score of its margins; one cell (n=1) or a two-cell blend (n=2) (report §3.4) | `ConfidenceLUT` | fp32, fp64, bf16, fp16 | input's | fp32, fp64, bf16, fp16 | fp32 | no | 1.501 · 1.751 | 4.478 · 6.697 | 728.0 · 840.0 |
| Confidence, fused | `ConfidenceLUT`'s mathematics in CUDA kernels (the report does not describe the kernels) | `FusedConfidenceLUT` | fp32, bf16, fp16 (CUDA) | input's | fp32, bf16 | fp32 | yes | 0.402 · 0.722 | 1.987 · 3.103 | 96.4 · 96.4 |
| Confidence, int8 | `ConfidenceLUT` trained quant-aware for an int8 read: int8 rows, power-of-two weights (report §3.5) | `QuantisedConfidenceLUT` | fp32, fp64 | input's | fp32, fp64 | fp32 | no | 2.948 · 4.484 | 11.740 · 19.123 | 568.8 · 1142.4 |
| Confidence, int8, fused | `QuantisedConfidenceLUT`'s mathematics in CUDA kernels (the report does not describe the kernels) | `FusedQuantisedConfidenceLUT` | fp32 (CUDA) | fp32 | fp32 | fp32 | yes | 0.688 · 0.952 | 1.915 · 2.744 | 128.4 · 128.4 |
| Confidence, int8, deployed | inference-only int8 read built from an export; read width from the export (report §3.5) | `DeployedQuantisedConfidenceLUT` | CUDA: fp32; CPU: fp32, fp64 | CUDA: fp32; CPU: input's | int8 (from the export) | int8 | yes (CUDA input); no (CPU input) | 0.353 · 0.476 | n/a | n/a |

- Input and table dtypes are independent: every listed combination runs. With float32 / float64 input and table the
  output is in the table's dtype (a float64 input with a float32 table gives float32, and the reverse float64). When the
  input or the table is bfloat16 / float16 the cartridge computes in float32 and rounds the output once to the input's
  dtype (see Precision above).
- † `ManifestoHardLUT` and `ManifestoSoftLUT` train eagerly (their training forward is not compiled; their eval forward
  is). Their train numbers are the uncompiled PyTorch step.
- n/a: `DeployedQuantisedConfidenceLUT` is inference-only; it has no train step.
- The non-default `uncertainty_func="softsign"`, in the same configuration:

  | python class | eval ms (H100) | train ms (H100) | train peak MiB (H100) |
  | --- | --- | --- | --- |
  | `ManifestoHardLUT` | 1.173 | 19.888 † | 1304.0 |
  | `ManifestoSoftLUT` | 2.128 | 20.175 † | 1359.0 |
  | `FusedManifestoHardLUT` | 0.619 | 3.012 | 96.0 |
  | `FusedManifestoSoftLUT` | 0.706 | 3.066 | 96.0 |

## Regularisers

Both regularisers belong to every cartridge that has a trainable table — every class except
`DeployedQuantisedConfidenceLUT` — and both are off by default (report §4).

**Cell total variation.** A table's `2**nap` cells are the corners of an `nap`-dimensional cube.
`cartridge.cell_tv()` is the mean of `||W[c] - W[c']||²` over every pair of cells whose addresses differ in one bit,
averaged over all tables of the layer (`d_out` is summed, not averaged). `cell_tv_penalty(module)` averages
`cell_tv()` over every cartridge with a trainable table inside `module` (a `DeployedQuantisedConfidenceLUT` is not
one, and its `cell_tv()` raises `TypeError`); scale it and add it to the loss, as in the training step above. On CUDA
`cell_tv()` is `torch.compile`d; the CPU runs it eager.

**Table dropout.** `table_dropout_rate=p` drops whole tables in training. One Bernoulli draw per (sample, group,
table) keeps each table with probability `1 - p`, and the survivors are scaled by `1/(1-p)`. Nothing is dropped at
evaluation, or when gradients are disabled.
