"""
PermutationalLut — A higher-level LUT that operates in pure permutation space.

Each table monitors `input_nap` input pairs (via the underlying MultiHeadLut) and
predicts dominance scores for `output_nap` output pairs. Per-pair signed votes are
scattered directly into a per-output-dim accumulator — no full [N, N] matrix is
materialised. The result is a real-valued per-dim score whose RANKING defines
the output permutation.

Two pair-mapping modes:
  - 'aligned'   : output pairs are the same as input pairs.
                  Requires input_nap == output_nap and n_inputs == n_outputs.
  - 'scrambled' : output pairs are sampled independently with full coverage policy.
                  input_nap and output_nap may differ; n_inputs and n_outputs may differ.

Three soft modes for the per-pair vote:
  - 'sigmoid'  : sigmoid(x / T) - 0.5     ∈ (-0.5, +0.5)
  - 'rational' : 0.5 * x / (T + |x|)      ∈ (-0.5, +0.5)
  - 'ste'      : hard forward (sign(x)*0.5), soft backward through 'rational'

Forward output: [B, n_heads, n_outputs] (same shape as a regular MultiHeadLut output).
The output is naturally centred around 0 — no LayerNorm needed afterwards.
"""
import math
import os
from typing import Optional
import torch
import torch.nn as nn

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import (
    UncertaintyMode, AnchorSamplingPolicy, get_balanced_anchor_pairs,
)

_FP8_DTYPE = getattr(torch, 'float8_e4m3fn', None)
_FP8_AMAX = 448.0  # max representable value in e4m3fn


def _quantize_to_fp8(w: torch.Tensor) -> torch.Tensor:
    """Quantize fp32 tensor to fp8 precision with per-tensor dynamic scaling + STE.

    Scales weights to fill the fp8 range before casting, then scales back.
    This avoids small weights being rounded to zero.
    """
    if _FP8_DTYPE is None:
        return w
    amax = w.detach().abs().amax().clamp(min=1e-8)
    scale = _FP8_AMAX / amax
    fp8 = (w * scale).to(_FP8_DTYPE).to(w.dtype) / scale
    return w + (fp8 - w).detach()


_USE_PERMLUT_CUSTOM_CUDA = os.environ.get("SPIKY_PERMLUT_NO_CUSTOM_CUDA", "0") != "1"
_USE_PERMLUT_COMPILE = os.environ.get("SPIKY_PERMLUT_NO_COMPILE", "0") != "1"

_SOFT_MODE_INT = {'sigmoid': 0, 'rational': 1, 'ste': 2}


def _get_permlut_native():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


# ---- Eager (uncompiled) per-mode forward bodies ----

def _perm_forward_eager_sigmoid(raw, idx_a, idx_b, n_outputs, T):
    B, H, TP = raw.shape[0], idx_a.shape[0], idx_a.shape[1]
    raw4 = raw.view(B, H, raw.shape[1] // H, raw.shape[2])
    d = torch.sigmoid(raw4 / T) - 0.5
    src = d.reshape(B, H, TP)
    idx_a_e = idx_a.view(1, H, TP).expand(B, -1, -1)
    idx_b_e = idx_b.view(1, H, TP).expand(B, -1, -1)
    out = torch.zeros(B, H, n_outputs, device=raw.device, dtype=raw.dtype)
    out.scatter_add_(2, idx_a_e, src)
    out.scatter_add_(2, idx_b_e, -src)
    return out


def _perm_forward_eager_rational(raw, idx_a, idx_b, n_outputs, T):
    B, H, TP = raw.shape[0], idx_a.shape[0], idx_a.shape[1]
    raw4 = raw.view(B, H, raw.shape[1] // H, raw.shape[2])
    d = 0.5 * raw4 / (T + raw4.abs())
    src = d.reshape(B, H, TP)
    idx_a_e = idx_a.view(1, H, TP).expand(B, -1, -1)
    idx_b_e = idx_b.view(1, H, TP).expand(B, -1, -1)
    out = torch.zeros(B, H, n_outputs, device=raw.device, dtype=raw.dtype)
    out.scatter_add_(2, idx_a_e, src)
    out.scatter_add_(2, idx_b_e, -src)
    return out


def _perm_forward_eager_ste(raw, idx_a, idx_b, n_outputs, T):
    # STE forward is just the hard sign — no need for the soft surrogate
    # in the forward pass (autograd does not see inside an autograd Function's
    # forward, so the .detach() trick is a no-op here).
    B, H, TP = raw.shape[0], idx_a.shape[0], idx_a.shape[1]
    raw4 = raw.view(B, H, raw.shape[1] // H, raw.shape[2])
    d = torch.where(raw4 > 0, 0.5, -0.5).to(raw.dtype)
    src = d.reshape(B, H, TP)
    idx_a_e = idx_a.view(1, H, TP).expand(B, -1, -1)
    idx_b_e = idx_b.view(1, H, TP).expand(B, -1, -1)
    out = torch.zeros(B, H, n_outputs, device=raw.device, dtype=raw.dtype)
    out.scatter_add_(2, idx_a_e, src)
    out.scatter_add_(2, idx_b_e, -src)
    return out


# ---- Optionally compiled versions (cached after first call) ----
if _USE_PERMLUT_COMPILE:
    _perm_forward_compiled_sigmoid = torch.compile(_perm_forward_eager_sigmoid, dynamic=True)
    _perm_forward_compiled_rational = torch.compile(_perm_forward_eager_rational, dynamic=True)
    _perm_forward_compiled_ste = torch.compile(_perm_forward_eager_ste, dynamic=True)
else:
    _perm_forward_compiled_sigmoid = _perm_forward_eager_sigmoid
    _perm_forward_compiled_rational = _perm_forward_eager_rational
    _perm_forward_compiled_ste = _perm_forward_eager_ste


_PERM_FORWARD_DISPATCH = {
    0: _perm_forward_compiled_sigmoid,
    1: _perm_forward_compiled_rational,
    2: _perm_forward_compiled_ste,
}


class _PermLutDomFunction(torch.autograd.Function):
    """Dominance-path autograd: fused CUDA forward+backward.
    Forward uses the gather kernel (no atomics — one thread per output).
    Backward is slot-centric and reuses the per-slot pair_idx / sign arrays."""

    @staticmethod
    def forward(ctx, raw, pair_idx, sign, inv_idx, inv_sign, n_pairs, soft_mode_int, temperature):
        native = _get_permlut_native()
        raw_c = raw.contiguous()
        out = native.perm_lut_dom_gather_forward(
            raw_c, inv_idx, inv_sign, int(n_pairs),
            int(soft_mode_int), float(temperature),
        )
        ctx.save_for_backward(raw_c, pair_idx, sign)
        ctx.soft_mode_int = int(soft_mode_int)
        ctx.temperature = float(temperature)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        raw, pair_idx, sign = ctx.saved_tensors
        native = _get_permlut_native()
        grad_raw = native.perm_lut_dom_backward(
            grad_out.contiguous(), raw, pair_idx, sign,
            ctx.soft_mode_int, ctx.temperature,
        )
        return grad_raw, None, None, None, None, None, None, None


class _PermLutFunction(torch.autograd.Function):
    """Custom autograd: PyTorch forward + fused CUDA backward.

    Forward uses PyTorch's optimised scatter_add (already fast).
    Backward uses a fused CUDA kernel that avoids materialising the
    [B, H*T, P] soft-vote intermediate — saves ~5x on backward time.
    """

    @staticmethod
    def forward(ctx, raw, idx_a, idx_b, n_outputs, soft_mode_int, temperature):
        # Forward delegated to a compiled function (fuses soft_vote + scatter)
        fn = _PERM_FORWARD_DISPATCH[int(soft_mode_int)]
        out = fn(raw, idx_a, idx_b, int(n_outputs), float(temperature))
        ctx.save_for_backward(raw, idx_a, idx_b)
        ctx.soft_mode_int = int(soft_mode_int)
        ctx.temperature = float(temperature)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        raw, idx_a, idx_b = ctx.saved_tensors
        native = _get_permlut_native()
        grad_raw = native.perm_lut_backward(
            grad_out.contiguous(), raw, idx_a, idx_b,
            ctx.soft_mode_int, ctx.temperature,
        )
        return grad_raw, None, None, None, None, None


# ---- Matmul-path forward bodies (signed_vote + einsum), optionally compiled ----

def _perm_matmul_forward_eager_sigmoid(raw, proj_matrix, T):
    d = torch.sigmoid(raw / T) - 0.5
    B, H, Tdim, P = raw.shape
    d_flat = d.reshape(B, H, Tdim * P)
    return torch.einsum('bhk,hkn->bhn', d_flat, proj_matrix)


def _perm_matmul_forward_eager_rational(raw, proj_matrix, T):
    d = 0.5 * raw / (T + raw.abs())
    B, H, Tdim, P = raw.shape
    d_flat = d.reshape(B, H, Tdim * P)
    return torch.einsum('bhk,hkn->bhn', d_flat, proj_matrix)


def _perm_matmul_forward_eager_ste(raw, proj_matrix, T):
    # STE hard forward only; backward is handled by the custom CUDA kernel
    # (autograd.Function). `T` kept in signature for a uniform dispatch table.
    d = torch.where(raw > 0, 0.5, -0.5).to(raw.dtype)
    B, H, Tdim, P = raw.shape
    d_flat = d.reshape(B, H, Tdim * P)
    return torch.einsum('bhk,hkn->bhn', d_flat, proj_matrix)


if _USE_PERMLUT_COMPILE:
    _perm_matmul_forward_compiled_sigmoid = torch.compile(_perm_matmul_forward_eager_sigmoid, dynamic=True)
    _perm_matmul_forward_compiled_rational = torch.compile(_perm_matmul_forward_eager_rational, dynamic=True)
    _perm_matmul_forward_compiled_ste = torch.compile(_perm_matmul_forward_eager_ste, dynamic=True)
else:
    _perm_matmul_forward_compiled_sigmoid = _perm_matmul_forward_eager_sigmoid
    _perm_matmul_forward_compiled_rational = _perm_matmul_forward_eager_rational
    _perm_matmul_forward_compiled_ste = _perm_matmul_forward_eager_ste


_PERM_MATMUL_FORWARD_DISPATCH = {
    0: _perm_matmul_forward_compiled_sigmoid,
    1: _perm_matmul_forward_compiled_rational,
    2: _perm_matmul_forward_compiled_ste,
}


def _signed_vote_jacobian(raw: torch.Tensor, soft_mode_int: int, T: float) -> torch.Tensor:
    """Jacobian d(d)/d(raw) for each soft mode. Used for explicit matmul backward."""
    if soft_mode_int == 0:  # sigmoid
        s = torch.sigmoid(raw / T)
        return s * (1.0 - s) / T
    # rational and ste (ste uses rational as backward surrogate)
    # d/draw[0.5 * raw / (T + |raw|)] = 0.5 * T / (T + |raw|)^2
    denom = T + raw.abs()
    return 0.5 * T / (denom * denom)


class _PermLutMatmulFunction(torch.autograd.Function):
    """Matmul forward + fused CUDA backward (hybrid path).

    Forward: d = signed_vote(raw); out = einsum(d, proj_matrix)
    Backward: uses the custom CUDA kernel that reads (grad_out, raw, idx_a, idx_b)
              and writes grad_raw directly in one pass — same kernel as scatter path.

    Falls back to explicit PyTorch backward if the native kernel is unavailable
    (CPU, fp16, or when SPIKY_PERMLUT_NO_CUSTOM_CUDA=1).
    """

    @staticmethod
    def forward(ctx, raw, proj_matrix, idx_a, idx_b, soft_mode_int, temperature):
        # raw: [B, H, T, P]. Forward delegated to a (optionally) compiled function.
        T = float(temperature)
        fn = _PERM_MATMUL_FORWARD_DISPATCH[int(soft_mode_int)]
        out = fn(raw, proj_matrix, T)

        ctx.save_for_backward(raw, proj_matrix, idx_a, idx_b)
        ctx.soft_mode_int = int(soft_mode_int)
        ctx.temperature = T
        ctx.raw_shape = raw.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        raw, proj_matrix, idx_a, idx_b = ctx.saved_tensors
        B, H, Tdim, P = ctx.raw_shape

        # Prefer the fused CUDA backward kernel (used by the scatter path too).
        # It expects raw as [B, H*T, P] and writes grad_raw in one pass.
        if (
            _USE_PERMLUT_CUSTOM_CUDA
            and raw.is_cuda
            and raw.dtype in (torch.float32, torch.float64)
        ):
            native = _get_permlut_native()
            if native is not None:
                raw_flat = raw.reshape(B, H * Tdim, P)
                grad_raw_flat = native.perm_lut_backward(
                    grad_out.contiguous(), raw_flat, idx_a, idx_b,
                    ctx.soft_mode_int, ctx.temperature,
                )
                grad_raw = grad_raw_flat.view(B, H, Tdim, P)
                return grad_raw, None, None, None, None, None

        # Fallback: explicit PyTorch backward (einsum + jacobian)
        grad_d_flat = torch.einsum('bhn,hkn->bhk', grad_out.contiguous(), proj_matrix)
        grad_d = grad_d_flat.view(B, H, Tdim, P)
        jac = _signed_vote_jacobian(raw, ctx.soft_mode_int, ctx.temperature)
        grad_raw = grad_d * jac
        return grad_raw, None, None, None, None, None


class PermutationalLut(nn.Module):
    """
    Permutational LUT with per-pair aggregation.

    Args:
        n_inputs:    Input dimensionality.
        n_outputs:   Output dimensionality.
        input_nap:   Number of input anchor pairs per table.
        output_nap:  Number of output dominance pairs per table.
        n_heads:     Number of heads.
        tph:         Tables per head.
        pair_mode:   'aligned' or 'scrambled'.
        soft_mode:   'sigmoid', 'rational', or 'ste' (default 'rational').
        aggregation: 'matmul' (default; einsum against precomputed projection matrix
                     + custom CUDA backward — fastest) or 'scatter' (scatter_add +
                     custom CUDA backward — kept as reference). Both paths are
                     mathematically equivalent.
        temperature: Temperature for the soft score function (default 0.1).
        random_seed: Seed for anchor pair sampling.
        device:      Target device.
        **mhlut_kwargs: Extra kwargs forwarded to the inner MultiHeadLut
                        (e.g. initial_weights_noise, recompute_in_backward, ...).
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        input_nap: int,
        output_nap: int,
        n_heads: int,
        tph: int,
        pair_mode: str = 'aligned',
        soft_mode: str = 'rational',
        aggregation: str = 'matmul',
        temperature: float = 0.1,
        use_fp8: bool = False,
        return_dominance: bool = False,
        scrambled_policy: AnchorSamplingPolicy = AnchorSamplingPolicy.FULL_COVERAGE,
        input_anchor_policy: AnchorSamplingPolicy = AnchorSamplingPolicy.FULL_COVERAGE,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        **mhlut_kwargs,
    ):
        super().__init__()

        if pair_mode not in ('aligned', 'scrambled'):
            raise ValueError(f"pair_mode must be 'aligned' or 'scrambled', got {pair_mode!r}")
        if soft_mode not in ('sigmoid', 'rational', 'ste'):
            raise ValueError(f"soft_mode must be 'sigmoid', 'rational', or 'ste', got {soft_mode!r}")
        if aggregation not in ('scatter', 'matmul'):
            raise ValueError(f"aggregation must be 'scatter' or 'matmul', got {aggregation!r}")
        if pair_mode == 'aligned':
            if input_nap != output_nap:
                raise ValueError(
                    f"'aligned' mode requires input_nap == output_nap "
                    f"(got input_nap={input_nap}, output_nap={output_nap})"
                )
            if n_inputs != n_outputs:
                raise ValueError(
                    f"'aligned' mode requires n_inputs == n_outputs "
                    f"(got n_inputs={n_inputs}, n_outputs={n_outputs})"
                )

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_nap = input_nap
        self.output_nap = output_nap
        self.n_heads = n_heads
        self.tph = tph
        self.pair_mode = pair_mode
        self.soft_mode = soft_mode
        self.aggregation = aggregation
        self.temperature = temperature
        self.use_fp8 = use_fp8
        self.return_dominance = return_dominance
        self._cached_raw = None
        self._cached_grad_output = None
        self._cached_lookup_indices = None
        self._bitflip_hooks_registered = False

        # Inner LUT: each table outputs `output_nap` values (one signed vote per output pair)
        self.inner = MultiHeadLut(
            input_dim=n_inputs,
            n_heads=n_heads,
            n_outputs=output_nap,
            n_anchor_pairs=input_nap,
            tables_per_head=tph,
            anchor_sampling_policy=input_anchor_policy,
            uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=random_seed,
            device=device,
            return_per_table_outputs=True,
            **mhlut_kwargs,
        )

        # Build output pair endpoint indices: [n_heads, tph, output_nap]
        if pair_mode == 'aligned':
            # Reuse inner LUT's input anchor pairs as output pairs.
            # Inner stores them as [n_lookup_tables, input_nap] = [n_heads*tph, input_nap].
            idx_a = self.inner.lookup.anchor_pairs_a.view(n_heads, tph, input_nap).long()
            idx_b = self.inner.lookup.anchor_pairs_b.view(n_heads, tph, input_nap).long()
        else:
            # Scrambled: sample new pairs over the OUTPUT dimension space.
            dev = device or torch.device("cpu")
            seed = (random_seed + 1_000_003) if random_seed is not None else None
            idx_a_flat, idx_b_flat = get_balanced_anchor_pairs(
                n_tables=n_heads * tph,
                n_anchor_pairs=output_nap,
                input_dim=n_outputs,
                device=dev,
                random_seed=seed,
                policy=scrambled_policy,
                n_heads=n_heads,
                shuffle_per_head=True,
            )
            idx_a = idx_a_flat.view(n_heads, tph, output_nap).long()
            idx_b = idx_b_flat.view(n_heads, tph, output_nap).long()

        # Store as [n_heads, tph * output_nap] for the scatter_add path.
        self.register_buffer('idx_a', idx_a.reshape(n_heads, tph * output_nap).contiguous())
        self.register_buffer('idx_b', idx_b.reshape(n_heads, tph * output_nap).contiguous())

        if aggregation == 'matmul':
            # Precomputed dense projection matrix M: [H, TP, N] where
            #   M[h, i, n] = +1 if idx_a[h, i] == n
            #             = -1 if idx_b[h, i] == n
            #             =  0 otherwise
            # Each row of M has exactly 2 non-zeros.
            # Forward: out = einsum('bhk,hkn->bhn', d, M)   (d has shape [B, H, T*P])
            # Backward is automatic via torch.einsum.
            TP = tph * output_nap
            M = torch.zeros(n_heads, TP, n_outputs, device=self.idx_a.device)
            h_idx = torch.arange(n_heads, device=self.idx_a.device).view(n_heads, 1).expand(n_heads, TP)
            tp_idx = torch.arange(TP, device=self.idx_a.device).view(1, TP).expand(n_heads, TP)
            M[h_idx, tp_idx, self.idx_a] = 1.0
            M[h_idx, tp_idx, self.idx_b] = -1.0
            self.register_buffer('proj_matrix', M)

        if return_dominance:
            P = n_outputs * (n_outputs - 1) // 2
            self._dom_n_pairs = P
            # Map each scrambled pair (a, b) to its canonical pair index.
            pair_map = torch.full((n_outputs, n_outputs), -1, dtype=torch.long)
            tri_i, tri_j = torch.triu_indices(n_outputs, n_outputs, offset=1)
            pair_map[tri_i, tri_j] = torch.arange(P)
            pair_map[tri_j, tri_i] = torch.arange(P)

            # Dedicated dominance kernel: one atomicAdd per slot.
            pair_idx = pair_map[self.idx_a.cpu(), self.idx_b.cpu()].to(self.idx_a.device).long()
            sign = torch.where(self.idx_a < self.idx_b, 1.0, -1.0).float()
            self.register_buffer('dom_pair_idx', pair_idx.contiguous())  # [H, TP]
            self.register_buffer('dom_sign', sign.contiguous())          # [H, TP]

            # Inverse index for the gather kernel: for each (h, p), list of raw
            # slot indices (within that head) that contribute, plus their ±1 sign.
            # Padded with -1. Assertion: within a single table, all output_nap
            # slots must map to DISTINCT canonical pairs (satisfied by scrambled
            # + full_coverage policy).
            TP_ = pair_idx.shape[1]
            counts_cpu = torch.zeros(n_heads, P, dtype=torch.long)
            pair_idx_cpu = pair_idx.cpu()
            sign_cpu = sign.cpu()
            for h in range(n_heads):
                for slot in range(TP_):
                    counts_cpu[h, pair_idx_cpu[h, slot].item()] += 1
            # Note: under FULL_COVERAGE scrambled sampling, a single table can
            # have multiple slots mapping to the same canonical pair (e.g. one
            # slot (0,5) and another (5,0)). The gather kernel sums them.
            # When we switch to a canonical-distinct-pairs sampler (option C),
            # K_max will drop to ~N_votes_per_pair.
            K_max = int(counts_cpu.max().item())
            inv_idx_cpu = torch.full((n_heads, P, K_max), -1, dtype=torch.long)
            inv_sign_cpu = torch.zeros(n_heads, P, K_max)
            cursor = torch.zeros(n_heads, P, dtype=torch.long)
            for h in range(n_heads):
                for slot in range(TP_):
                    p = int(pair_idx_cpu[h, slot].item())
                    k = int(cursor[h, p].item())
                    inv_idx_cpu[h, p, k] = slot
                    inv_sign_cpu[h, p, k] = float(sign_cpu[h, slot].item())
                    cursor[h, p] += 1
            self.register_buffer('dom_inv_idx', inv_idx_cpu.to(self.idx_a.device).contiguous())   # [H, P, K]
            self.register_buffer('dom_inv_sign', inv_sign_cpu.to(self.idx_a.device).float().contiguous())  # [H, P, K]
            # Borda matrix for converting dominance back to ranks
            # Borda matrix: each row sums d_v - 1 non-zero ±1 entries → divide
            # by sqrt(n_outputs - 1) for CLT-stable unit scale on Borda output.
            borda_m = torch.zeros(n_outputs, P, device=self.idx_a.device)
            for p_idx in range(P):
                borda_m[int(tri_i[p_idx]), p_idx] = 1.0
                borda_m[int(tri_j[p_idx]), p_idx] = -1.0
            borda_m = borda_m / math.sqrt(max(n_outputs - 1, 1))
            self.register_buffer('dom_borda_m', borda_m)

    def enable_bitflip_cache(self):
        """Enable caching of raw outputs and output gradients for BitFlipOptimizer."""
        self._bitflip_hooks_registered = True

    def _signed_vote(self, raw: torch.Tensor) -> torch.Tensor:
        """Compute centred per-pair signed vote in (-0.5, +0.5)."""
        if self.soft_mode == 'sigmoid':
            return torch.sigmoid(raw / self.temperature) - 0.5
        if self.soft_mode == 'rational':
            return 0.5 * raw / (self.temperature + raw.abs())
        # ste: hard forward, soft (rational) backward
        hard = torch.where(raw > 0, 0.5, -0.5)
        soft = 0.5 * raw / (self.temperature + raw.abs())
        return hard.detach() + (soft - soft.detach())

    def _can_use_native(self, raw: torch.Tensor) -> bool:
        return (
            _USE_PERMLUT_CUSTOM_CUDA
            and raw.is_cuda
            and raw.dtype in (torch.float32, torch.float64)
            and _get_permlut_native() is not None
        )

    def _forward_dominance(self, raw: torch.Tensor) -> torch.Tensor:
        """Return dominance pair scores [B, H, P] via dedicated fused CUDA kernel
        (one atomicAdd per slot). Falls back to pure PyTorch on CPU/fp16."""
        B = raw.shape[0]
        raw_flat = raw.reshape(B, self.n_heads * self.tph, self.output_nap)
        P = self._dom_n_pairs
        sign = self.dom_sign.to(raw.dtype)
        inv_sign = self.dom_inv_sign.to(raw.dtype)
        if self._can_use_native(raw):
            out = _PermLutDomFunction.apply(
                raw_flat, self.dom_pair_idx, sign,
                self.dom_inv_idx, inv_sign,
                P, _SOFT_MODE_INT[self.soft_mode], self.temperature,
            )
        else:
            # PyTorch fallback: compute signed_vote then scatter_add with sign.
            T = float(self.temperature)
            if self.soft_mode == 'sigmoid':
                d = torch.sigmoid(raw / T) - 0.5
            elif self.soft_mode == 'rational':
                d = 0.5 * raw / (T + raw.abs())
            else:
                hard = torch.where(raw > 0, 0.5, -0.5).to(raw.dtype)
                soft = 0.5 * raw / (T + raw.abs())
                d = hard.detach() + (soft - soft.detach())
            H = self.n_heads
            TP = self.tph * self.output_nap
            src = d.reshape(B, H, TP) * sign.unsqueeze(0)
            idx = self.dom_pair_idx.view(1, H, TP).expand(B, -1, -1)
            out = torch.zeros(B, H, P, device=raw.device, dtype=raw.dtype)
            out.scatter_add_(2, idx, src)
        n_votes_per_pair = self.tph * self.output_nap / float(P)
        return out / math.sqrt(n_votes_per_pair)

    def _borda_scale(self) -> float:
        """CLT-stable scale for implicit-Borda aggregation: each output dim sums
        ~tph*output_nap*2/n_outputs independent ±0.5 votes."""
        n_votes_per_output = self.tph * self.output_nap * 2 / float(self.n_outputs)
        return math.sqrt(n_votes_per_output)

    def _forward_matmul(self, raw: torch.Tensor) -> torch.Tensor:
        """Matmul forward + fused CUDA backward (hybrid)."""
        out = _PermLutMatmulFunction.apply(
            raw, self.proj_matrix, self.idx_a, self.idx_b,
            _SOFT_MODE_INT[self.soft_mode], self.temperature,
        )
        return out / self._borda_scale()

    def _forward_pytorch(self, raw: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch path: signed_vote + scatter_add. CPU/fp16/debug fallback."""
        d = self._signed_vote(raw)  # [B, H, T, P], in (-0.5, +0.5)

        B = raw.shape[0]
        H = self.n_heads
        TP = self.tph * self.output_nap
        N = self.n_outputs

        src = d.reshape(B, H, TP)
        idx_a = self.idx_a.view(1, H, TP).expand(B, -1, -1)
        idx_b = self.idx_b.view(1, H, TP).expand(B, -1, -1)

        out = torch.zeros(B, H, N, device=raw.device, dtype=raw.dtype)
        out.scatter_add_(2, idx_a, src)
        out.scatter_add_(2, idx_b, -src)
        return out / self._borda_scale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_inputs]

        Returns:
            [B, n_heads, n_outputs] — per-output net dominance scores (real-valued,
            centred around 0). Their ranking defines the output permutation.
        """
        if self.use_fp8:
            orig_data = self.inner.projection.weights.data
            self.inner.projection.weights.data = _quantize_to_fp8(orig_data)

        # Per-table raw outputs: [B, n_heads, tph, output_nap]
        raw = self.inner(x)

        if self.use_fp8:
            self.inner.projection.weights.data = orig_data

        if self._bitflip_hooks_registered:
            self._cached_raw = raw.detach()
            # Compute and cache lookup indices (cheap: just anchor pair comparisons)
            with torch.no_grad():
                from spiky.lutorch.anchor_pairs_lookup import _compute_anchor_data
                li, _, _, _, _ = _compute_anchor_data(
                    x, self.inner.lookup.anchor_pairs_a,
                    self.inner.lookup.anchor_pairs_b,
                    self.inner.lookup.powers, self.inner.lookup.cmp_eps, 1,
                )
                self._cached_lookup_indices = li  # [B, n_tables]

        if self.return_dominance:
            out = self._forward_dominance(raw)
        elif self.aggregation == 'matmul':
            out = self._forward_matmul(raw)
        else:
            B = raw.shape[0]
            raw_flat = raw.reshape(B, self.n_heads * self.tph, self.output_nap)
            if self._can_use_native(raw):
                out = _PermLutFunction.apply(
                    raw_flat, self.idx_a, self.idx_b,
                    self.n_outputs, _SOFT_MODE_INT[self.soft_mode], self.temperature,
                )
                out = out / self._borda_scale()
            else:
                out = self._forward_pytorch(raw)

        if self._bitflip_hooks_registered and out.requires_grad:
            out.register_hook(lambda g: setattr(self, '_cached_grad_output', g.detach()))

        return out
