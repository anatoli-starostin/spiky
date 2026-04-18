"""Latent fp8 Adam optimizer for BitPermutationLUT.

Training recipe (from transformer_exps/bitflip_clean/train_fp8_adam.py):
  - Binary weights are a read-only view of a float latent: w = sign(latent).
  - Latent is stored as torch.float8_e4m3fn with a fixed scale (values in
    [-1, 1] fill the fp8 range). One byte per weight slot.
  - Adam moments m, v are stored as fp8 with a per-table (n_heads*tph)
    dynamic amax scale, held as float32.
  - Latent init: (U(0,1) - 0.5) * 2 * latent_init_std, default +- 0.001.

Each step:
  1. Refresh bit_weights := sign(latent) so the discrete forward uses them.
  2. Forward + loss.backward(). A forward-hook registers an output-grad
     hook to stash grad_out; `lookup_indices` is read from the module's
     `last_lookup_indices` cache (no re-running of anchor lookup).
  3. Project grad_out through lookup_indices to a per-entry weight gradient
     of shape [n_heads*tph, table_dim, output_nap]. Only `entry_main(b, n)`
     receives gradient -- the forward read one row only.
  4. Dequantize latent/m/v to float32, Adam update, requantize to fp8.

Safety:
  - If any element of weight_grad is non-finite (NaN/Inf), the whole step
    for that module is skipped (state untouched). This keeps one bad
    batch from corrupting fp8 state permanently.
  - The Adam update is fused to avoid materializing mhat / vhat (would be
    two extra latent-sized float32 temporaries per step).
"""
import math
from typing import Callable, Iterable, List, Optional

import torch


_FP8 = getattr(torch, "float8_e4m3fn", None)
_FP8_AMAX = 448.0


def _to_fp8_fixed(t_f32: torch.Tensor, mx: float = 1.0) -> torch.Tensor:
    """Quantize float32 -> fp8 with fixed scale = _FP8_AMAX / mx.

    Clamps to [-mx, mx] before casting (also truncates any NaN/Inf to +-mx
    implicitly -- but `.clamp` returns NaN on NaN input, so caller must
    guard against NaN in t_f32).
    """
    s = _FP8_AMAX / mx
    return (t_f32.clamp(-mx, mx) * s).to(_FP8)


def _from_fp8_fixed(t_fp8: torch.Tensor, mx: float = 1.0) -> torch.Tensor:
    s = _FP8_AMAX / mx
    return t_fp8.to(torch.float32) / s


def _to_fp8_per_table(t_f32: torch.Tensor):
    """Quantize float32 -> (fp8, scale) using per-table (dim 0) amax.

    `scale` is float32 with shape [N, 1, ..., 1] so t_fp8.to(f32) / scale
    recovers t_f32 (up to fp8 rounding). amax is clamped to a small floor
    (1e-20) to avoid div-by-zero when a table is all zeros.
    """
    amax = t_f32.abs().amax(dim=tuple(range(1, t_f32.dim())), keepdim=True).clamp(min=1e-20)
    scale = _FP8_AMAX / amax
    return (t_f32 * scale).to(_FP8), scale


def _from_fp8_per_table(t_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return t_fp8.to(torch.float32) / scale


def _project_grad_out_to_weight_grad(
    grad_out: torch.Tensor,            # [B, H, P] float
    lookup_indices: torch.Tensor,      # int16 [B, N]    (N = H*tph)
    pair_idx_per_slot: torch.Tensor,   # int32 [H, tph, output_nap]
    n_heads: int,
    tph: int,
    output_nap: int,
    table_dim: int,
    scale: float,
) -> torch.Tensor:
    """Project dominance-output gradient back to per-entry weight gradient.

    For each (b, n=h*tph+t) the forward read a single row
    bit_weights[n, entry_main(b, n), :]. Only that row receives gradient:
      weight_grad[n, e, k] += scale * grad_out[b, h, pair_idx[h, t, k]]
         iff entry_main(b, n) == e
    """
    B, N = lookup_indices.shape
    assert N == n_heads * tph, "lookup_indices last dim must equal n_heads*tph"
    pair_flat = pair_idx_per_slot.reshape(n_heads, tph * output_nap).long()
    g_slot = grad_out.gather(2, pair_flat.unsqueeze(0).expand(B, -1, -1)) * scale
    g_slot = g_slot.reshape(B, N, output_nap).to(torch.float32)

    entries = lookup_indices.long()
    N_idx = torch.arange(N, device=grad_out.device).unsqueeze(0).expand(B, -1)
    flat_idx = (N_idx * table_dim + entries).reshape(-1)
    wg = torch.zeros(N * table_dim, output_nap, device=grad_out.device, dtype=torch.float32)
    wg.index_add_(0, flat_idx, g_slot.reshape(-1, output_nap))
    return wg.view(N, table_dim, output_nap)


class BitPermutationLUTOptimizer:
    """Latent fp8 Adam for one or more `BitPermutationLUT` modules.

    Args:
        bit_luts:             iterable of BitPermutationLUT modules to optimize.
        lr:                   base learning rate (scaled by lr_schedule_fn).
        beta1, beta2, eps:    Adam hyper-parameters.
        latent_init_std:      half-width of uniform latent init (default 0.001).
        lr_schedule_fn:       optional callable step -> multiplier on `lr`.
        seed:                 optional seed for latent initialization (per-module).

    Usage:
        opt = BitPermutationLUTOptimizer([bit_lut], lr=1e-3)
        for step in range(n_steps):
            opt.zero_grad()
            loss = compute_loss(bit_lut(x))
            loss.backward()
            opt.step()
        opt.close()                      # remove forward hooks when done
    """

    def __init__(
        self,
        bit_luts: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        latent_init_std: float = 0.001,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
    ):
        if _FP8 is None:
            raise RuntimeError(
                "BitPermutationLUTOptimizer requires torch.float8_e4m3fn "
                "(available in PyTorch 2.1+ with fp8 support)."
            )
        self.modules = list(bit_luts)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_schedule_fn = lr_schedule_fn
        self._step_count = 0
        self._skip_count = 0            # count of modules skipped due to non-finite grad
        self._handles: List = []
        self._states: List[dict] = []

        for i, lut in enumerate(self.modules):
            dev = lut.bit_weights.device
            N = lut.n_heads * lut.tph
            shape = (N, lut.table_dim, lut.output_nap)

            gen = torch.Generator(device=dev).manual_seed(seed + 100_003 * i) if seed is not None else None
            # Build latent in fp8 directly (avoid holding a full float32 copy).
            latent_f32 = (torch.rand(shape, device=dev, generator=gen) - 0.5) * (2 * latent_init_std)
            latent_fp8 = _to_fp8_fixed(latent_f32, mx=1.0)
            del latent_f32

            # m, v initialised to zero in fp8 directly (scale floor handles zeros).
            m_fp8 = torch.zeros(shape, device=dev, dtype=_FP8)
            v_fp8 = torch.zeros(shape, device=dev, dtype=_FP8)
            m_scale = torch.full((N, 1, 1), _FP8_AMAX / 1e-20, device=dev, dtype=torch.float32)
            v_scale = m_scale.clone()

            state = {
                "latent_fp8": latent_fp8,   # fp8, fixed scale
                "m_fp8": m_fp8,             # fp8
                "m_scale": m_scale,         # float32 [N, 1, 1]
                "v_fp8": v_fp8,
                "v_scale": v_scale,
                "grad_out": None,
            }
            self._states.append(state)

            self._refresh_weights(lut, state)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    # --- hook ---
    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    # --- housekeeping ---
    @staticmethod
    def _refresh_weights(lut, state: dict) -> None:
        latent_f32 = _from_fp8_fixed(state["latent_fp8"], mx=1.0)
        signs = latent_f32.sign()
        signs[signs == 0] = 1.0
        lut.set_bit_weights_from_signs(signs)

    def zero_grad(self) -> None:
        for state in self._states:
            state["grad_out"] = None

    def close(self) -> None:
        """Detach forward hooks. After this, the optimizer is inert."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # --- step ---
    @torch.no_grad()
    def step(self) -> None:
        self._step_count += 1
        lr = self.lr
        if self.lr_schedule_fn is not None:
            lr = lr * self.lr_schedule_fn(self._step_count)
        bias1 = 1.0 - self.beta1 ** self._step_count
        bias2 = 1.0 - self.beta2 ** self._step_count
        bias2_sqrt = math.sqrt(max(bias2, 1e-30))

        for lut, state in zip(self.modules, self._states):
            go = state["grad_out"]
            li = getattr(lut, "last_lookup_indices", None)
            if go is None or li is None:
                continue

            weight_grad = _project_grad_out_to_weight_grad(
                go, li,
                lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
            )

            # Safety: reject non-finite grads rather than corrupt fp8 state.
            if not torch.isfinite(weight_grad).all():
                self._skip_count += 1
                state["grad_out"] = None
                continue

            latent_f = _from_fp8_fixed(state["latent_fp8"], mx=1.0)
            m_f = _from_fp8_per_table(state["m_fp8"], state["m_scale"])
            v_f = _from_fp8_per_table(state["v_fp8"], state["v_scale"])

            # Moments (in-place).
            m_f.mul_(self.beta1).add_(weight_grad, alpha=1 - self.beta1)
            v_f.mul_(self.beta2).addcmul_(weight_grad, weight_grad, value=1 - self.beta2)
            del weight_grad                           # done with it; free early

            # Fused Adam update without materializing mhat/vhat:
            #   update = -lr * (m/bias1) / (sqrt(v/bias2) + eps)
            #          = -(lr/bias1) * m / (sqrt(v)/sqrt(bias2) + eps)
            #          = -(lr * sqrt(bias2) / bias1) * m / (sqrt(v) + eps*sqrt(bias2))
            denom = v_f.sqrt().add_(self.eps * bias2_sqrt)    # one extra alloc
            latent_f.addcdiv_(m_f, denom, value=-lr * bias2_sqrt / bias1)
            del denom

            # Requantize.
            state["latent_fp8"] = _to_fp8_fixed(latent_f, mx=1.0)
            state["m_fp8"], state["m_scale"] = _to_fp8_per_table(m_f)
            state["v_fp8"], state["v_scale"] = _to_fp8_per_table(v_f)
            del latent_f, m_f, v_f

            self._refresh_weights(lut, state)
            state["grad_out"] = None

    # --- introspection ---
    def state_as_float(self, idx: int = 0) -> dict:
        """Return dequantized float32 state for module `idx` (for tests / debug)."""
        s = self._states[idx]
        return {
            "latent": _from_fp8_fixed(s["latent_fp8"], mx=1.0),
            "m": _from_fp8_per_table(s["m_fp8"], s["m_scale"]),
            "v": _from_fp8_per_table(s["v_fp8"], s["v_scale"]),
        }
