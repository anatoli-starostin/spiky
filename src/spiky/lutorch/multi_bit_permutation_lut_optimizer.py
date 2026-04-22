"""MultiBitPermutationLUTOptimizer — bf16-latent Adam for MultiBitPermutationLUT.

Simplified from BitPermutationLUTOptimizer:
  * Only bf16 latent path (no fp8 / fp32).
  * Soft backward always on.
  * Per-step sequence: (1) loss.backward() populates grad_out via backward hook
    and lookup_indices via forward hook. (2) Project grad_out into
    weight_grad with bit_perm_lut_weight_grad (same kernel; bit-width agnostic).
    (3) Apply STE Jacobian gate (rational) on the latent. (4) Run Adam on the
    bf16 latent. (5) Re-pack bit_weights from the updated latent via
    multi_bit_pack.

Module support: designed for MultiBitPermutationLUT instances.
"""
import math
from typing import Callable, Iterable, List, Optional

import torch

from spiky.lutorch.determinism import is_deterministic


def _get_native():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


def _project_grad_out_to_weight_grad_mb(
    grad_out: torch.Tensor,            # [B, n_heads, P] float32
    lookup_indices: torch.Tensor,      # int16 [B, n_heads*tph]
    pair_idx_per_slot: torch.Tensor,   # int32 [n_heads, tph, output_nap]
    n_heads: int,
    tph: int,
    output_nap: int,
    table_dim: int,
    scale: float,
    wg_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter grad_out through (lookup_indices, pair_idx) into weight_grad.
    Same shape/logic as _project_grad_out_to_weight_grad for BitPermLUT —
    K-bit-agnostic (STE through quantisation means the continuous latent
    absorbs the grad as if the weights were floats)."""
    B, N = lookup_indices.shape
    assert N == n_heads * tph
    if wg_buffer is None:
        wg = torch.zeros(N, table_dim, output_nap, device=grad_out.device, dtype=torch.float32)
    else:
        wg = wg_buffer
        wg.zero_()

    native = _get_native()
    use_kernel = (
        not is_deterministic()
        and grad_out.is_cuda
        and native is not None
    )
    if use_kernel:
        n_pairs = grad_out.size(2)
        native.bit_perm_lut_weight_grad(
            grad_out.contiguous().to(torch.float32),
            lookup_indices.contiguous(),
            pair_idx_per_slot.contiguous(),
            wg,
            int(n_heads), int(tph), int(output_nap), int(table_dim),
            int(n_pairs),
            float(scale),
        )
        return wg

    # Deterministic fallback: index_add_ scatter.
    pair_flat = pair_idx_per_slot.reshape(n_heads, tph * output_nap).long()
    g_slot = grad_out.gather(2, pair_flat.unsqueeze(0).expand(B, -1, -1)) * scale
    g_slot = g_slot.reshape(B, N, output_nap).to(torch.float32)
    entries = lookup_indices.long()
    N_idx = torch.arange(N, device=grad_out.device).unsqueeze(0).expand(B, -1)
    flat_idx = (N_idx * table_dim + entries).reshape(-1)
    wg.view(N * table_dim, output_nap).index_add_(0, flat_idx, g_slot.reshape(-1, output_nap))
    return wg


class MultiBitPermutationLUTOptimizer:
    """Adam on bf16 latent with STE gate, matching BitPermutationLUTOptimizer's
    interface but without fp8/fp32 paths.

    Args:
        luts: iterable of MultiBitPermutationLUT modules.
        lr: base learning rate.
        beta1, beta2, eps: Adam hyperparams.
        weight_decay: standard weight decay on the bf16 latent.
        lr_schedule_fn: optional callable step -> multiplier on lr.
        weight_grad_gate: if True (default), apply the rational STE gate
            `0.5 * T / (T + |latent|)^2` on weight_grad. Mirrors BitPermLUT.
        weight_grad_gate_temperature: T used by the gate (default 0.1).
        weight_grad_gate_scale_matched: if True, pre-multiply weight_grad by
            lut.scale / 0.5 to match the analytical Jacobian of the scaled
            forward (same convention as BitPermLUT).
    """

    def __init__(
        self,
        luts: Iterable[torch.nn.Module],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
        weight_grad_gate: bool = True,
        weight_grad_gate_temperature: float = 0.1,
        weight_grad_gate_scale_matched: bool = True,
    ):
        self.modules: List[torch.nn.Module] = list(luts)
        if not self.modules:
            raise ValueError("MultiBitPermutationLUTOptimizer requires >=1 module")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.lr_schedule_fn = lr_schedule_fn
        self.weight_grad_gate = bool(weight_grad_gate)
        self.weight_grad_gate_temperature = float(weight_grad_gate_temperature)
        self.weight_grad_gate_scale_matched = bool(weight_grad_gate_scale_matched)
        if self.weight_grad_gate_temperature <= 0:
            raise ValueError("weight_grad_gate_temperature must be > 0")

        self._step_count = 0
        self._state = []  # per-LUT state dicts
        self._handles = []

        dev = self.modules[0].bit_weights.device
        # Shared scratch for wg_buffer (one per LUT is simplest).
        for lut in self.modules:
            state = {
                "mode": "bf16",
                "grad_out": None,
                "lookup_indices": None,
                # Per-LUT Adam state in fp32.
                "m_f32": torch.zeros_like(lut.latent_bf16, dtype=torch.float32),
                "v_f32": torch.zeros_like(lut.latent_bf16, dtype=torch.float32),
                "wg_buffer": torch.zeros_like(lut.latent_bf16, dtype=torch.float32),
            }
            self._state.append(state)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                x = inputs[0]
                with torch.no_grad():
                    li, _, _, _, _ = module.anchor(x)
                state["lookup_indices"] = li
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    def zero_grad(self) -> None:
        """No-op: grad_out / lookup_indices are populated by hooks."""
        pass

    def step(self) -> None:
        self._step_count += 1
        lr_now = self.lr * (self.lr_schedule_fn(self._step_count)
                             if self.lr_schedule_fn is not None else 1.0)

        bias1 = 1.0 - self.beta1 ** self._step_count
        bias2_sqrt = math.sqrt(1.0 - self.beta2 ** self._step_count)

        for lut, state in zip(self.modules, self._state):
            go = state["grad_out"]
            li = state["lookup_indices"]
            if go is None or li is None:
                continue

            weight_grad = _project_grad_out_to_weight_grad_mb(
                go, li,
                lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
                wg_buffer=state["wg_buffer"],
            )

            torch.nan_to_num_(weight_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale-matched STE gate on weight_grad (rational Jacobian on latent).
            if self.weight_grad_gate and self.weight_grad_gate_scale_matched:
                # lut.scale = 0.5 / (half_range * sqrt(nvpp)). To keep the
                # effective gate multiplier at 1/sqrt(nvpp) (same as BitPermLUT)
                # we undo the 1/half_range baked into lut.scale.
                half_range = float(getattr(lut, 'half_range', 1))
                weight_grad.mul_(lut.scale * half_range / 0.5)
            if self.weight_grad_gate:
                T = self.weight_grad_gate_temperature
                latent_f = lut.latent_bf16.to(torch.float32)
                denom = T + latent_f.abs()
                gate = (0.5 * T) / (denom * denom)
                weight_grad.mul_(gate)

            # Adam on bf16 latent via fp32 moments.
            m = state["m_f32"]
            v = state["v_f32"]
            m.mul_(self.beta1).add_(weight_grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(weight_grad, weight_grad, value=1 - self.beta2)
            if self.weight_decay != 0.0:
                lut.latent_bf16.mul_((1.0 - lr_now * self.weight_decay))
            denom = v.sqrt().add_(self.eps * bias2_sqrt)
            update = -lr_now * bias2_sqrt / bias1 * (m / denom)
            lut.latent_bf16.add_(update.to(torch.bfloat16))

            # Re-pack bit_weights from the updated bf16 latent.
            lut.refresh_bit_weights()

            state["grad_out"] = None
            state["lookup_indices"] = None

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
