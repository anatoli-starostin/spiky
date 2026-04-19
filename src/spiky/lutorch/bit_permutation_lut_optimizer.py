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


def _to_fp8_per_lut(t_f32: torch.Tensor):
    """Quantize with a single global amax over the whole tensor. Returns scale
    shape [N, 1, 1] (broadcast of one scalar) so it plugs into the existing
    per-table dequant path — "uniform per-table" is a special case of
    per-table, so the fused CUDA Adam kernel reads it transparently.
    """
    amax = t_f32.abs().amax().clamp(min=1e-20)
    scalar = _FP8_AMAX / amax
    scale = scalar.view(1, 1, 1).expand(t_f32.size(0), 1, 1).contiguous()
    return (t_f32 * scalar).to(_FP8), scale


def _get_bit_permlut_native():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


def _project_grad_out_to_weight_grad(
    grad_out: torch.Tensor,            # [B, H, P] float
    lookup_indices: torch.Tensor,      # int16 [B, N]    (N = H*tph)
    pair_idx_per_slot: torch.Tensor,   # int32 [H, tph, output_nap]
    n_heads: int,
    tph: int,
    output_nap: int,
    table_dim: int,
    scale: float,
    wg_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project dominance-output gradient back to per-entry weight gradient.

    CUDA path: small table_dim -> custom kernel atomicAdds into touched
    (n, entry_main(b, n), k) positions. Large table_dim -> torch's
    vectorized index_add_ is faster than the atomic loop.
    `wg_buffer` may be supplied to avoid per-step allocation; it will be
    zeroed in-place.
    """
    B, N = lookup_indices.shape
    assert N == n_heads * tph, "lookup_indices last dim must equal n_heads*tph"

    if wg_buffer is None:
        wg = torch.zeros(N, table_dim, output_nap, device=grad_out.device, dtype=torch.float32)
    else:
        wg = wg_buffer
        wg.zero_()

    native = _get_bit_permlut_native()
    # Crossover: for small table_dim the kernel wins; for large table_dim
    # (e.g. out_proj with table_dim=1024) index_add_ vectorized scatter wins.
    use_kernel = (
        grad_out.is_cuda and native is not None
        and table_dim <= 64
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

    # Fallback: vectorized scatter via index_add_.
    pair_flat = pair_idx_per_slot.reshape(n_heads, tph * output_nap).long()
    g_slot = grad_out.gather(2, pair_flat.unsqueeze(0).expand(B, -1, -1)) * scale
    g_slot = g_slot.reshape(B, N, output_nap).to(torch.float32)
    entries = lookup_indices.long()
    N_idx = torch.arange(N, device=grad_out.device).unsqueeze(0).expand(B, -1)
    flat_idx = (N_idx * table_dim + entries).reshape(-1)
    wg.view(N * table_dim, output_nap).index_add_(0, flat_idx, g_slot.reshape(-1, output_nap))
    return wg


class BitPermutationLUTOptimizer:
    """Latent fp8 Adam for one or more `BitPermutationLUT` modules.

    Args:
        bit_luts:             iterable of BitPermutationLUT modules to optimize.
        lr:                   base learning rate (scaled by lr_schedule_fn).
        beta1, beta2, eps:    Adam hyper-parameters.
        lr_schedule_fn:       optional callable step -> multiplier on `lr`.

    Latent state (`lut.latent_fp8`) is owned by the BitPermutationLUT itself
    and is initialized in its constructor (see `initial_weights_noise` there).
    The optimizer only holds the Adam moments and transient buffers.

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
        weight_decay: float = 0.0,
        weight_grad_gate: bool = True,
        weight_grad_gate_temperature: float = 0.1,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
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
        self.weight_decay = weight_decay
        self.weight_grad_gate = bool(weight_grad_gate)
        self.weight_grad_gate_temperature = float(weight_grad_gate_temperature)
        if self.weight_grad_gate_temperature <= 0:
            raise ValueError(
                f"weight_grad_gate_temperature must be > 0, got {weight_grad_gate_temperature}"
            )
        self.lr_schedule_fn = lr_schedule_fn
        self._step_count = 0
        self._handles: List = []
        self._states: List[dict] = []

        # Shared f32 scratches sized to the largest LUT. Step() processes
        # LUTs sequentially so one buffer of each kind suffices for all
        # (vs one-per-LUT duplication that would waste ~N_luts × max_size).
        max_numel = max(
            lut.n_heads * lut.tph * lut.table_dim * lut.output_nap
            for lut in self.modules
        )
        dev = self.modules[0].bit_weights.device
        self._wg_buffer = torch.zeros(max_numel, device=dev, dtype=torch.float32)
        self._m_f32_scratch = torch.empty(max_numel, device=dev, dtype=torch.float32)
        self._v_f32_scratch = torch.empty(max_numel, device=dev, dtype=torch.float32)
        self._m_amax_scratch = torch.zeros(1, device=dev, dtype=torch.float32)
        self._v_amax_scratch = torch.zeros(1, device=dev, dtype=torch.float32)

        for lut in self.modules:
            dev = lut.bit_weights.device
            N = lut.n_heads * lut.tph
            shape = (N, lut.table_dim, lut.output_nap)

            mode = getattr(lut, 'latent_dtype', 'fp8')
            if mode == 'fp8':
                # m, v initialised to zero in fp8 directly (scale floor handles zeros).
                m_fp8 = torch.zeros(shape, device=dev, dtype=_FP8)
                v_fp8 = torch.zeros(shape, device=dev, dtype=_FP8)
                m_scale = torch.full((N, 1, 1), _FP8_AMAX / 1e-20, device=dev, dtype=torch.float32)
                v_scale = m_scale.clone()
                # Re-quantize the module's initial latent onto the fixed-scale
                # grid (mx=1.0, scale=_FP8_AMAX constant).
                latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
                lut.latent_fp8.copy_(_to_fp8_fixed(latent_f, mx=1.0))
                lut.latent_scale.fill_(_FP8_AMAX)
                state = {
                    "mode": "fp8",
                    "m_fp8": m_fp8, "m_scale": m_scale,
                    "v_fp8": v_fp8, "v_scale": v_scale,
                    "grad_out": None,
                    "lookup_indices": None,
                }
            elif mode == 'fp32':
                # Standard Adam state: per-weight f32 m, v. No fp8 anywhere.
                m_f32 = torch.zeros(shape, device=dev, dtype=torch.float32)
                v_f32 = torch.zeros(shape, device=dev, dtype=torch.float32)
                state = {
                    "mode": "fp32",
                    "m_f32": m_f32,
                    "v_f32": v_f32,
                    "grad_out": None,
                    "lookup_indices": None,
                }
            elif mode == 'bf16':
                # bf16 latent + m + v (6 bytes/weight). m and v use per-LUT
                # scalar scale (f32) so that stored bf16 values span [-1, 1]
                # using the full mantissa precision.
                m_bf16 = torch.zeros(shape, device=dev, dtype=torch.bfloat16)
                v_bf16 = torch.zeros(shape, device=dev, dtype=torch.bfloat16)
                # amax scales start at 1.0 (neutral — bf16 ≈ f32 at small scale).
                m_scale = torch.ones(1, device=dev, dtype=torch.float32)
                v_scale = m_scale.clone()
                state = {
                    "mode": "bf16",
                    "m_bf16": m_bf16, "m_scale": m_scale,
                    "v_bf16": v_bf16, "v_scale": v_scale,
                    "grad_out": None,
                    "lookup_indices": None,
                }
            else:
                raise NotImplementedError(
                    f"BitPermutationLUTOptimizer does not yet support latent_dtype={mode!r}"
                )
            self._states.append(state)

            # Ensure bit_weights are consistent with the initial latent.
            self._refresh_weights(lut)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    # --- hook ---
    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                x = inputs[0]
                # Re-run the (cheap) anchor lookup on the forward input to get
                # lookup_indices for the optimizer's weight-grad projection.
                # ~0.04 ms per call; simpler than caching from forward().
                with torch.no_grad():
                    li, _, _, _, _ = module.anchor(x)
                state["lookup_indices"] = li
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    # --- housekeeping ---
    @staticmethod
    def _refresh_weights(lut) -> None:
        """Pack sign(latent) into bit_weights."""
        mode = getattr(lut, 'latent_dtype', 'fp8')
        if mode == 'fp32':
            lut.set_bit_weights_from_signs(lut.latent_fp32)
            return
        if mode == 'bf16':
            lut.set_bit_weights_from_signs(lut.latent_bf16.to(torch.float32))
            return
        # fp8 mode: pack sign(latent_fp8) directly from the fp8 bytes
        # (no float materialization). sign of (latent_fp8 / latent_scale) is
        # the same as sign of latent_fp8 since latent_scale > 0.
        native = _get_bit_permlut_native()
        if native is not None and hasattr(native, "bit_pack_fp8_signs"):
            native.bit_pack_fp8_signs(
                lut.latent_fp8, lut.bit_weights, int(lut.output_nap),
            )
        else:
            # Fallback: dequant to float32 then pack.
            latent_f32 = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
            lut.set_bit_weights_from_signs(latent_f32)

    def zero_grad(self) -> None:
        """No-op. `grad_out` and `lookup_indices` are filled by hooks during
        forward/backward and consumed/cleared by `step()`. Zeroing them here
        would wipe state captured by the forward hook if `zero_grad` is called
        between forward and backward (as is common in training loops)."""
        return

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
            li = state.get("lookup_indices")
            if go is None or li is None:
                continue

            N = lut.n_heads * lut.tph
            numel = N * lut.table_dim * lut.output_nap
            wg_view = self._wg_buffer[:numel].view(N, lut.table_dim, lut.output_nap)
            weight_grad = _project_grad_out_to_weight_grad(
                go, li,
                lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
                wg_buffer=wg_view,
            )

            # Safety: replace any non-finite gradient entries with 0 in-place.
            # Element-wise masking keeps the update on the GPU (no d2h sync).
            torch.nan_to_num_(weight_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # PermLut-style STE Jacobian gate on weight_grad (matches
            # soft_mode='ste' in PermutationalLut): gradient fades as |latent|
            # grows, concentrating updates on "undecided" bits near zero.
            #   gate(x) = 0.5 * T / (T + |x|)^2
            # At x=0: 0.5/T (peak). At |x|→∞: 0 (locked-in weight).
            # Lower T → sharper peak; higher T → gentler gating.
            #
            # fp8 and bf16 modes apply the gate INSIDE the fused Adam kernel
            # (per-thread, no extra Python op). fp32 mode does it here.
            gate_T_kernel = 0.0   # >0 signals "apply gate" to fused kernel
            dtype_mode = getattr(lut, 'latent_dtype', 'fp8')
            if self.weight_grad_gate:
                T = self.weight_grad_gate_temperature
                if dtype_mode == 'fp32':
                    latent_f = lut.latent_fp32
                    denom = T + latent_f.abs()
                    gate = (0.5 * T) / (denom * denom)
                    weight_grad.mul_(gate)
                else:
                    # fp8 / bf16: let the kernel handle it.
                    gate_T_kernel = T

            if state["mode"] == "fp32":
                # Standard Adam on f32 latent — identical to torch.optim.Adam
                # (and AdamW when weight_decay != 0).
                m = state["m_f32"]
                v = state["v_f32"]
                m.mul_(self.beta1).add_(weight_grad, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(weight_grad, weight_grad, value=1 - self.beta2)
                if self.weight_decay != 0.0:
                    lut.latent_fp32.mul_(1.0 - lr * self.weight_decay)
                denom = v.sqrt().add_(self.eps * bias2_sqrt)
                lut.latent_fp32.addcdiv_(m, denom, value=-lr * bias2_sqrt / bias1)
                self._refresh_weights(lut)
                state["grad_out"] = None
                state["lookup_indices"] = None
                continue

            if state["mode"] == "bf16":
                native = _get_bit_permlut_native()
                if native is not None and hasattr(native, "fused_bf16_adam_full_inkernel"):
                    # Fully fused CUDA kernel (mirrors fused_fp8_adam_full_inkernel).
                    if self.weight_decay != 0.0:
                        lut.latent_bf16.mul_(1.0 - lr * self.weight_decay)
                    numel = state["m_bf16"].numel()
                    m_scratch = self._m_f32_scratch[:numel].view_as(state["m_bf16"])
                    v_scratch = self._v_f32_scratch[:numel].view_as(state["v_bf16"])
                    native.fused_bf16_adam_full_inkernel(
                        lut.latent_bf16,
                        state["m_bf16"], state["m_scale"],
                        state["v_bf16"], state["v_scale"],
                        weight_grad,
                        m_scratch, v_scratch,
                        self._m_amax_scratch, self._v_amax_scratch,
                        float(self.beta1), float(self.beta2),
                        float(self.eps), float(bias1), float(bias2),
                        float(lr),
                        gate_T=float(gate_T_kernel),
                    )
                else:
                    # Python fallback (slow).
                    m_f = state["m_bf16"].to(torch.float32) * state["m_scale"]
                    v_f = state["v_bf16"].to(torch.float32) * state["v_scale"]
                    latent_f = lut.latent_bf16.to(torch.float32)
                    m_f.mul_(self.beta1).add_(weight_grad, alpha=1 - self.beta1)
                    v_f.mul_(self.beta2).addcmul_(weight_grad, weight_grad, value=1 - self.beta2)
                    if self.weight_decay != 0.0:
                        latent_f.mul_(1.0 - lr * self.weight_decay)
                    denom = v_f.sqrt().add_(self.eps * bias2_sqrt)
                    latent_f.addcdiv_(m_f, denom, value=-lr * bias2_sqrt / bias1)
                    m_amax = m_f.abs().amax().clamp_(min=1e-20)
                    v_amax = v_f.abs().amax().clamp_(min=1e-20)
                    state["m_scale"].copy_(m_amax.view(1))
                    state["v_scale"].copy_(v_amax.view(1))
                    state["m_bf16"].copy_((m_f / m_amax).to(torch.bfloat16))
                    state["v_bf16"].copy_((v_f / v_amax).to(torch.bfloat16))
                    lut.latent_bf16.copy_(latent_f.to(torch.bfloat16))
                self._refresh_weights(lut)
                state["grad_out"] = None
                state["lookup_indices"] = None
                continue

            # fp8 mode: AdamW-style decoupled weight decay on latent (pre-step).
            # Done via an fp8 → f32 → shrink → fp8 round-trip when wd != 0.
            if self.weight_decay != 0.0:
                latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
                latent_f.mul_(1.0 - lr * self.weight_decay)
                lut.latent_fp8.copy_(_to_fp8_fixed(latent_f, mx=1.0))

            native = _get_bit_permlut_native()
            if native is not None and hasattr(native, "fused_fp8_adam_full_inkernel"):
                # Fully fused: kernel 1 does dequant + Adam + latent in-kernel
                # quant (in-place) + atomic-reduce of |m|, |v| to global amax
                # scalars; kernel 2 quantizes m, v to fp8 using those scalars.
                # All state updated in-place; no Python post-quant. Shared
                # f32 scratch (sized to the largest LUT) reused across LUTs.
                numel = state["m_fp8"].numel()
                m_scratch = self._m_f32_scratch[:numel].view_as(state["m_fp8"])
                v_scratch = self._v_f32_scratch[:numel].view_as(state["v_fp8"])
                native.fused_fp8_adam_full_inkernel(
                    lut.latent_fp8, lut.latent_scale,
                    state["m_fp8"], state["m_scale"],
                    state["v_fp8"], state["v_scale"],
                    weight_grad,
                    m_scratch, v_scratch,
                    self._m_amax_scratch, self._v_amax_scratch,
                    float(self.beta1), float(self.beta2),
                    float(self.eps), float(bias1), float(bias2),
                    float(lr),
                    gate_T=float(gate_T_kernel),
                )
            elif native is not None and hasattr(native, "fused_fp8_adam_latent_inplace"):
                # Intermediate variant (still has Python per-LUT requant of m, v).
                m_f, v_f = native.fused_fp8_adam_latent_inplace(
                    lut.latent_fp8, lut.latent_scale,
                    state["m_fp8"], state["m_scale"],
                    state["v_fp8"], state["v_scale"],
                    weight_grad,
                    float(self.beta1), float(self.beta2),
                    float(self.eps), float(bias1), float(bias2),
                    float(lr),
                )
                state["m_fp8"], state["m_scale"] = _to_fp8_per_lut(m_f)
                state["v_fp8"], state["v_scale"] = _to_fp8_per_lut(v_f)
            elif native is not None:
                # Legacy fused kernel fallback (pre-rebuild or older .so).
                latent_f, m_f, v_f = native.fused_fp8_adam(
                    lut.latent_fp8, lut.latent_scale,
                    state["m_fp8"], state["m_scale"],
                    state["v_fp8"], state["v_scale"],
                    weight_grad,
                    float(self.beta1), float(self.beta2),
                    float(self.eps), float(bias1), float(bias2),
                    float(lr),
                )
                lut.latent_fp8.copy_(_to_fp8_fixed(latent_f, mx=1.0))
                state["m_fp8"], state["m_scale"] = _to_fp8_per_lut(m_f)
                state["v_fp8"], state["v_scale"] = _to_fp8_per_lut(v_f)
            else:
                # CPU / fallback path (no fused kernel available).
                latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
                m_f = _from_fp8_per_table(state["m_fp8"], state["m_scale"])
                v_f = _from_fp8_per_table(state["v_fp8"], state["v_scale"])
                m_f.mul_(self.beta1).add_(weight_grad, alpha=1 - self.beta1)
                v_f.mul_(self.beta2).addcmul_(weight_grad, weight_grad, value=1 - self.beta2)
                denom = v_f.sqrt().add_(self.eps * bias2_sqrt)
                latent_f.addcdiv_(m_f, denom, value=-lr * bias2_sqrt / bias1)
                # fixed-scale fp8 quantization clamps to ±1 inside.
                lut.latent_fp8.copy_(_to_fp8_fixed(latent_f, mx=1.0))
                state["m_fp8"], state["m_scale"] = _to_fp8_per_lut(m_f)
                state["v_fp8"], state["v_scale"] = _to_fp8_per_lut(v_f)

            self._refresh_weights(lut)
            state["grad_out"] = None
            state["lookup_indices"] = None

    # --- introspection ---
    def state_as_float(self, idx: int = 0) -> dict:
        """Return dequantized float32 state for module `idx` (for tests / debug)."""
        s = self._states[idx]
        lut = self.modules[idx]
        return {
            "latent": _from_fp8_per_table(lut.latent_fp8, lut.latent_scale),
            "m": _from_fp8_per_table(s["m_fp8"], s["m_scale"]),
            "v": _from_fp8_per_table(s["v_fp8"], s["v_scale"]),
        }
