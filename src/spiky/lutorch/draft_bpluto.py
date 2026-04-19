"""DraftBPLUTO — pure-PyTorch reference optimizer for BitPermutationLUT.

A readable, un-fused, un-kerneled sibling of BitPermutationLUTOptimizer.
Purpose: pin down the math before optimizing. Every step is written the way
the paper would describe it; nothing is hidden behind a fused kernel.

Pipeline per step:
  1. `zero_grad()` is a no-op (hook state must survive forward → backward).
  2. A forward hook re-runs the anchor lookup to capture `lookup_indices`,
     and registers an output-grad hook to capture `grad_out`.
  3. `step()`:
       a. Project `grad_out` → per-entry `weight_grad` by scatter-add.
       b. Dequantize latent / m / v from fp8 (per-table scale) to float32.
       c. Standard Adam update (explicit mhat / vhat; no fused form).
       d. Safety-clamp latent to ±10.
       e. Requantize latent / m / v to fp8 with fresh per-table scale.
       f. Repack `bit_weights := sign(latent)`.
"""
import math
from typing import Callable, Iterable, List, Optional

import torch


_FP8 = getattr(torch, "float8_e4m3fn", None)
_FP8_AMAX = 448.0  # representable max of float8_e4m3fn


def _to_fp8_per_table(t_f32: torch.Tensor):
    """Quantize float32 → (fp8, scale) using per-table (dim 0) amax."""
    amax = t_f32.abs().amax(dim=tuple(range(1, t_f32.dim())), keepdim=True).clamp(min=1e-20)
    scale = _FP8_AMAX / amax
    return (t_f32 * scale).to(_FP8), scale


def _from_fp8_per_table(t_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return t_fp8.to(torch.float32) / scale


def _to_fp8_fixed(t_f32: torch.Tensor, mx: float = 1.0) -> torch.Tensor:
    """Quantize float32 → fp8 with a FIXED scale = 448/mx. Clamps to [-mx, mx]
    before casting. Used for latent, which stays in a bounded range throughout
    training — fixed scale keeps the fp8 quantum constant (no resolution
    degradation as amax drifts). Returns fp8 bytes only; caller stores the
    scale as a `[N, 1, 1]` constant buffer (`_FP8_AMAX/mx` replicated).
    """
    s = _FP8_AMAX / mx
    return (t_f32.clamp(-mx, mx) * s).to(_FP8)


def _from_fp8_fixed(t_fp8: torch.Tensor, mx: float = 1.0) -> torch.Tensor:
    s = _FP8_AMAX / mx
    return t_fp8.to(torch.float32) / s


def _to_fp8_per_lut(t_f32: torch.Tensor):
    """Quantize float32 → (fp8, scale) using a SINGLE global amax over the
    whole tensor. Returns scale shape [N, 1, 1] (broadcast of one scalar) so
    it plugs into the existing per-table dequant path (same shape as
    `_to_fp8_per_table`); every row carries the identical value. This is
    consistent with `lut.latent_scale`'s [N, 1, 1] buffer, so the existing
    soft-backward CUDA kernel works unchanged — "uniform per-table" is a
    special case of "per-table".
    """
    amax = t_f32.abs().amax().clamp(min=1e-20)
    scalar = _FP8_AMAX / amax
    scale = scalar.view(1, 1, 1).expand(t_f32.size(0), 1, 1).contiguous()
    return (t_f32 * scalar).to(_FP8), scale


def _to_fp8_per_pair(
    t_f32: torch.Tensor,                      # [N, td, nap]
    pair_id_per_slot: torch.Tensor,           # [N, nap] int64
    n_pairs_total: int,
):
    """Quantize with per-canonical-pair amax. Scale stored as [n_pairs_total]
    fp32. All weights feeding the same pair (across tables and entries) share
    one amax-based scale. `t_f32` is [N, td, nap].
    """
    per_slot_amax = t_f32.abs().amax(dim=1)                                # [N, nap]
    amax_pp = torch.zeros(n_pairs_total, device=t_f32.device, dtype=torch.float32)
    amax_pp.scatter_reduce_(
        0, pair_id_per_slot.reshape(-1), per_slot_amax.reshape(-1),
        reduce="amax", include_self=True,
    )
    amax_pp.clamp_(min=1e-20)
    scale_pp = _FP8_AMAX / amax_pp                                         # [n_pairs_total]
    scale_per_slot = scale_pp[pair_id_per_slot]                            # [N, nap]
    fp8 = (t_f32 * scale_per_slot.unsqueeze(1)).to(_FP8)
    return fp8, scale_pp


def _from_fp8_per_pair(
    t_fp8: torch.Tensor,                      # [N, td, nap]
    scale_pp: torch.Tensor,                   # [n_pairs_total]
    pair_id_per_slot: torch.Tensor,           # [N, nap] int64
) -> torch.Tensor:
    scale_per_slot = scale_pp[pair_id_per_slot]                            # [N, nap]
    return t_fp8.to(torch.float32) / scale_per_slot.unsqueeze(1)


def _project_grad_out_to_weight_grad(
    grad_out: torch.Tensor,            # [B, H, P] float
    lookup_indices: torch.Tensor,      # int16 [B, N]    (N = H*tph)
    pair_idx_per_slot: torch.Tensor,   # int32 [H, tph, output_nap]
    n_heads: int, tph: int, output_nap: int, table_dim: int,
    scale: float,
) -> torch.Tensor:
    """Project dominance-output gradient back to per-entry weight gradient.

    Pure PyTorch scatter-add:
      - Gather per-slot gradient from `grad_out` via `pair_idx_per_slot`.
      - Scatter into the (entry_main(b, n)) row of wg[n] using index_add_.

    Shape of returned weight_grad: [N, table_dim, output_nap] float32.
    Only entries actually looked up in this batch receive non-zero gradient.
    """
    B, N = lookup_indices.shape
    device = grad_out.device

    wg = torch.zeros(N, table_dim, output_nap, device=device, dtype=torch.float32)

    # For each (h, slot, k in output_nap): which canonical pair does this slot write?
    pair_flat = pair_idx_per_slot.reshape(n_heads, tph * output_nap).long()
    # Gather grad for each slot: g_slot[b, h, slot*nap + k] = grad_out[b, h, pair_flat[h, slot*nap + k]]
    g_slot = grad_out.gather(2, pair_flat.unsqueeze(0).expand(B, -1, -1)) * scale
    g_slot = g_slot.reshape(B, N, output_nap).to(torch.float32)

    # Scatter into wg[n, entry_main(b, n), :] += g_slot[b, n, :]
    entries = lookup_indices.long()                                     # [B, N]
    N_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)   # [B, N]
    flat_idx = (N_idx * table_dim + entries).reshape(-1)                # [B*N]
    wg.view(N * table_dim, output_nap).index_add_(0, flat_idx, g_slot.reshape(-1, output_nap))
    return wg


class DraftBPLUTO:
    """Pure-PyTorch Adam for `BitPermutationLUT` modules (reference impl).

    Args:
        bit_luts:             iterable of BitPermutationLUT modules to optimize.
        lr, beta1, beta2, eps: standard Adam hyper-parameters.
        lr_schedule_fn:       optional callable step -> multiplier on `lr`.

    This class mirrors the API of `BitPermutationLUTOptimizer` but contains
    no CUDA kernel dispatches and no fused math. It is meant to be read
    alongside standard Adam to verify the training recipe; performance will
    be much worse than the production optimizer.
    """

    def __init__(
        self,
        bit_luts: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
    ):
        if _FP8 is None:
            raise RuntimeError("DraftBPLUTO requires torch.float8_e4m3fn (PyTorch 2.1+)")
        self.modules = list(bit_luts)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_schedule_fn = lr_schedule_fn
        self._step_count = 0
        self._handles: List = []
        self._states: List[dict] = []

        for lut in self.modules:
            dev = lut.bit_weights.device
            N = lut.n_heads * lut.tph
            shape = (N, lut.table_dim, lut.output_nap)

            # m, v per-weight fp8 with per-table dynamic scale.
            m_fp8 = torch.zeros(shape, device=dev, dtype=_FP8)
            v_fp8 = torch.zeros(shape, device=dev, dtype=_FP8)
            m_scale = torch.full((N, 1, 1), _FP8_AMAX / 1e-20, device=dev, dtype=torch.float32)
            v_scale = m_scale.clone()

            # Force latent onto the fixed-scale grid (mx=1.0, scale=448 constant).
            latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
            lut.latent_fp8.copy_(_to_fp8_fixed(latent_f, mx=1.0))
            lut.latent_scale.fill_(_FP8_AMAX)        # 448 everywhere

            state = {
                "m_fp8": m_fp8, "m_scale": m_scale,
                "v_fp8": v_fp8, "v_scale": v_scale,
                "grad_out": None,
                "lookup_indices": None,
            }
            self._states.append(state)

            self._refresh_weights(lut)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    # --- hook ---
    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                # Re-run the anchor lookup on the forward input to capture
                # `lookup_indices`. (Cheap; avoids touching forward() internals.)
                with torch.no_grad():
                    li, _, _, _, _ = module.anchor(inputs[0])
                state["lookup_indices"] = li
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    # --- housekeeping ---
    @staticmethod
    def _refresh_weights(lut) -> None:
        """Pack bit_weights := sign(latent) using per-table scale from the lut."""
        latent_f32 = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
        lut.set_bit_weights_from_signs(latent_f32)

    def zero_grad(self) -> None:
        """No-op. `grad_out` and `lookup_indices` are filled by hooks during
        forward/backward and consumed by `step()`. Zeroing them here would
        wipe state captured by the forward hook when the usual order is
        zero_grad → forward → backward → step.
        """
        return

    def close(self) -> None:
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
        t = self._step_count
        bias1 = 1.0 - self.beta1 ** t
        bias2 = 1.0 - self.beta2 ** t

        for lut, state in zip(self.modules, self._states):
            go = state["grad_out"]
            li = state.get("lookup_indices")
            if go is None or li is None:
                continue

            # (a) dY -> dW projection.
            weight_grad = _project_grad_out_to_weight_grad(
                go, li, lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
            )
            # Non-finite grads → 0 (one bad batch should not poison fp8 state).
            torch.nan_to_num_(weight_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # (b) dequantize to float32.
            #   latent: fixed-scale (lut.latent_scale = 448 constant).
            #   m, v  : per-weight fp8 (per-table scale).
            latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
            m_f = _from_fp8_per_table(state["m_fp8"], state["m_scale"])
            v_f = _from_fp8_per_table(state["v_fp8"], state["v_scale"])

            # (c) standard Adam (per-weight m, v).
            m_f.mul_(self.beta1).add_(weight_grad, alpha=1.0 - self.beta1)
            v_f.mul_(self.beta2).addcmul_(weight_grad, weight_grad, value=1.0 - self.beta2)
            mhat = m_f / bias1
            vhat = v_f / bias2

            if self.weight_decay != 0.0:
                latent_f.mul_(1.0 - lr * self.weight_decay)
            latent_f -= lr * mhat / (vhat.sqrt() + self.eps)

            # (d) latent: FIXED-scale fp8 (mx=1.0). Implicit clamp to [-1, 1]
            # via `_to_fp8_fixed`. Fixed scale keeps fp8 quantum constant so
            # small updates don't get lost as amax drifts (per-table dynamic
            # scale regressed bitflip_clean convergence — see commit 202f237).
            lut.latent_fp8.copy_(_to_fp8_fixed(latent_f, mx=1.0))
            # lut.latent_scale stays at _FP8_AMAX (set in __init__), unchanged.

            # (e) m, v: per-LUT dynamic scale (single global amax per module).
            # Scale tensor is [N, 1, 1] but every row carries the same scalar,
            # so a future in-kernel quantize needs no per-table reduction.
            state["m_fp8"], state["m_scale"] = _to_fp8_per_lut(m_f)
            state["v_fp8"], state["v_scale"] = _to_fp8_per_lut(v_f)

            # (f) bit_weights := sign(latent).
            self._refresh_weights(lut)

            state["grad_out"] = None
            state["lookup_indices"] = None

    # --- introspection ---
    def state_as_float(self, idx: int = 0) -> dict:
        s = self._states[idx]
        lut = self.modules[idx]
        return {
            "latent": _from_fp8_per_table(lut.latent_fp8, lut.latent_scale),
            "m": _from_fp8_per_table(s["m_fp8"], s["m_scale"]),
            "v": _from_fp8_per_table(s["v_fp8"], s["v_scale"]),
        }


class DraftBPLUTOPerPair:
    """Per-pair Adam: m, v, latent_scale are scalars per canonical output pair.

    Sharing rule: all weight positions `(n, entry, k)` that write into the same
    canonical pair `(h, p)` share a single scalar `m`, `v`, and latent scale.
    For the bitflip_clean config that is ~8.4K weights per scalar, 496 scalars
    per head.

    Algorithm per step:
      a. Project `grad_out` → per-weight `weight_grad` [N, table_dim, nap].
      b. Aggregate to per-pair gradient: mean of `weight_grad` over all
         (n, entry, k) feeding each pair. (Mean, so group-size invariant.)
      c. Adam on the per-pair gradient → scalar update `u[pair]` per pair.
      d. Broadcast: every weight feeding pair `g` adds `u[g]` to its latent.
      e. Per-pair amax of latent → per-pair fp8 scale; requant `latent_fp8`.
      f. bit_weights := sign(latent).

    Storage summary:
      latent_fp8       — per-parameter (lives on lut), fp8
      latent_scale_pp  — per-pair, float32
      m, v             — per-pair, float32
      count_per_pair   — precomputed constant, float32
      pair_id_per_slot — precomputed constant, int64 [N, output_nap]
    """

    def __init__(
        self,
        bit_luts: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
    ):
        if _FP8 is None:
            raise RuntimeError("DraftBPLUTOPerPair requires torch.float8_e4m3fn")
        self.modules = list(bit_luts)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_schedule_fn = lr_schedule_fn
        self._step_count = 0
        self._handles: List = []
        self._states: List[dict] = []

        for lut in self.modules:
            dev = lut.bit_weights.device
            H, tph, nap, td = lut.n_heads, lut.tph, lut.output_nap, lut.table_dim
            N = H * tph
            P = lut.n_pairs                                  # per head
            n_pairs_total = H * P

            # Global pair id for each (n, k): pair_id_per_slot[n, k] ∈ [0, H*P).
            # Does NOT depend on `entry` — all table_dim entries of the same
            # (n, k) write into the same canonical pair.
            pair_local = lut.pair_idx_per_slot.long()        # [H, tph, nap]
            h_idx = torch.arange(H, device=dev).view(-1, 1, 1)
            pair_global = pair_local + h_idx * P             # [H, tph, nap]
            pair_id_per_slot = pair_global.reshape(N, nap).contiguous()

            # Weights per pair group: (# slots mapping to g) × table_dim.
            count_per_pair = torch.zeros(n_pairs_total, device=dev, dtype=torch.float32)
            count_per_pair.scatter_add_(
                0, pair_id_per_slot.reshape(-1),
                torch.full((N * nap,), float(td), device=dev, dtype=torch.float32),
            )

            # m, v in float32. One scalar per pair (per-head).
            m = torch.zeros(n_pairs_total, device=dev, dtype=torch.float32)
            v = torch.zeros(n_pairs_total, device=dev, dtype=torch.float32)

            # Requantize the module's latent with per-pair scale so subsequent
            # dequant paths are consistent. Starts from the module's per-table
            # quantized init.
            latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
            latent_scale_pp = self._compute_latent_scale_pp(
                latent_f, pair_id_per_slot, n_pairs_total,
            )
            self._requant_latent_to_lut(lut, latent_f, latent_scale_pp, pair_id_per_slot)

            state = {
                "m": m, "v": v,
                "latent_scale_pp": latent_scale_pp,          # [n_pairs_total] f32
                "pair_id_per_slot": pair_id_per_slot,        # [N, nap] int64
                "count_per_pair": count_per_pair,            # [n_pairs_total] f32
                "n_pairs_total": n_pairs_total,
                "grad_out": None,
                "lookup_indices": None,
            }
            self._states.append(state)
            self._refresh_weights(lut, state)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    # --- helpers ---
    @staticmethod
    def _compute_latent_scale_pp(
        latent_f: torch.Tensor,                # [N, td, nap] f32
        pair_id_per_slot: torch.Tensor,        # [N, nap] int64
        n_pairs_total: int,
    ) -> torch.Tensor:
        """Per-pair amax → fp8 scale. First reduce over `entry` (table_dim),
        then scatter-reduce over pair id. Correct because amax(amax(parts)) =
        amax(union).
        """
        per_slot_amax = latent_f.abs().amax(dim=1)           # [N, nap]
        amax_pp = torch.zeros(n_pairs_total, device=latent_f.device, dtype=torch.float32)
        amax_pp.scatter_reduce_(
            0, pair_id_per_slot.reshape(-1), per_slot_amax.reshape(-1),
            reduce="amax", include_self=True,
        )
        amax_pp.clamp_(min=1e-20)
        return _FP8_AMAX / amax_pp

    @staticmethod
    def _requant_latent_to_lut(
        lut,
        latent_f: torch.Tensor,                # [N, td, nap] f32
        latent_scale_pp: torch.Tensor,         # [n_pairs_total] f32
        pair_id_per_slot: torch.Tensor,        # [N, nap] int64
    ) -> None:
        """latent_fp8 := (latent_f * scale_pp[pair_id]).to(fp8). Broadcasts
        the per-pair scale across the table_dim axis (same scale for all
        entries of a given (n, k)).
        """
        scale_per_slot = latent_scale_pp[pair_id_per_slot]   # [N, nap]
        lut.latent_fp8.copy_((latent_f * scale_per_slot.unsqueeze(1)).to(_FP8))

    @staticmethod
    def _dequant_latent(
        lut,
        latent_scale_pp: torch.Tensor,
        pair_id_per_slot: torch.Tensor,
    ) -> torch.Tensor:
        scale_per_slot = latent_scale_pp[pair_id_per_slot]   # [N, nap]
        return lut.latent_fp8.to(torch.float32) / scale_per_slot.unsqueeze(1)

    # --- hook ---
    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                with torch.no_grad():
                    li, _, _, _, _ = module.anchor(inputs[0])
                state["lookup_indices"] = li
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    # --- housekeeping ---
    def _refresh_weights(self, lut, state: dict) -> None:
        """Pack bit_weights := sign(latent). Uses dequantized latent_f32."""
        latent_f = self._dequant_latent(
            lut, state["latent_scale_pp"], state["pair_id_per_slot"],
        )
        lut.set_bit_weights_from_signs(latent_f)

    def zero_grad(self) -> None:
        return

    def close(self) -> None:
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
        t = self._step_count
        bias1 = 1.0 - self.beta1 ** t
        bias2 = 1.0 - self.beta2 ** t

        for lut, state in zip(self.modules, self._states):
            go = state["grad_out"]
            li = state.get("lookup_indices")
            if go is None or li is None:
                continue

            # (a) per-weight grad.
            weight_grad = _project_grad_out_to_weight_grad(
                go, li, lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
            )
            torch.nan_to_num_(weight_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # (b) aggregate to per-pair gradient (mean over the group).
            # Treat the whole pair group as a single scalar parameter; its
            # gradient is the mean of per-weight gradients. Sum over entry
            # (table_dim) first, then scatter-add by pair id.
            wg_per_slot = weight_grad.sum(dim=1)              # [N, nap]
            pair_id_flat = state["pair_id_per_slot"].reshape(-1)
            g_sum = torch.zeros(
                state["n_pairs_total"], device=weight_grad.device, dtype=torch.float32,
            )
            g_sum.scatter_add_(0, pair_id_flat, wg_per_slot.reshape(-1))
            g_pair = g_sum / state["count_per_pair"]          # [n_pairs_total] f32

            # (c) Adam on the per-pair scalar gradient (single-parameter Adam).
            m, v = state["m"], state["v"]
            m.mul_(self.beta1).add_(g_pair, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g_pair, g_pair, value=1.0 - self.beta2)
            mhat = m / bias1
            vhat = v / bias2
            # Single-parameter semantics: the update magnitude per PAIR is
            # ~lr. When broadcast to all count_per_pair weights in the group,
            # divide by count so the net effect on the pair aggregate matches
            # what a single scalar parameter would see (~lr), not n*lr.
            u_pp = -lr * mhat / (vhat.sqrt() + self.eps)      # [n_pairs_total]
            u_pp = u_pp / state["count_per_pair"]

            # (d) broadcast scalar update to every weight in the pair group.
            latent_f = self._dequant_latent(
                lut, state["latent_scale_pp"], state["pair_id_per_slot"],
            )
            u_per_slot = u_pp[state["pair_id_per_slot"]]      # [N, nap]
            latent_f += u_per_slot.unsqueeze(1)               # broadcast across td
            latent_f.clamp_(-10.0, 10.0)

            # (e) fresh per-pair latent scale + fp8 requant.
            latent_scale_pp = self._compute_latent_scale_pp(
                latent_f, state["pair_id_per_slot"], state["n_pairs_total"],
            )
            state["latent_scale_pp"] = latent_scale_pp
            self._requant_latent_to_lut(
                lut, latent_f, latent_scale_pp, state["pair_id_per_slot"],
            )

            # (f) bit_weights := sign(latent).
            lut.set_bit_weights_from_signs(latent_f)

            state["grad_out"] = None
            state["lookup_indices"] = None

    # --- introspection ---
    def state_as_float(self, idx: int = 0) -> dict:
        s = self._states[idx]
        lut = self.modules[idx]
        return {
            "latent": self._dequant_latent(lut, s["latent_scale_pp"], s["pair_id_per_slot"]),
            "m": s["m"],
            "v": s["v"],
            "latent_scale_pp": s["latent_scale_pp"],
        }


class DraftBPLUTOPerTable:
    """Per-table Adam: m, v, latent_scale are scalars per table (n).

    Sharing rule: all weight positions `(n, entry, k)` that share table index
    `n` share a single scalar `m`, `v`, and latent scale. Group size is
    `table_dim * output_nap` weights (2048 for bitflip_clean).

    Same "single-parameter" semantics as DraftBPLUTOPerPair:
      - per-table gradient = mean of per-weight gradients
      - Adam on the scalar → u_pt ~ lr
      - u_pt / count_per_table broadcast to every weight in the table
    """

    def __init__(
        self,
        bit_luts: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
    ):
        if _FP8 is None:
            raise RuntimeError("DraftBPLUTOPerTable requires torch.float8_e4m3fn")
        self.modules = list(bit_luts)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_schedule_fn = lr_schedule_fn
        self._step_count = 0
        self._handles: List = []
        self._states: List[dict] = []

        for lut in self.modules:
            dev = lut.bit_weights.device
            H, tph, nap, td = lut.n_heads, lut.tph, lut.output_nap, lut.table_dim
            N = H * tph
            count_per_table = float(td * nap)

            m = torch.zeros(N, device=dev, dtype=torch.float32)
            v = torch.zeros(N, device=dev, dtype=torch.float32)

            # Start from the module's per-table quantized latent (already
            # per-table scale — no requant needed; we just adopt it).
            latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
            latent_scale_pt = self._compute_latent_scale_pt(latent_f)
            self._requant_latent_to_lut(lut, latent_f, latent_scale_pt)

            state = {
                "m": m, "v": v,
                "latent_scale_pt": latent_scale_pt,   # [N] f32
                "count_per_table": count_per_table,   # scalar
                "N": N,
                "grad_out": None,
                "lookup_indices": None,
            }
            self._states.append(state)
            self._refresh_weights(lut, state)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    # --- helpers ---
    @staticmethod
    def _compute_latent_scale_pt(latent_f: torch.Tensor) -> torch.Tensor:
        """Per-table amax → fp8 scale. latent_f shape [N, td, nap]."""
        amax = latent_f.abs().amax(dim=(1, 2)).clamp(min=1e-20)   # [N]
        return _FP8_AMAX / amax

    @staticmethod
    def _requant_latent_to_lut(
        lut,
        latent_f: torch.Tensor,            # [N, td, nap] f32
        latent_scale_pt: torch.Tensor,     # [N] f32
    ) -> None:
        scale_b = latent_scale_pt.view(-1, 1, 1)                   # [N, 1, 1]
        lut.latent_fp8.copy_((latent_f * scale_b).to(_FP8))
        lut.latent_scale.copy_(scale_b)                            # keep lut state consistent

    @staticmethod
    def _dequant_latent(lut, latent_scale_pt: torch.Tensor) -> torch.Tensor:
        scale_b = latent_scale_pt.view(-1, 1, 1)
        return lut.latent_fp8.to(torch.float32) / scale_b

    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                with torch.no_grad():
                    li, _, _, _, _ = module.anchor(inputs[0])
                state["lookup_indices"] = li
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    def _refresh_weights(self, lut, state: dict) -> None:
        latent_f = self._dequant_latent(lut, state["latent_scale_pt"])
        lut.set_bit_weights_from_signs(latent_f)

    def zero_grad(self) -> None:
        return

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def step(self) -> None:
        self._step_count += 1
        lr = self.lr
        if self.lr_schedule_fn is not None:
            lr = lr * self.lr_schedule_fn(self._step_count)
        t = self._step_count
        bias1 = 1.0 - self.beta1 ** t
        bias2 = 1.0 - self.beta2 ** t

        for lut, state in zip(self.modules, self._states):
            go = state["grad_out"]
            li = state.get("lookup_indices")
            if go is None or li is None:
                continue

            # (a) per-weight grad.
            weight_grad = _project_grad_out_to_weight_grad(
                go, li, lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
            )
            torch.nan_to_num_(weight_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # (b) per-table gradient: mean over (entry, k). No scatter needed.
            g_per_table = weight_grad.mean(dim=(1, 2))        # [N]

            # (c) Adam on the per-table scalar gradient.
            m, v = state["m"], state["v"]
            m.mul_(self.beta1).add_(g_per_table, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g_per_table, g_per_table, value=1.0 - self.beta2)
            mhat = m / bias1
            vhat = v / bias2
            u_pt = -lr * mhat / (vhat.sqrt() + self.eps)      # [N]

            # (d) broadcast scalar update to every weight in the table.
            latent_f = self._dequant_latent(lut, state["latent_scale_pt"])
            latent_f += u_pt.view(-1, 1, 1)
            latent_f.clamp_(-10.0, 10.0)

            # (e) per-table requant.
            latent_scale_pt = self._compute_latent_scale_pt(latent_f)
            state["latent_scale_pt"] = latent_scale_pt
            self._requant_latent_to_lut(lut, latent_f, latent_scale_pt)

            # (f) bit_weights := sign(latent).
            lut.set_bit_weights_from_signs(latent_f)

            state["grad_out"] = None
            state["lookup_indices"] = None

    def state_as_float(self, idx: int = 0) -> dict:
        s = self._states[idx]
        lut = self.modules[idx]
        return {
            "latent": self._dequant_latent(lut, s["latent_scale_pt"]),
            "m": s["m"],
            "v": s["v"],
            "latent_scale_pt": s["latent_scale_pt"],
        }


class DraftBPLUTOPerSlot:
    """Per-(table, output_k) Adam: m, v, latent_scale shape [N, output_nap].

    Sharing rule: all `table_dim` entries at a given (n, k) share a single
    scalar m, v, and latent scale. Group size = table_dim (64 for
    bitflip_clean). # independent knobs = N * output_nap = 65K.

    Within a group, only one entry is looked up per sample (anchor lookup
    picks entry_main(b, n)), so ~1/table_dim of the weights receive grad per
    sample. Aggregation: mean over entries.
    """

    def __init__(
        self,
        bit_luts: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
    ):
        if _FP8 is None:
            raise RuntimeError("DraftBPLUTOPerSlot requires torch.float8_e4m3fn")
        self.modules = list(bit_luts)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_schedule_fn = lr_schedule_fn
        self._step_count = 0
        self._handles: List = []
        self._states: List[dict] = []

        for lut in self.modules:
            dev = lut.bit_weights.device
            H, tph, nap, td = lut.n_heads, lut.tph, lut.output_nap, lut.table_dim
            N = H * tph

            m = torch.zeros(N, nap, device=dev, dtype=torch.float32)
            v = torch.zeros(N, nap, device=dev, dtype=torch.float32)

            latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
            latent_scale_ps = self._compute_latent_scale_ps(latent_f)
            self._requant_latent_to_lut(lut, latent_f, latent_scale_ps)

            state = {
                "m": m, "v": v,
                "latent_scale_ps": latent_scale_ps,   # [N, nap] f32
                "count_per_slot": float(td),
                "grad_out": None,
                "lookup_indices": None,
            }
            self._states.append(state)
            self._refresh_weights(lut, state)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    @staticmethod
    def _compute_latent_scale_ps(latent_f: torch.Tensor) -> torch.Tensor:
        """Per-(n, k) amax over entries. latent_f shape [N, td, nap]."""
        amax = latent_f.abs().amax(dim=1).clamp(min=1e-20)     # [N, nap]
        return _FP8_AMAX / amax

    @staticmethod
    def _requant_latent_to_lut(
        lut,
        latent_f: torch.Tensor,                # [N, td, nap]
        latent_scale_ps: torch.Tensor,         # [N, nap]
    ) -> None:
        scale_b = latent_scale_ps.unsqueeze(1)                 # [N, 1, nap]
        lut.latent_fp8.copy_((latent_f * scale_b).to(_FP8))

    @staticmethod
    def _dequant_latent(lut, latent_scale_ps: torch.Tensor) -> torch.Tensor:
        scale_b = latent_scale_ps.unsqueeze(1)
        return lut.latent_fp8.to(torch.float32) / scale_b

    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                with torch.no_grad():
                    li, _, _, _, _ = module.anchor(inputs[0])
                state["lookup_indices"] = li
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    def _refresh_weights(self, lut, state: dict) -> None:
        latent_f = self._dequant_latent(lut, state["latent_scale_ps"])
        lut.set_bit_weights_from_signs(latent_f)

    def zero_grad(self) -> None:
        return

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def step(self) -> None:
        self._step_count += 1
        lr = self.lr
        if self.lr_schedule_fn is not None:
            lr = lr * self.lr_schedule_fn(self._step_count)
        t = self._step_count
        bias1 = 1.0 - self.beta1 ** t
        bias2 = 1.0 - self.beta2 ** t

        for lut, state in zip(self.modules, self._states):
            go = state["grad_out"]
            li = state.get("lookup_indices")
            if go is None or li is None:
                continue

            weight_grad = _project_grad_out_to_weight_grad(
                go, li, lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
            )
            torch.nan_to_num_(weight_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # Mean over entry dim (table_dim) → per-slot gradient.
            g_per_slot = weight_grad.mean(dim=1)              # [N, nap]

            m, v = state["m"], state["v"]
            m.mul_(self.beta1).add_(g_per_slot, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g_per_slot, g_per_slot, value=1.0 - self.beta2)
            mhat = m / bias1
            vhat = v / bias2
            u_ps = -lr * mhat / (vhat.sqrt() + self.eps)      # [N, nap]

            # Broadcast scalar update across entries (table_dim axis).
            latent_f = self._dequant_latent(lut, state["latent_scale_ps"])
            latent_f += u_ps.unsqueeze(1)                     # [N, 1, nap]
            latent_f.clamp_(-10.0, 10.0)

            latent_scale_ps = self._compute_latent_scale_ps(latent_f)
            state["latent_scale_ps"] = latent_scale_ps
            self._requant_latent_to_lut(lut, latent_f, latent_scale_ps)
            lut.set_bit_weights_from_signs(latent_f)

            state["grad_out"] = None
            state["lookup_indices"] = None

    def state_as_float(self, idx: int = 0) -> dict:
        s = self._states[idx]
        lut = self.modules[idx]
        return {
            "latent": self._dequant_latent(lut, s["latent_scale_ps"]),
            "m": s["m"],
            "v": s["v"],
            "latent_scale_ps": s["latent_scale_ps"],
        }


class DraftBPLUTOVPerPair:
    """Hybrid: m per-weight (fp8, per-table scale), v per-pair scalar,
    latent scale per-table (one scalar per table — lut.latent_scale [N,1,1]).

    Rationale:
      - m: per-weight fp8 with per-table amax scale.
      - v: per-pair scalar fp32. Shared adaptive-lr across a pair group.
      - latent_scale: per-table (same granularity as m). Per-LUT scale is
        catastrophic here because per-pair v already makes small-|g| weights
        move slowly; combined with coarse latent scale they underflow fp8.

    Storage:
      latent_fp8       — per-weight fp8 (on lut)
      lut.latent_scale — per-table fp32 [N, 1, 1]
      m_fp8            — per-weight fp8 [N, td, nap]
      m_scale          — per-table fp32 [N, 1, 1]
      v_pp             — per-pair fp32 [n_pairs_total]

    Adam update (per weight w feeding pair p):
      latent[w] ← (1 - lr·wd)·latent[w]                           # AdamW
      latent[w] ← latent[w] - lr · mhat[w] / (sqrt(vhat[p]) + eps)
    """

    def __init__(
        self,
        bit_luts: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
    ):
        if _FP8 is None:
            raise RuntimeError("DraftBPLUTOVPerPair requires torch.float8_e4m3fn")
        self.modules = list(bit_luts)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_schedule_fn = lr_schedule_fn
        self._step_count = 0
        self._handles: List = []
        self._states: List[dict] = []

        for lut in self.modules:
            dev = lut.bit_weights.device
            H, tph, nap, td = lut.n_heads, lut.tph, lut.output_nap, lut.table_dim
            N = H * tph
            P = lut.n_pairs
            n_pairs_total = H * P

            pair_local = lut.pair_idx_per_slot.long()
            h_idx = torch.arange(H, device=dev).view(-1, 1, 1)
            pair_id_per_slot = (pair_local + h_idx * P).reshape(N, nap).contiguous()

            count_per_pair = torch.zeros(n_pairs_total, device=dev, dtype=torch.float32)
            count_per_pair.scatter_add_(
                0, pair_id_per_slot.reshape(-1),
                torch.full((N * nap,), float(td), device=dev, dtype=torch.float32),
            )

            # m per-weight fp8 with per-table scale; v per-pair scalar.
            m_fp8 = torch.zeros((N, td, nap), device=dev, dtype=_FP8)
            m_scale = torch.full(
                (N, 1, 1), _FP8_AMAX / 1e-20, device=dev, dtype=torch.float32,
            )
            v_pp = torch.zeros(n_pairs_total, device=dev, dtype=torch.float32)

            # Initial latent: keep the module's per-table fp8 / scale as-is
            # (BitPermutationLUT already stores them consistent at init).
            latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)

            state = {
                "m_fp8": m_fp8,
                "m_scale": m_scale,
                "v_pp": v_pp,
                "pair_id_per_slot": pair_id_per_slot,
                "count_per_pair": count_per_pair,
                "n_pairs_total": n_pairs_total,
                "grad_out": None,
                "lookup_indices": None,
            }
            self._states.append(state)
            lut.set_bit_weights_from_signs(latent_f)
            self._handles.append(lut.register_forward_hook(self._make_hook(state)))

    @staticmethod
    def _make_hook(state: dict):
        def hook(module, inputs, output):
            if output.requires_grad:
                with torch.no_grad():
                    li, _, _, _, _ = module.anchor(inputs[0])
                state["lookup_indices"] = li
                output.register_hook(lambda g: state.__setitem__("grad_out", g.detach()))
        return hook

    def zero_grad(self) -> None:
        return

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def step(self) -> None:
        self._step_count += 1
        lr = self.lr
        if self.lr_schedule_fn is not None:
            lr = lr * self.lr_schedule_fn(self._step_count)
        t = self._step_count
        bias1 = 1.0 - self.beta1 ** t
        bias2 = 1.0 - self.beta2 ** t

        for lut, state in zip(self.modules, self._states):
            go = state["grad_out"]
            li = state.get("lookup_indices")
            if go is None or li is None:
                continue

            pair_id = state["pair_id_per_slot"]                  # [N, nap]
            pair_id_flat = pair_id.reshape(-1)

            # (a) per-weight gradient.
            weight_grad = _project_grad_out_to_weight_grad(
                go, li, lut.pair_idx_per_slot,
                lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
            )
            torch.nan_to_num_(weight_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # (b) m: per-weight EMA of gradient (per-table fp8 scale).
            m_f = _from_fp8_per_table(state["m_fp8"], state["m_scale"])
            m_f.mul_(self.beta1).add_(weight_grad, alpha=1.0 - self.beta1)

            # (c) v: per-pair scalar EMA of mean(g^2) over the group.
            # Equivalent (by EMA linearity) to per-weight EMA then mean, but
            # at scalar storage cost.
            wg2_per_slot = (weight_grad * weight_grad).sum(dim=1)  # [N, nap]
            g2_sum = torch.zeros(
                state["n_pairs_total"], device=weight_grad.device, dtype=torch.float32,
            )
            g2_sum.scatter_add_(0, pair_id_flat, wg2_per_slot.reshape(-1))
            g2_pair = g2_sum / state["count_per_pair"]            # mean(g^2) per pair
            v_pp = state["v_pp"]
            v_pp.mul_(self.beta2).add_(g2_pair, alpha=1.0 - self.beta2)

            # (d) bias correction + update.
            mhat_f = m_f / bias1                                  # per-weight
            vhat_pp = v_pp / bias2                                # per-pair
            denom_per_slot = (vhat_pp.sqrt() + self.eps)[pair_id] # [N, nap]

            # Dequant latent using per-LUT scale stored on the module.
            latent_f = _from_fp8_per_table(lut.latent_fp8, lut.latent_scale)
            # AdamW-style decoupled weight decay on latent (keeps magnitudes
            # near 0, which is also what makes a single per-LUT scale work).
            if self.weight_decay != 0.0:
                latent_f.mul_(1.0 - lr * self.weight_decay)
            latent_f += -lr * mhat_f / denom_per_slot.unsqueeze(1)
            # Safety clamp: fp8_e4m3 max magnitude is ~448; ±256 leaves headroom.
            latent_f.clamp_(-256.0, 256.0)

            # (e) requant with fresh per-table scales for both latent and m.
            lat_fp8, lat_scale = _to_fp8_per_table(latent_f)
            lut.latent_fp8.copy_(lat_fp8)
            lut.latent_scale.copy_(lat_scale)

            state["m_fp8"], state["m_scale"] = _to_fp8_per_table(m_f)

            # (f) bit_weights := sign(latent).
            lut.set_bit_weights_from_signs(latent_f)

            state["grad_out"] = None
            state["lookup_indices"] = None

    def state_as_float(self, idx: int = 0) -> dict:
        s = self._states[idx]
        lut = self.modules[idx]
        return {
            "latent": _from_fp8_per_table(lut.latent_fp8, lut.latent_scale),
            "m": _from_fp8_per_table(s["m_fp8"], s["m_scale"]),
            "v_pp": s["v_pp"],
            "latent_scale": lut.latent_scale,
        }
