"""
Winner-Take-All lookup implementation.
"""
import os
import torch
import torch.nn as nn
from typing import Optional, Tuple

from spiky.lutorch.lut_helpers import UncertaintyMode

# Optional torch.compile; set SPIKY_LUTORCH_NO_COMPILE=1 to disable (e.g. debugging or older PyTorch).
_USE_LUTORCH_COMPILE = os.environ.get("SPIKY_LUTORCH_NO_COMPILE", "0") != "1"
# Optional native CUDA kernels in lutorch; set SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS=1 to disable.
_USE_LUTORCH_CUSTOM_CUDA_KERNELS = os.environ.get("SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS", "0") != "1"
_LUTORCH_CUDA_THREADS_PER_BLOCK = int(os.environ.get("SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK", "256"))
if _LUTORCH_CUDA_THREADS_PER_BLOCK < 1 or _LUTORCH_CUDA_THREADS_PER_BLOCK > 1024:
    raise ValueError(
        "SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK must be in range [1, 1024], "
        f"got {_LUTORCH_CUDA_THREADS_PER_BLOCK}"
    )


def _get_native_lutorch_manager():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


def _first_tensor_device(args, kwargs):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.device
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return None


def _maybe_compile(fn):
    # Use torch.compile only when LUTorch compile is enabled and the current
    # call uses CUDA tensors. On CPU, torch.compile can introduce heavy
    # multiprocessing overhead with little benefit.
    compiled_fn = None
    if _USE_LUTORCH_COMPILE and hasattr(torch, "compile"):
        compiled_fn = torch.compile(fn, dynamic=True)

    def wrapper(*args, **kwargs):
        device = _first_tensor_device(args, kwargs)
        if device is not None and device.type == "cuda" and compiled_fn is not None and _USE_LUTORCH_COMPILE:
            return compiled_fn(*args, **kwargs)
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_maybe_compile
def _wta_lookup_forward_fallback(x: torch.Tensor, n_alternatives: int):
    """
    Pure-PyTorch WTA forward. Returns flat tensors for save_for_backward.

    Returns:
        winner_inds_flat: int64 [BC]
        alt_inds_flat:    int64 [BC, n_alternatives]
        alt_deltas_flat:  float [BC, n_alternatives]
    """
    B, C, N = x.shape
    BC = B * C
    topk_vals, topk_inds = torch.topk(
        x.reshape(BC, N), k=n_alternatives + 1, dim=-1, largest=True
    )
    winner_inds_flat = topk_inds[:, 0].contiguous()
    alt_inds_flat    = topk_inds[:, 1:].contiguous()
    alt_deltas_flat  = (topk_vals[:, 0:1] - topk_vals[:, 1:]).contiguous()
    return winner_inds_flat, alt_inds_flat, alt_deltas_flat


class WTALookup(nn.Module):
    """
    Winner-Take-All lookup for channeled input.

    Processes input of shape [B, C, n_inputs]. Within each channel, finds the
    winner (argmax) index and n_alternatives runner-up indices via topk.
    Returns integer index tensors and gradient carriers that allow smooth
    gradient propagation back to the input.

    Args:
        n_inputs: Size of the N dimension (number of candidates per channel).
        n_alternatives: Number of runner-up indices per channel.
        uncertainty_mode: Uncertainty function for gradient weighting.
    """

    def __init__(
        self,
        n_inputs: int,
        n_alternatives: int = 1,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
    ):
        super().__init__()
        if n_alternatives < 1:
            raise ValueError(f"n_alternatives must be >= 1, got {n_alternatives}")
        self.n_inputs = n_inputs
        self.n_alternatives = n_alternatives
        self.uncertainty_mode = uncertainty_mode
        self._cached_batch_offset: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, n_inputs].

        Returns:
            In training mode:
                - lookup_indices: int64 [B, C]
                - lookup_alt_indices: int64 [B, C, n_alternatives]
                - lookup_alt_deltas: float [B, C, n_alternatives]
                - lookup_indices_grad_c: float [B, C]
                - lookup_alt_indices_grad_c: float [B, C, n_alternatives]
            In eval mode:
                - lookup_indices: int64 [B, C]
                - lookup_alt_indices: int64 [B, C, n_alternatives]
                - lookup_alt_deltas: float [B, C, n_alternatives]
        """
        if self.training:
            return self._forward_train(x)
        return self._forward_eval(x)

    def _forward_eval(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_native = (
            _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
            and _get_native_lutorch_manager() is not None
            and self.n_alternatives in (1, 2, 3)
        )
        if use_native:
            return self._native_forward(x)
        B, C, N = x.shape
        wi, ai, ad = _wta_lookup_forward_fallback(x, self.n_alternatives)
        return wi.view(B, C), ai.view(B, C, self.n_alternatives), ad.view(B, C, self.n_alternatives)

    def _forward_train(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, _ = x.shape
        expected_len = B * C * self.n_alternatives
        if (
            self._cached_batch_offset is None
            or self._cached_batch_offset.numel() != expected_len
            or self._cached_batch_offset.device != x.device
        ):
            # offset[bc * n_alternatives + k] = bc * n_inputs, for bc in range(B*C)
            self._cached_batch_offset = (
                torch.arange(B * C, device=x.device, dtype=torch.long)
                .repeat_interleave(self.n_alternatives) * self.n_inputs
            ).contiguous()

        uncertainty_mode_int = 0 if self.uncertainty_mode == UncertaintyMode.INVERSE_L1 else 1
        return WTALookupFunction.apply(
            x, self.n_alternatives, uncertainty_mode_int, self._cached_batch_offset
        )

    def _native_forward(self, x: torch.Tensor):
        """Call the native CUDA kernel for forward pass (na1/na2/na3)."""
        native = _get_native_lutorch_manager()
        if self.n_alternatives == 1:
            winner_inds, alt_inds, alt_deltas = native.wta_lookup_forward_na1(
                x, _LUTORCH_CUDA_THREADS_PER_BLOCK
            )
        elif self.n_alternatives == 2:
            winner_inds, alt_inds, alt_deltas = native.wta_lookup_forward_na2(
                x, _LUTORCH_CUDA_THREADS_PER_BLOCK
            )
        else:
            winner_inds, alt_inds, alt_deltas = native.wta_lookup_forward_na3(
                x, _LUTORCH_CUDA_THREADS_PER_BLOCK
            )
        return winner_inds, alt_inds, alt_deltas


class WTALookupFunction(torch.autograd.Function):
    """Custom autograd function for WTA lookup with smooth gradient propagation."""

    @staticmethod
    def forward(ctx, x, n_alternatives, uncertainty_mode_int, batch_offset):
        """
        Args:
            x: float [B, C, N]
            n_alternatives: int
            uncertainty_mode_int: 0 = INVERSE_L1, 1 = INVERSE_QUADRATIC
            batch_offset: int64 [B*C*n_alternatives]

        Returns:
            lookup_indices: int64 [B, C]
            lookup_alt_indices: int64 [B, C, n_alternatives]
            lookup_alt_deltas: float [B, C, n_alternatives]  (winner_val - alt_val, always >= 0)
            lookup_indices_grad_c: float [B, C]
            lookup_alt_indices_grad_c: float [B, C, n_alternatives]
        """
        B, C, N = x.shape
        BC = B * C

        use_native = (
            _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
            and _get_native_lutorch_manager() is not None
            and n_alternatives in (1, 2, 3)
        )
        if use_native:
            native = _get_native_lutorch_manager()
            if n_alternatives == 1:
                winner_inds, alt_inds, alt_deltas = native.wta_lookup_forward_na1(
                    x, _LUTORCH_CUDA_THREADS_PER_BLOCK
                )
            elif n_alternatives == 2:
                winner_inds, alt_inds, alt_deltas = native.wta_lookup_forward_na2(
                    x, _LUTORCH_CUDA_THREADS_PER_BLOCK
                )
            else:
                winner_inds, alt_inds, alt_deltas = native.wta_lookup_forward_na3(
                    x, _LUTORCH_CUDA_THREADS_PER_BLOCK
                )
            # winner_inds: [B, C]; alt_inds: [B, C, n_alt]; alt_deltas: [B, C, n_alt]
            # Flatten to [BC] / [BC, n_alt] for save_for_backward
            winner_inds_flat = winner_inds.view(BC).contiguous()
            alt_inds_flat    = alt_inds.view(BC, n_alternatives).contiguous()
            alt_deltas_flat  = alt_deltas.view(BC, n_alternatives).contiguous()
        else:
            winner_inds_flat, alt_inds_flat, alt_deltas_flat = _wta_lookup_forward_fallback(
                x, n_alternatives
            )
            winner_inds = winner_inds_flat.view(B, C)
            alt_inds    = alt_inds_flat.view(B, C, n_alternatives)
            alt_deltas  = alt_deltas_flat.view(B, C, n_alternatives)

        z = x.sum() * 0
        lookup_indices_grad_c     = z.expand(B, C)
        lookup_alt_indices_grad_c = z.expand(B, C, n_alternatives)

        ctx.inv_l1          = (uncertainty_mode_int == 0)
        ctx.batch_offset    = batch_offset
        ctx.B               = B
        ctx.C               = C
        ctx.N               = N
        ctx.n_alternatives  = n_alternatives
        ctx.save_for_backward(x, winner_inds_flat, alt_inds_flat, alt_deltas_flat)

        return (
            winner_inds,
            alt_inds,
            alt_deltas,
            lookup_indices_grad_c,
            lookup_alt_indices_grad_c,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Propagates gradients back to x via the uncertainty function.

        Only grad_lookup_indices_grad_c and grad_lookup_alt_indices_grad_c carry
        meaningful gradients (the integer index outputs have no gradient path).

        Gradient rule mirrors AnchorPairsLookup:
            du[b,c,k] = grad_diff[b,c,k] * d(uncertainty)/d(delta[b,c,k])
            x_grad[b, c, winner_idx]  +=  du[b,c,k]  (summed over k)
            x_grad[b, c, alt_idx[k]] -=  du[b,c,k]
        """
        (
            _,
            _,
            _,
            grad_lookup_indices_grad_c,
            grad_lookup_alt_indices_grad_c,
        ) = grad_outputs

        x, winner_inds_flat, alt_inds_flat, alt_deltas_flat = ctx.saved_tensors
        B, C, N = ctx.B, ctx.C, ctx.N
        BC = B * C
        n_alternatives = ctx.n_alternatives

        use_native = (
            _USE_LUTORCH_CUSTOM_CUDA_KERNELS
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float64)
            and _get_native_lutorch_manager() is not None
            and n_alternatives in (1, 2, 3)
        )
        if use_native:
            native = _get_native_lutorch_manager()
            # winner_ids: [BC*n_alt] — winner index repeated once per alternative
            winner_ids_flat = winner_inds_flat.repeat_interleave(n_alternatives).contiguous()
            alt_ids_flat    = alt_inds_flat.reshape(-1).contiguous()
            alt_deltas_bc   = alt_deltas_flat.reshape(-1).contiguous()
            grad_alt_flat   = grad_lookup_alt_indices_grad_c.reshape(-1).contiguous()

            x_grad_flat = native.wta_lookup_backward(
                x.contiguous(),
                winner_ids_flat, alt_ids_flat, alt_deltas_bc,
                ctx.batch_offset,
                grad_lookup_indices_grad_c.contiguous(),
                grad_alt_flat,
                n_alternatives, ctx.inv_l1, _LUTORCH_CUDA_THREADS_PER_BLOCK,
            )
            return x_grad_flat.view(B, C, N), None, None, None

        def _wta_lookup_backward_impl(
            x,
            winner_inds_flat,
            alt_inds_flat,
            alt_deltas_flat,
            batch_offset,
            grad_main,
            grad_alt,
            inv_l1,
            n_alternatives,
        ):
            BC = winner_inds_flat.shape[0]
            N = x.shape[2]
            alt_deltas_bca = alt_deltas_flat.view(BC, n_alternatives)

            grad_diff = grad_main.unsqueeze(2) - grad_alt   # [B, C, n_alternatives]

            if inv_l1:
                abs_delta = alt_deltas_bca.abs()
                one_plus_abs = 1.0 + abs_delta
                minus_uncertainty_derivative = (
                    0.5 * alt_deltas_bca.sign() / (one_plus_abs * one_plus_abs)
                )
            else:
                delta_sq = alt_deltas_bca * alt_deltas_bca
                one_plus_sq = 1.0 + delta_sq
                minus_uncertainty_derivative = alt_deltas_bca / (one_plus_sq * one_plus_sq)

            du = grad_diff.view(BC, n_alternatives) * minus_uncertainty_derivative
            if n_alternatives > 1:
                du = du / n_alternatives

            winner_repeated = winner_inds_flat.repeat_interleave(n_alternatives)
            alt_flat        = alt_inds_flat.reshape(-1)
            du_flat         = du.reshape(-1)

            x_grad_flat = torch.zeros(BC * N, device=x.device, dtype=x.dtype)
            x_grad_flat.scatter_add_(0, batch_offset + winner_repeated, du_flat)
            x_grad_flat.scatter_add_(0, batch_offset + alt_flat, -du_flat)
            return x_grad_flat

        _wta_lookup_backward_impl = _maybe_compile(_wta_lookup_backward_impl)

        x_grad_flat = _wta_lookup_backward_impl(
            x,
            winner_inds_flat,
            alt_inds_flat,
            alt_deltas_flat,
            ctx.batch_offset,
            grad_lookup_indices_grad_c,
            grad_lookup_alt_indices_grad_c,
            ctx.inv_l1,
            n_alternatives,
        )

        return x_grad_flat.view(B, C, N), None, None, None
