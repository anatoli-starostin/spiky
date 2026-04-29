"""TinyMultiHeadLutOptimizer — AdamW for TinyMultiHeadLut weights.

Mirrors BitPermutationLUTOptimizer's design pattern but for fp32-style real-valued
weights. Adam moments (m, v) are stored in the same dtype as the LUT weights
(default bf16) for ~2× memory savings vs PyTorch's stock AdamW which always
keeps state in fp32.

Compute is performed in fp32 (cast on the fly) for numerical stability:
  - bf16 has fp32-equivalent exponent range, so no underflow on (1-β2)·grad²
  - bf16's 7-bit mantissa is sufficient for Adam moments in practice
  - Casting to fp32 for sqrt / division avoids precision artifacts in the update

For pure-bf16 compute (skipping fp32 casts), set `compute_dtype=torch.bfloat16`
in step(). Default mode keeps storage in bf16 + compute in fp32.

Manages only TinyMultiHeadLut modules. Other params should go through standard
AdamW.
"""
from typing import Iterable, Optional, Callable, List

import torch


def _stochastic_round_fp32_to_fp16(x_fp32: torch.Tensor) -> torch.Tensor:
    """Unbiased stochastic-round fp32 → fp16.

    fp16 is not just the high 16 bits of fp32 (different exponent/mantissa
    split), so the bf16 bit-trick doesn't apply. We use the standard
    nextafter-bracket method instead:
      1. RNE-cast x to fp16 → "lo" (one of the two bracketing fp16 values).
      2. Compute "hi" = nextafter(lo, ±∞) on the side x leans toward.
      3. p = (x - lo) / (hi - lo) ∈ [0, 1] = how close x is to hi.
      4. Sample uniform r ∈ [0, 1); return hi if r < p else lo.

    E[output] = lo·(1-p) + hi·p = lo + p·(hi-lo) = lo + (x-lo) = x.   ✓
    """
    if x_fp32.dtype != torch.float32:
        x_fp32 = x_fp32.float()
    lo = x_fp32.to(torch.float16)
    lo_f = lo.float()
    diff = x_fp32 - lo_f
    inf_pos = torch.tensor(float("inf"),  dtype=torch.float16, device=x_fp32.device)
    inf_neg = torch.tensor(float("-inf"), dtype=torch.float16, device=x_fp32.device)
    hi = torch.where(diff >= 0, torch.nextafter(lo, inf_pos),
                                 torch.nextafter(lo, inf_neg))
    span = hi.float() - lo_f
    # When x exactly hits an fp16 grid point, span = 0; just return lo.
    frac = torch.where(span != 0, diff / span, torch.zeros_like(diff))
    rand = torch.rand_like(frac)
    return torch.where(rand < frac, hi, lo)


def _stochastic_round_fp32_to_bf16(x_fp32: torch.Tensor) -> torch.Tensor:
    """Unbiased stochastic-round fp32 → bf16.

    bf16 occupies the high 16 bits of fp32; the low 16 mantissa bits are dropped.
    Round-to-nearest-even (PyTorch default) is biased for repeated small updates
    when the update is sub-ULP — many alternating-sign updates accumulate to
    nothing because each one rounds back to the same value.

    Stochastic rounding adds a uniform [0, 2^16) integer perturbation to the low
    16 bits BEFORE truncation. Probability of rounding up = (low_bits / 2^16),
    so E[round(x)] = x exactly. Eliminates the precision floor that biased RNE
    creates for in-place SGD-style updates.

    Implementation: int32 view + add randint + zero low 16 bits + cast to bf16.
    Carry from the low 16 bits propagates correctly through the high mantissa
    bits and (if needed) into the exponent — matching standard fp arithmetic.
    """
    if x_fp32.dtype != torch.float32:
        x_fp32 = x_fp32.float()
    x_int = x_fp32.view(torch.int32)
    rand = torch.randint(
        0, 1 << 16, x_int.shape, device=x_fp32.device, dtype=torch.int32,
    )
    # Zero the low 16 bits of the perturbed value so the subsequent .bfloat16()
    # cast is an exact truncation (no implicit RNE on top of our stochastic add).
    mask = torch.tensor(~0xFFFF, dtype=torch.int32, device=x_fp32.device)
    perturbed_masked = (x_int + rand).bitwise_and_(mask)
    return perturbed_masked.view(torch.float32).to(torch.bfloat16)


class TinyMultiHeadLutOptimizer:
    """AdamW with bf16 (or fp16) state for TinyMultiHeadLut weights.

    Args:
        modules: iterable of TinyMultiHeadLut modules.
        lr: learning rate.
        beta1, beta2: Adam exponential-decay rates.
        eps: numerical-stability term.
        weight_decay: AdamW-style decoupled weight decay (applied directly
            to weights, not to gradient).
        lr_schedule_fn: optional callable(step) -> scale factor for lr.
        state_dtype: dtype for stored m, v (default torch.bfloat16).
        compute_dtype: dtype for Adam math (default torch.float32 for
            numerical stability; set to torch.bfloat16 for pure-bf16).
    """

    def __init__(
        self,
        modules: Iterable[torch.nn.Module],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_schedule_fn: Optional[Callable[[int], float]] = None,
        state_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.float32,
        stochastic_rounding: bool = True,
    ):
        self.modules: List[torch.nn.Module] = list(modules)
        if not self.modules:
            raise ValueError("TinyMultiHeadLutOptimizer requires >= 1 module")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.lr_schedule_fn = lr_schedule_fn
        self.state_dtype = state_dtype
        self.compute_dtype = compute_dtype
        # Stochastic rounding for fp32 → bf16 weight updates. Default True for
        # bf16 weights (eliminates the round-to-nearest precision floor that
        # caused the v1 bf16-pipeline gap to compound over training).
        self.stochastic_rounding = bool(stochastic_rounding)

        self._step_count = 0
        self._states: List[dict] = []
        for m in self.modules:
            if not hasattr(m, 'weights'):
                raise TypeError(
                    f"TinyMultiHeadLutOptimizer expects modules with `weights` "
                    f"nn.Parameter; got {type(m).__name__}"
                )
            w = m.weights
            self._states.append({
                'm': torch.zeros_like(w, dtype=state_dtype),
                'v': torch.zeros_like(w, dtype=state_dtype),
            })

    def zero_grad(self, set_to_none: bool = True) -> None:
        for m in self.modules:
            if m.weights.grad is None:
                continue
            if set_to_none:
                m.weights.grad = None
            else:
                m.weights.grad.detach_()
                m.weights.grad.zero_()

    def step(self) -> None:
        self._step_count += 1
        lr = self.lr
        if self.lr_schedule_fn is not None:
            lr = lr * self.lr_schedule_fn(self._step_count)
        bias1 = 1.0 - self.beta1 ** self._step_count
        bias2 = 1.0 - self.beta2 ** self._step_count

        for module, state in zip(self.modules, self._states):
            w = module.weights
            grad = w.grad
            if grad is None:
                continue

            # Cast for compute (default: fp32 for numerical stability).
            cd = self.compute_dtype
            m_c = state['m'].to(cd)
            v_c = state['v'].to(cd)
            g_c = grad.to(cd)

            # Adam updates.
            m_c.mul_(self.beta1).add_(g_c, alpha=1.0 - self.beta1)
            v_c.mul_(self.beta2).addcmul_(g_c, g_c, value=1.0 - self.beta2)

            # Parameter update.
            m_hat = m_c / bias1
            v_hat = v_c / bias2
            denom = v_hat.sqrt_().add_(self.eps)
            update = m_hat / denom

            # Apply update + (optional) decoupled weight decay. For bf16 weights
            # with stochastic_rounding=True, the entire (w_old - lr*update -
            # lr*wd*w_old) arithmetic happens in fp32 and the final cast to
            # bf16 uses unbiased random rounding — preserves persistent small
            # updates that round-to-nearest would silently drop.
            if self.stochastic_rounding and w.dtype in (torch.bfloat16, torch.float16):
                new_w = w.data.float() - lr * update
                if self.weight_decay > 0.0:
                    new_w.mul_(1.0 - lr * self.weight_decay)
                if w.dtype == torch.bfloat16:
                    w.data.copy_(_stochastic_round_fp32_to_bf16(new_w))
                else:
                    w.data.copy_(_stochastic_round_fp32_to_fp16(new_w))
            else:
                w.data.add_(update.to(w.dtype), alpha=-lr)
                if self.weight_decay > 0.0:
                    w.data.mul_(1.0 - lr * self.weight_decay)

            # Store back the updated state in state_dtype.
            state['m'].copy_(m_c.to(self.state_dtype))
            state['v'].copy_(v_c.to(self.state_dtype))

    def state_dict(self) -> dict:
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'step_count': self._step_count,
            'states': [
                {'m': st['m'].clone(), 'v': st['v'].clone()} for st in self._states
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        self.lr = sd.get('lr', self.lr)
        self.beta1 = sd.get('beta1', self.beta1)
        self.beta2 = sd.get('beta2', self.beta2)
        self.eps = sd.get('eps', self.eps)
        self.weight_decay = sd.get('weight_decay', self.weight_decay)
        self._step_count = sd.get('step_count', 0)
        for st_self, st_loaded in zip(self._states, sd.get('states', [])):
            st_self['m'].copy_(st_loaded['m'])
            st_self['v'].copy_(st_loaded['v'])
