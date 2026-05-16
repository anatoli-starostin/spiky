"""Sliding-window gradient smoother.

For each registered parameter, keeps a ring buffer of the last W gradients.
After every backward pass, call `step_and_smooth()`: it pushes the current
`p.grad` into the ring and replaces `p.grad` with the mean of the last
`min(step+1, W)` slots.

Use case here: smooths LUT gradients across W consecutive micro-batches so
Adam's per-row v_t sees lower variance. Other parameters (LN, tok_emb,
unembedder) are untouched and use their normal per-batch gradient.

Memory cost: W * sum(p.numel()) * dtype_bytes. For 29 M LUT params (fp32)
at W=8 that's ~928 MB on top of the standard optimizer state.

Difference vs true grad-accum:
- true accum: 1 optimizer step per W micro-batches, fresh independent grads.
- windowed:   1 optimizer step per micro-batch, with the last W grads averaged.
              Steps are 8x more frequent; consecutive steps share W-1 of W grads.

For sparse-row Adam (LUT weights touched once per minibatch), the higher
step rate + lower per-step variance is exactly what we want to test.
"""
from __future__ import annotations
from typing import Iterable, List
import torch


class WindowedGradSmoother:
    def __init__(self, params: Iterable[torch.nn.Parameter], window_size: int):
        self.params: List[torch.nn.Parameter] = [p for p in params if p.requires_grad]
        self.W = int(window_size)
        assert self.W >= 1, "window_size must be >= 1"
        self.buffers: List[torch.Tensor] = []
        for p in self.params:
            self.buffers.append(torch.zeros((self.W, *p.shape),
                                            device=p.device, dtype=p.dtype))
        self.idx = 0      # next slot to write
        self.count = 0    # number of valid slots in [0, W]

    @torch.no_grad()
    def step_and_smooth(self):
        """Call AFTER backward, BEFORE optimizer.step(). Replaces p.grad with
        the mean of the last min(step+1, W) gradients per parameter."""
        slot = self.idx
        # Write current grad into slot.
        for p, buf in zip(self.params, self.buffers):
            if p.grad is None:
                continue
            buf[slot].copy_(p.grad)
        # Increment count (capped at W).
        self.count = min(self.count + 1, self.W)
        # Overwrite p.grad with windowed mean.
        if self.count < self.W:
            for p, buf in zip(self.params, self.buffers):
                if p.grad is None:
                    continue
                p.grad.copy_(buf[:self.count].mean(dim=0))
        else:
            for p, buf in zip(self.params, self.buffers):
                if p.grad is None:
                    continue
                p.grad.copy_(buf.mean(dim=0))
        self.idx = (self.idx + 1) % self.W

    def total_buffer_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers)
