"""Function-emitting LUT cells: a cell stores BASIS PARAMETERS, not an output vector.

A raw FastMultiHeadLut cell stores one learnable value per output dimension -- 384 numbers
to describe one 384-long curve. Here a cell instead stores the parameters of a few simple
functions over the output-index axis i = 0..D-1, and its output vector is those functions
evaluated at every i:

    Gaussian:  W[c, i] = sum_k  amp[c,k] * exp(-0.5 * ((i - mu[c,k]) / sigma[c,k])^2)
    Fourier:   W[c, i] = a0[c] + sum_n  ac[c,n] cos(2*pi*n*i/D) + as[c,n] sin(2*pi*n*i/D)

INJECTION POINT. FastMultiHeadLut.forward reads `self.weights` at call time and hands it to
the autograd Function, which gathers rows with F.embedding_bag(mode='sum'). So the whole
change is to make `weights` a synthesized tensor instead of a stored Parameter: subclass,
drop the Parameter, expose `weights` as a property. Nothing in fast_multi_head_lut.py
changes, the gather/sum path is untouched, and gradients flow back through the synthesis
into the basis parameters by ordinary autograd.

WHERE TO SYNTHESISE, and why the whole table rather than the gathered rows. Per micro-batch
each slot gathers B_tok * n_heads * tph rows -- 6,144 * 4 * 128 = 3.1M -- from a table of
only n_tables * table_dim = 131,072 distinct cells. Every cell is hit ~24 times, so
synthesising the table ONCE per forward is ~24x cheaper than synthesising per gather.

A STRUCTURAL WARNING ABOUT THE FOURIER VARIANT, stated up front because it decides what the
experiment can show. The forward SUMS the gathered cells, and with FIXED frequencies the
basis matrix Phi is shared by every cell, so

    sum_c W[c] = sum_c (A[c] @ Phi) = (sum_c A[c]) @ Phi

i.e. the synthesis commutes with the gather-sum. That makes it cheap -- gather in the 2N-dim
amplitude space, synthesise once per (token, head) -- but it also means the model is
EXACTLY a CompressionMHL decompress whose Linear is frozen to a DCT/DFT basis. A learned
Linear can represent any fixed basis, so on expressiveness this variant is a strict subset
of exp_n_0121 / exp_n_0138. It is a compactness point, not a quality point.

The Gaussian variant does NOT commute: each cell has its own mu and sigma, so its curve
shape is per-cell and cannot be pulled out of the sum. That is where the idea is actually
new, and it is also the expensive one. `fourier_learn_freq` makes the Fourier variant
per-cell too, at the same loss of factorisation, for a like-for-like comparison.
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut


class _CellBasis(nn.Module):
    """Base: owns per-cell parameters, synthesises [n_cells, n_outputs]."""

    def __init__(self, n_cells: int, n_outputs: int, k: int):
        super().__init__()
        self.n_cells, self.n_outputs, self.k = n_cells, n_outputs, k
        self.register_buffer(
            "idx", torch.arange(n_outputs, dtype=torch.float32), persistent=False
        )

    def extra_repr(self):
        return (f"n_cells={self.n_cells}, n_outputs={self.n_outputs}, k={self.k}, "
                f"params/cell={self.params_per_cell}")


class GaussianCells(_CellBasis):
    """K bumps per cell: (mean, log_sigma, amplitude) each -> 3K params per cell.

    Means are initialised spread evenly across the index axis with a small jitter, sigma at
    the spacing between bumps, so a cell starts as a smooth partition of the axis rather
    than a pile of coincident bumps.
    """

    params_per_cell = property(lambda self: 3 * self.k)

    def __init__(self, n_cells, n_outputs, k, init_std, chunk=4096, generator=None,
                 compile_synth=False):
        super().__init__(n_cells, n_outputs, k)
        g = generator
        centres = torch.linspace(0.0, n_outputs - 1.0, k).view(1, k).repeat(n_cells, 1)
        spacing = n_outputs / max(k, 1)
        centres = centres + (torch.rand(n_cells, k, generator=g) - 0.5) * spacing
        self.mean = nn.Parameter(centres)
        self.log_sigma = nn.Parameter(
            torch.full((n_cells, k), math.log(spacing / 2.0))
            + 0.05 * torch.randn(n_cells, k, generator=g)
        )
        # amplitude scaled so the SUM of k overlapping bumps lands near init_std
        self.amp = nn.Parameter(
            torch.randn(n_cells, k, generator=g) * (init_std / math.sqrt(k)) * 2.0
        )
        self.chunk = chunk
        # The synthesis is a long elementwise chain -- subtract, scale, square, exp,
        # multiply, sum -- over a [chunk, K, n_outputs] intermediate. Eager runs each as
        # its own kernel and materialises the intermediate every time; inductor fuses the
        # whole chain into one kernel that never writes it. Measured 3.5x faster and 4.7
        # GiB lighter at K=24, which is the difference between a 24-hour run and a 7-hour
        # one. Off by default so the eager path stays the reference.
        if compile_synth:
            self._synth = torch.compile(self._synth, dynamic=False)

    def _synth(self, mean, log_sigma, amp):
        out = torch.empty(self.n_cells, self.n_outputs,
                          dtype=amp.dtype, device=amp.device)
        i = self.idx.view(1, 1, -1)
        for s in range(0, self.n_cells, self.chunk):
            e = min(s + self.chunk, self.n_cells)
            z = (i - mean[s:e].unsqueeze(-1)) * torch.exp(-log_sigma[s:e]).unsqueeze(-1)
            out[s:e] = (amp[s:e].unsqueeze(-1) * torch.exp(-0.5 * z * z)).sum(dim=1)
        return out

    def forward(self):
        # Checkpointed: the [chunk, k, n_outputs] intermediates are never kept for
        # backward, which is the difference between ~200 MB and ~5 GB per slot.
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._synth, self.mean, self.log_sigma, self.amp,
                              use_reentrant=False)
        return self._synth(self.mean, self.log_sigma, self.amp)


class FourierCells(_CellBasis):
    """N harmonics per cell: cos and sin amplitudes -> 2N + 1 params per cell.

    With `learn_freq=False` the frequencies are the fixed harmonics n/D and the synthesis
    is a single matmul against a shared [2N+1, D] basis -- see the module docstring for why
    that makes this variant a frozen-Linear decompress rather than something new.
    """

    params_per_cell = property(
        lambda self: (2 * self.k + 1) + (self.k if self.learn_freq else 0))

    def __init__(self, n_cells, n_outputs, k, init_std, learn_freq=False,
                 generator=None):
        super().__init__(n_cells, n_outputs, k)
        self.learn_freq = learn_freq
        g = generator
        scale = init_std / math.sqrt(2 * k + 1)
        self.dc = nn.Parameter(torch.randn(n_cells, 1, generator=g) * scale)
        self.a_cos = nn.Parameter(torch.randn(n_cells, k, generator=g) * scale)
        self.a_sin = nn.Parameter(torch.randn(n_cells, k, generator=g) * scale)
        base = torch.arange(1, k + 1, dtype=torch.float32)
        if learn_freq:
            self.freq = nn.Parameter(base.view(1, k).repeat(n_cells, 1))
        else:
            self.register_buffer("freq", base, persistent=False)
            phase = (2.0 * math.pi / n_outputs) * torch.outer(base, self.idx)  # [k, D]
            self.register_buffer(
                "basis", torch.cat([torch.ones(1, n_outputs), torch.cos(phase),
                                    torch.sin(phase)], dim=0), persistent=False)

    def forward(self):
        if not self.learn_freq:
            # [n_cells, 2k+1] @ [2k+1, D] -- one GEMM, and it COMMUTES with the
            # gather-sum, which is exactly why this variant is not new.
            return torch.cat([self.dc, self.a_cos, self.a_sin], dim=1) @ self.basis
        phase = (2.0 * math.pi / self.n_outputs) * (
            self.freq.unsqueeze(-1) * self.idx.view(1, 1, -1))
        return (self.dc
                + (self.a_cos.unsqueeze(-1) * torch.cos(phase)).sum(1)
                + (self.a_sin.unsqueeze(-1) * torch.sin(phase)).sum(1))


class FunctionEmittingFastMHL(FastMultiHeadLut):
    """FastMultiHeadLut whose cells emit basis functions instead of stored vectors.

    Routing is untouched: same anchors, same nap, same table_dim, same gather. Only the
    contents of `weights` change, from a stored Parameter to a synthesised tensor.
    """

    def __init__(self, *args, fe_basis="gaussian", fe_k=24, fe_learn_freq=False,
                 fe_chunk=4096, fe_seed=0, fe_compile=False, **kwargs):
        super().__init__(*args, **kwargs)
        # The parent has already built and registered the full [n_tables, table_dim,
        # n_outputs] Parameter. Pop it: it is what we are replacing, and leaving it
        # registered would put 50M dead weights in the optimizer.
        # nn.Module.__setattr__ intercepts Parameter assignment and writes straight to
        # _parameters, so the parent's `self.weights = nn.Parameter(...)` never went
        # through the property below -- and reading it here must not either.
        w = self._parameters.pop("weights")
        n_tables, table_dim, n_out = w.shape
        init_std, dev = float(w.detach().std()), w.device
        del w
        self._n_tables = n_tables
        n_cells = n_tables * table_dim
        g = torch.Generator().manual_seed(int(fe_seed) + 7)
        if fe_basis == "gaussian":
            self.cells = GaussianCells(n_cells, n_out, fe_k, init_std,
                                       chunk=fe_chunk, generator=g,
                                       compile_synth=fe_compile)
        elif fe_basis == "fourier":
            self.cells = FourierCells(n_cells, n_out, fe_k, init_std,
                                      learn_freq=fe_learn_freq, generator=g)
        else:
            raise ValueError(f"fe_basis must be 'gaussian' or 'fourier', got {fe_basis}")
        self.cells.to(dev)
        self.fe_basis, self.fe_k = fe_basis, fe_k

    @property
    def weights(self):
        """Synthesised on demand. FastMultiHeadLut.forward reads this at call time."""
        return self.cells().view(self._n_tables, self.table_dim, self.n_outputs)

    def basis_parameters(self):
        """The parameters that replaced the table -- for optimizer grouping."""
        return list(self.cells.parameters())
