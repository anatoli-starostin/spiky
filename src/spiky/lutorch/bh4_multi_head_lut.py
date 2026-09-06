"""BH4-addressed multi-head LUT: the LookupFFN routing, with our tables on top.

WHAT THIS REPLACES. In CompressionMultiHeadLUT the route from x to a table address is
``compress`` (a dense Linear into a narrow per-head code) followed by anchor-pair sign
tests, ``index = pack(sign(z[a_j] - z[b_j]))``. This module replaces BOTH: the projection
becomes BH4, a structured O(d log d) transform, and the address becomes the sign of the
projected COORDINATES, ``index = pack(sign(z_j))`` -- which is what LookupFFN does
(arXiv:2403.07221). ``decompress`` is unchanged and still sits on top.

REUSED, NOT REIMPLEMENTED. ``fwht`` and the BH4 factorisation are taken from nucstar's
reference implementation, ``research/lookupffn/lookup_ffn.py`` on branch
research/lookupffn (head 93de410d), which was itself verified against the paper's repo.
The only change here is that the four heads' block matrices are held in ONE parameter and
applied with a single batched einsum, instead of four separate modules; that is a pure
kernel-launch optimisation and ``test_bh4_matches_reference`` asserts it agrees with the
reference module elementwise.

TWO DEVIATIONS FROM THE REFERENCE, both deliberate and both recorded here:

 1. The paper projects ``hidden_size -> num_table * code_length`` in ONE rectangular BH4
    and splits the result into per-table code vectors. Coordinate-sign addressing cannot
    reuse coordinates across tables the way anchor PAIRS can, so the coordinate budget is
    unavoidable: with H heads, tph tables per head and NAP bits per table we need
    ``H * tph * NAP`` sign coordinates. We give each head its own SQUARE BH4 (their
    sketch's shape) at a width that covers that head's ``tph * NAP``, rather than one
    rectangular transform over all of them. Per-head costs the same parameters, wastes
    far less padding (384 -> 1024 rather than 384 -> 4096), and matches the per-head
    structure the rest of our stack already has.

 2. BH4 needs a power-of-two width and n_embd is 384, so x is ZERO-PADDED to the working
    width.

 3. A FIXED HADAMARD IS APPLIED TO THE PADDED INPUT BEFORE the BH4 stack. This is not
    decoration; without it the layer is half dead, and the sign-constancy diagnostic
    caught it. The normalised Walsh-Hadamard transform is involutory (H H = I), so with
    the reference's near-identity init of the B_i the whole product collapses to
        R = B_4 H B_3 H B_2 H B_1 H  ~=  H^4  =  I,
    i.e. BH4 at initialisation hands back its input -- INCLUDING the padding. Measured on
    real tokens that left ~512 of each head's 896 code coordinates sitting on padded
    zeros, with the pooled code std at 0.65 ~= sqrt(384/896) confirming it exactly, and
    layer 5 reaching only 7.9 distinct addresses out of 128. Pre-multiplying by one fixed
    H puts the 384 informative coordinates into every one of the 1024 outputs before the
    learnable stack sees them, and because that H sits OUTSIDE the product the H^4 = I
    cancellation can no longer restore the padding structure. It adds no parameters and
    leaves the reference's BH4 itself untouched.

The score is our shared ``_confidence_score``, so every confidence_form the rest of the
codebase supports is available here; ``margin`` is the form that is algebraically the
paper's own score.
"""
import math

import torch
import torch.nn as nn

from .fast_multi_head_lut import _confidence_score


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalised Fast Walsh-Hadamard Transform along the last dim.

    Verbatim from research/lookupffn/lookup_ffn.py. The butterfly algorithm in
    O(n log n) add/subtract operations; the 1/sqrt(n) keeps it orthonormal.
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n & (n - 1) == 0, f"fwht needs a power-of-two length, got {n}"
    x = x.clone()
    h = 1
    while h < n:
        x = x.view(*orig_shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*orig_shape)
        h *= 2
    return x / math.sqrt(n)


def hadamard_matrix(n: int, device=None) -> torch.Tensor:
    """The same normalised Hadamard fwht applies, materialised as an n x n matrix.

    Built by running fwht on the identity, so it is the transform by construction rather
    than by a re-derivation. H is symmetric and involutory (H H = I), so `x @ H` is exactly
    `fwht(x)`; the equivalence is asserted in diag_bh4_verify.py.
    """
    return fwht(torch.eye(n)).to(device)


class BH4MultiHead(nn.Module):
    """H independent BH4 transforms, R_h = B_4 H B_3 H B_2 H B_1 H, applied in one pass.

    Parameter count is exactly the reference's, per head:
        n_factors * (dim / block) * block^2  =  n_factors * dim * block
    Init is near-identity (I + 0.02 N(0,1)) exactly as the reference, so the transform
    starts approximately norm-preserving.

    forward: [N, H, dim] -> [N, H, dim]
    """

    def __init__(self, dim: int, n_heads: int, block: int = 4, n_factors: int = 4,
                 random_seed=None, device=None):
        super().__init__()
        assert dim % block == 0, f"dim {dim} must be divisible by block {block}"
        assert dim & (dim - 1) == 0, f"BH4 needs a power-of-two dim, got {dim}"
        self.dim, self.block, self.n_heads = dim, block, n_heads
        self.n_blocks, self.n_factors = dim // block, n_factors
        g = None
        if random_seed is not None:
            g = torch.Generator(device='cpu').manual_seed(random_seed)
        eye = torch.eye(block)
        w = eye.expand(n_heads, n_factors, self.n_blocks, block, block).clone()
        w = w + 0.02 * torch.randn(w.shape, generator=g)
        self.blocks = nn.Parameter(w.to(device))
        # The Hadamard as an explicit matrix, applied by one GEMM instead of log2(dim)
        # butterfly stages. Identical arithmetic -- hadamard_matrix() is built BY fwht, and
        # test (b) asserts x @ H == fwht(x) -- but the butterfly's 10 stages each allocate a
        # [N, H, dim] temporary, which made the layer memory-bound: measured 2.2 s/step
        # against 0.2 s/step for the Light runs, i.e. a 16k-step run would have taken ~10 h.
        # A GEMM uses tensor cores and one allocation.
        self.register_buffer("hadamard", hadamard_matrix(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, H, D = x.shape
        for i in range(self.n_factors):
            xb = x.view(N, H, self.n_blocks, self.block)
            # per-head, per-block B_i  (the reference's einsum with a head axis added)
            xb = torch.einsum('nhqb,hqkb->nhqk', xb, self.blocks[:, i])
            x = xb.reshape(N, H, D) @ self.hadamard          # == fwht(...)
        return x


class BH4MultiHeadLUT(nn.Module):
    """LookupFFN routing over our table bank.

    forward, identical in train and eval::

        xp    = zero_pad(x, width)                     # [N, width]
        z     = BH4_h(xp)                              # [N, H, width]
        zc    = z[..., :tph*NAP].view(N, H, tph, NAP)  # this head's code coordinates
        index = pack(sign(zc.detach()))                # [N, H, tph]  MSB-first, NO grad
        score = confidence(|zc|)                       # [N, H, tph]  differentiable
        out   = sum_t score * tables[h, t, index]      # [N, H, output_dim]

    As in LightMultiHeadLUT the address is detached and there is no straight-through
    estimator, so gradient reaches x only through the score -- see doc/lutorch/
    lut_mechanisms.pdf. The difference from Light is purely which vector is signed:
    coordinates here, coordinate DIFFERENCES there.
    """

    def __init__(self, input_dim: int, n_heads: int, tables_per_head: int,
                 n_anchor_pairs: int, output_dim: int, block: int = 4,
                 n_factors: int = 4, confidence_form: str = "margin",
                 confidence_gain: float = 1.0, initial_weights_noise: float = 1e-3,
                 random_seed=None, device=None):
        super().__init__()
        if confidence_form not in ("bounded", "margin", "bounded_norm"):
            raise ValueError(
                "confidence_form must be 'bounded', 'margin' or 'bounded_norm', "
                f"got {confidence_form!r}")
        self.input_dim, self.n_heads = input_dim, n_heads
        self.tables_per_head, self.n_anchor_pairs = tables_per_head, n_anchor_pairs
        self.output_dim = output_dim
        self.confidence_form, self.confidence_gain = confidence_form, float(confidence_gain)
        self.table_size = 2 ** n_anchor_pairs
        self.n_code = tables_per_head * n_anchor_pairs

        # Working width: a power of two covering both the input and the code budget.
        need = max(input_dim, self.n_code)
        self.width = 1 << (need - 1).bit_length()
        self.bh4 = BH4MultiHead(self.width, n_heads, block=block, n_factors=n_factors,
                                random_seed=random_seed, device=device)

        # MSB-first bit-pack, matching FastMultiHeadLut / LightMultiHeadLUT.
        self.register_buffer(
            "powers",
            (2 ** torch.arange(n_anchor_pairs - 1, -1, -1, device=device)).to(torch.int64))
        self.register_buffer(
            "table_offset",
            torch.arange(n_heads * tables_per_head, device=device, dtype=torch.int64)
            * self.table_size)

        # Same table init convention as Light: Uniform[-noise, +noise], per-head draws.
        blocks_ = []
        for h in range(n_heads):
            g = (None if random_seed is None
                 else torch.Generator(device='cpu').manual_seed(random_seed + h + 1))
            blocks_.append(torch.rand(tables_per_head, self.table_size, output_dim,
                                      generator=g) - 0.5)
        u = torch.cat(blocks_, dim=0).to(device)
        self.tables = nn.Parameter(u * (2.0 * initial_weights_noise))

    def code(self, x: torch.Tensor) -> torch.Tensor:
        """The per-table code coordinates [N, H, tph, NAP]. Exposed for diagnostics."""
        N = x.shape[0]
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x must be [N, {self.input_dim}], got {tuple(x.shape)}")
        xp = x
        if self.width != self.input_dim:
            xp = torch.nn.functional.pad(x, (0, self.width - self.input_dim))
        # Fixed pre-Hadamard: spreads the informative coordinates over the whole working
        # width so the padded zeros cannot survive the near-identity BH4 (see deviation 3).
        xp = fwht(xp)
        z = self.bh4(xp.unsqueeze(1).expand(N, self.n_heads, self.width))
        return z[..., :self.n_code].view(N, self.n_heads,
                                         self.tables_per_head, self.n_anchor_pairs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        zc = self.code(x)                                        # [N,H,T,NAP]
        H, T = self.n_heads, self.tables_per_head
        index = ((zc.detach() > 0).to(torch.int64) * self.powers).sum(-1)   # [N,H,T]
        score = _confidence_score(zc, self.confidence_form, self.confidence_gain)
        flat = self.tables.reshape(H * T * self.table_size, self.output_dim)
        flat_idx = (index + self.table_offset.view(1, H, T)).reshape(N * H, T)
        out = torch.nn.functional.embedding_bag(
            flat_idx, flat, mode='sum',
            per_sample_weights=score.reshape(N * H, T).to(flat.dtype))
        return out.view(N, H, self.output_dim)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, n_heads={self.n_heads}, "
                f"tph={self.tables_per_head}, NAP={self.n_anchor_pairs}, "
                f"width={self.width}, table_size={self.table_size}, "
                f"confidence_form={self.confidence_form!r}")
