"""DominanceLUT -- reference PyTorch implementation (sibling of spiky's ManifestoHardLUT).

A normal sign-address LUT on the input side; on the output side every table row stores ``nop`` *deltas*
instead of ``d_out`` values. Table ``t`` owns ``nop`` output anchor pairs ``(a_tk, b_tk)`` in ``range(d_out)``;
delta ``k`` of the addressed row is the vote "output ``a_tk`` should exceed output ``b_tk`` by delta".
The tables of one head are aggregated by Borda counting on the antisymmetric dominance matrix
``D[a,b] += delta, D[b,a] -= delta``; the head output is ``y_j = sum_b D[j,b]`` (see README.md for the math).

Conventions mirrored from ``spiky.lutorch_ex`` (cartridges/manifesto_base.py, manifesto_hard.py):
``LUTSpec`` geometry (h_in, h_out, tph, nap, d_in, d_out), ``anchor_mode='pairs'``, input anchors from
``canonical_full_coverage_pairs(seed=seed)`` (identical to ManifestoHardLUT for the same seed), MSB-first
sign-bit addresses, tables ``[G, tph, 2**nap, nop]`` initialised per group from a CPU generator seeded
``seed_w + g + 1``, forward ``[B, h_in, d_in] -> [B, h_out, d_out]``, head routing (per-head / fan-out /
fan-in) identical to ``ManifestoLUT._route``, hard forward with the "two-alternative" surrogate input gradient.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

SPIKY_SRC = os.environ.get("SPIKY_SRC", os.path.expanduser("~/work/repos/anatoli-starostin/spiky/src"))
try:
    import spiky.lutorch_ex  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, SPIKY_SRC)
from spiky.lutorch_ex import LUTSpec
from spiky.lutorch_ex.anchors import canonical_full_coverage_pairs
from spiky.lutorch_ex.addressing import msb_first_powers

OUTPUT_ANCHORS = ("same", "random", "given")
VOTES = ("delta", "sign")
NORMALIZE = ("none", "count")


def sample_output_pairs(d_out: int, n_groups: int, tph: int, nop: int, *, seed: int) -> torch.Tensor:
    """Random output anchor pairs ``[G, tph, nop, 2]``, all canonical (a < b), like the input pairs and ``'same'``.
    Per table the ``nop`` unordered pairs are distinct (a random subset of the C(d_out,2) pool). Group ``g`` uses a CPU
    ``torch.Generator`` seeded ``seed + g`` (cross-device deterministic). Orientation is a pure sign gauge for
    real-valued zero-mean deltas (flipping a pair negates that delta's contribution, i.e. ``delta -> -delta``), so
    no random flip is drawn; user-supplied ``output_pairs`` (``'given'``) are never canonicalised."""
    P = d_out * (d_out - 1) // 2
    if nop > P:
        raise ValueError(f"nop={nop} exceeds the number of distinct output pairs C({d_out},2)={P}")
    ti, tj = torch.triu_indices(d_out, d_out, offset=1)
    out = torch.empty(n_groups, tph, nop, 2, dtype=torch.long)
    for g in range(n_groups):
        gen = torch.Generator().manual_seed(seed + g)
        sel = torch.rand(tph, P, generator=gen).argsort(dim=1)[:, :nop]          # [tph, nop] distinct pool ids
        out[g, :, :, 0] = ti[sel]
        out[g, :, :, 1] = tj[sel]
    return out


def borda_scatter(votes: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, d_out: int,
                  inv_count: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Borda scores without materialising D. ``votes [B,G,tph,nop]``, ``pa/pb [G,tph,nop]`` (output anchors).
    ``y[b,g,j] = sum_{(t,k): a_tk=j} v - sum_{(t,k): b_tk=j} v`` (times ``inv_count[g,j]`` if given)."""
    B, G = votes.shape[:2]
    v = votes.reshape(B, G, -1)
    ia = pa.reshape(1, G, -1).expand(B, G, -1)
    ib = pb.reshape(1, G, -1).expand(B, G, -1)
    y = votes.new_zeros(B, G, d_out)
    y = y.scatter_add(2, ia, v).scatter_add(2, ib, -v)
    return y if inv_count is None else y * inv_count


def dominance_matrix(votes: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, d_out: int) -> torch.Tensor:
    """Explicit antisymmetric dominance matrices ``D [B,G,d_out,d_out]`` (reference / debug only; O(d_out^2))."""
    B, G, T, K = votes.shape
    D = votes.new_zeros(B, G, d_out, d_out)
    for g in range(G):
        for t in range(T):
            for k in range(K):
                a, b = int(pa[g, t, k]), int(pb[g, t, k])
                D[:, g, a, b] += votes[:, g, t, k]
                D[:, g, b, a] -= votes[:, g, t, k]
    return D


def borda_reference(votes, pa, pb, d_out, normalize="none"):
    """Brute-force Borda via the explicit D: ``y_j = sum_b D[j,b]`` (/ votes touching j if normalize='count')."""
    D = dominance_matrix(votes, pa, pb, d_out)
    y = D.sum(-1)
    if normalize == "count":
        y = y * vote_inv_count(pa, pb, d_out)
    return y


def vote_inv_count(pa: torch.Tensor, pb: torch.Tensor, d_out: int) -> torch.Tensor:
    """``[G, d_out]``: 1 / max(1, number of (table, k) votes touching output j) (a pair occurrence counts once per end)."""
    G = pa.shape[0]
    cnt = torch.zeros(G, d_out)
    ones = torch.ones(pa[0].numel())
    for g in range(G):
        cnt[g].index_add_(0, pa[g].reshape(-1), ones).index_add_(0, pb[g].reshape(-1), ones)
    return 1.0 / cnt.clamp(min=1.0)


def anchors_equal(m1: "DominanceLUT", m2: "DominanceLUT", *, output: bool = True) -> bool:
    """True iff both layers have identical input anchors (and output anchors if ``output``)."""
    ok = torch.equal(m1.anchor_a, m2.anchor_a) and torch.equal(m1.anchor_b, m2.anchor_b)
    if output:
        ok = ok and torch.equal(m1.out_a, m2.out_a) and torch.equal(m1.out_b, m2.out_b)
    return ok


class DominanceLUT(nn.Module):
    """See module docstring and README.md.

    Args:
        spec: ``LUTSpec`` (anchor_mode must be 'pairs'); ``spec.nap`` = input anchor pairs / address bits,
            ``spec.d_out`` = output width per head.
        nop: output pairs (deltas) per table; default ``nap``. With ``output_anchors='same'`` must equal ``nap``.
        output_anchors: 'same' | 'random' | 'given'.
        output_pairs: for 'given': long tensor ``[G, tph, nop, 2]`` (G = spec.n_groups = "heads").
        vote: 'delta' (real deltas) | 'sign' (hard +-1 vote, straight-through gradient to the delta).
        normalize: 'none' | 'count' (divide each output by the number of votes touching it).
        seed: seed of the INPUT anchors (same as ManifestoHardLUT(seed=...)).
        output_seed: seed of the 'random' output anchors (default ``seed + 1_000_003``).
        weight_seed: seed of the table init (default ``seed``); lets a student share anchors but not tables.
        weight_init_std: std of the normal init; default ``sqrt(d_out / (2*tph*nop))`` (x ``sqrt(h_out/n_groups)``
            for fan-in) -> Borda output std ~1 (derivation in README).
        add_bias: learnable per-output bias ``[h_out, d_out]`` (default False; Borda outputs are zero-sum).
        input_anchors: optional ``(anchor_a, anchor_b)`` ``[G, tph, nap]`` long (e.g. a teacher's) overriding seeded ones.
        dense_D: debug: compute the forward through the explicit D matrix.
    """

    def __init__(self, spec: LUTSpec, *, nop: Optional[int] = None, output_anchors: str = "same",
                 output_pairs: Optional[torch.Tensor] = None, vote: str = "delta", normalize: str = "none",
                 seed: int = 0, output_seed: Optional[int] = None, weight_seed: Optional[int] = None,
                 weight_init_std: Optional[float] = None, add_bias: bool = False,
                 input_anchors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, dense_D: bool = False):
        super().__init__()
        if spec.anchor_mode != "pairs":
            raise ValueError("DominanceLUT needs anchor_mode='pairs'")
        if spec.d_in < 2:
            raise ValueError(f"DominanceLUT needs d_in >= 2, got {spec.d_in}")
        if spec.d_out < 2:
            raise ValueError(f"DominanceLUT needs d_out >= 2, got {spec.d_out}")
        if output_anchors not in OUTPUT_ANCHORS:
            raise ValueError(f"output_anchors must be one of {OUTPUT_ANCHORS}, got {output_anchors!r}")
        if vote not in VOTES:
            raise ValueError(f"vote must be one of {VOTES}, got {vote!r}")
        if normalize not in NORMALIZE:
            raise ValueError(f"normalize must be one of {NORMALIZE}, got {normalize!r}")
        self.spec, self.vote, self.normalize, self.dense_D = spec, vote, normalize, dense_D
        self.output_anchors = output_anchors
        G, tph, nap, d_in, d_out = spec.n_groups, spec.tph, spec.nap, spec.d_in, spec.d_out
        nop = nap if nop is None else nop
        if nop < 1:
            raise ValueError("nop must be >= 1")
        self.nop = nop

        # ---- input anchors: exactly the standard layer's (canonical full coverage, a < b) ----
        if input_anchors is None:
            a, b = canonical_full_coverage_pairs(d_in, G, tph, nap, seed=seed)
        else:
            a, b = (t.detach().clone().long().cpu() for t in input_anchors)
            if a.shape != (G, tph, nap) or b.shape != (G, tph, nap):
                raise ValueError(f"input_anchors must have shape {(G, tph, nap)}")
        self.register_buffer("anchor_a", a)
        self.register_buffer("anchor_b", b)
        self.register_buffer("powers", msb_first_powers(nap))      # MSB-first: pair 0 = high bit
        self.register_buffer("in_head", torch.arange(G, dtype=torch.long) % spec.h_in)

        # ---- output anchors ----
        if output_anchors == "same":
            if d_in != d_out or nop != nap:
                raise ValueError(f"output_anchors='same' requires d_in == d_out and nop == nap "
                                 f"(got d_in={d_in}, d_out={d_out}, nop={nop}, nap={nap})")
            oa, ob = a.clone(), b.clone()
        elif output_anchors == "random":
            pairs = sample_output_pairs(d_out, G, tph, nop,
                                        seed=(seed + 1_000_003) if output_seed is None else output_seed)
            oa, ob = pairs[..., 0].contiguous(), pairs[..., 1].contiguous()
        else:
            if output_pairs is None:
                raise ValueError("output_anchors='given' needs output_pairs [G, tph, nop, 2]")
            p = output_pairs.detach().clone().long().cpu()
            if p.shape != (G, tph, nop, 2):
                raise ValueError(f"output_pairs must have shape {(G, tph, nop, 2)}, got {tuple(p.shape)}")
            oa, ob = p[..., 0].contiguous(), p[..., 1].contiguous()
        if int(oa.min()) < 0 or int(ob.min()) < 0 or int(oa.max()) >= d_out or int(ob.max()) >= d_out:
            raise ValueError(f"output anchors must lie in [0, d_out={d_out})")
        if bool((oa == ob).any()):
            raise ValueError("output pairs need a != b")
        self.register_buffer("out_a", oa)
        self.register_buffer("out_b", ob)
        self.register_buffer("inv_count", vote_inv_count(oa, ob, d_out))   # [G, d_out]

        # ---- delta tables [G, tph, K, nop] ----
        if weight_init_std is None:
            weight_init_std = math.sqrt(d_out / (2.0 * tph * nop)) * math.sqrt(spec.h_out / G)
        self.weight_init_std = weight_init_std
        ws = seed if weight_seed is None else weight_seed
        blocks = [torch.randn(tph, spec.n_cells, nop, generator=torch.Generator().manual_seed(ws + g + 1))
                  * weight_init_std for g in range(G)]
        self.weights = nn.Parameter(torch.stack(blocks, 0))
        self.bias = nn.Parameter(torch.zeros(spec.h_out, d_out)) if add_bias else None

    # ------------------------------------------------------------------ helpers
    @classmethod
    def like(cls, other: "DominanceLUT", **overrides) -> "DominanceLUT":
        """New layer sharing ``other``'s geometry and its input AND output anchors (given), e.g. a student of a
        teacher. Override any constructor kwarg (``weight_seed=``, ``vote=``, ...)."""
        kw = dict(nop=other.nop, output_anchors="given", output_pairs=torch.stack([other.out_a, other.out_b], -1),
                  vote=other.vote, normalize=other.normalize, input_anchors=(other.anchor_a, other.anchor_b),
                  add_bias=other.bias is not None, dense_D=other.dense_D,
                  weight_init_std=other.weight_init_std)
        kw.update(overrides)
        return cls(other.spec, **kw)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3 or x.shape[1] != self.spec.h_in or x.shape[2] != self.spec.d_in:
            raise ValueError(f"expected input [B, {self.spec.h_in}, {self.spec.d_in}], got {tuple(x.shape)}")

    def addresses(self, x: torch.Tensor):
        """``(u, c, c_alt, u_abs_star)``: margins ``[B,G,tph,nap]``, cell ids ``[B,G,tph]`` (bit_j = u_j > 0, MSB-first),
        the least-confident-bit-flip neighbour cell, and ``|u_{j*}|`` -- same maths as ManifestoLUT._addresses."""
        self._check_input(x)
        B = x.shape[0]
        G, tph, nap = self.spec.n_groups, self.spec.tph, self.spec.nap
        z = x[:, self.in_head, :]
        ia = self.anchor_a.reshape(1, G, tph * nap).expand(B, G, -1)
        ib = self.anchor_b.reshape(1, G, tph * nap).expand(B, G, -1)
        u = (z.gather(2, ia) - z.gather(2, ib)).reshape(B, G, tph, nap)
        c = ((u > 0).long() * self.powers).sum(-1)
        u_abs, j = u.abs().min(-1)
        return u, c, c ^ self.powers[j], u_abs

    def _read(self, W: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Row gather ``W[g,t,c[b,g,t]] -> [B,G,tph,nop]`` (flat index gather; backward scatters into selected rows)."""
        G, tph, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
        off = (torch.arange(G * tph, device=c.device).reshape(1, G, tph)) * K
        return W.reshape(G * tph * K, self.nop)[c + off]

    def _vote(self, delta: torch.Tensor) -> torch.Tensor:
        if self.vote == "delta":
            return delta
        return delta + (torch.sign(delta) - delta).detach()          # +-1 value, identity (straight-through) gradient

    def _borda(self, votes: torch.Tensor) -> torch.Tensor:
        inv = self.inv_count if self.normalize == "count" else None
        if self.dense_D:
            y = dominance_matrix(votes, self.out_a, self.out_b, self.spec.d_out).sum(-1)
            return y if inv is None else y * inv
        return borda_scatter(votes, self.out_a, self.out_b, self.spec.d_out, inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, h_in, d_in] -> [B, h_out, d_out]``.

        Value: hard read ``y_g = Borda(votes of the addressed rows of the tph tables of group g)``.
        Gradient: tables only through the addressed rows (hard). Input: as ManifestoHardLUT, the forward value is the
        hard read but the input gradient is that of the two-cell blend ``(1-U) row[c] + U row[c']`` with
        ``U = 0.5/(1+|u_{j*}|)`` ('rational'), index set stop-gradient. Because Borda is linear in the votes the blend
        is applied to the votes before aggregation. (Skipped when the input does not require grad.)"""
        u, c, c_alt, u_abs = self.addresses(x)
        votes = self._vote(self._read(self.weights, c))
        y = self._borda(votes)
        if torch.is_grad_enabled() and x.requires_grad:
            U = (0.5 / (1.0 + u_abs)).unsqueeze(-1)
            Wd = self.weights.detach()
            v_c, v_alt = self._vote(self._read(Wd, c)), self._vote(self._read(Wd, c_alt))
            surr = self._borda(U * (v_alt - v_c))
            y = y + (surr - surr.detach())
        if self.spec.h_out != self.spec.n_groups:                    # fan-in: sum groups into the shared output
            y = y.sum(dim=1, keepdim=True)
        return y if self.bias is None else y + self.bias
