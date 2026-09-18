"""MatrixMenuMultiHeadLUT: a LightMultiHeadLUT whose cells select a LINEAR MAP instead of a vector.

LightMultiHeadLUT (``cell_mode="constant"``) stores a ``d_out`` vector per cell and emits it. Here each
cell stores LOGITS over a per-head MENU of ``M`` matrices ``W[h, m]`` of shape ``[d_in, d_out]``, and its
contribution is ``x_h @ W_selected`` -- the addressed cell picks a linear map applied to that head's own
input slice. Everything upstream of the payload is LightMultiHeadLUT's, unchanged: the anchor-pair (or
single-anchor) margins, the detached sign address, the confidence score (``margin`` etc., via the same
``confidence_score``) and head-level table dropout. Only the cell PAYLOAD changes.

    d      = x_h[anchor_a] - x_h[anchor_b]                     # [N, H, T, NAP], as Light
    c      = pack(sign(d.detach()))                             # [N, H, T], as Light
    s      = score(|d|)                                         # [N, H, T], as Light (+ head dropout)
    p      = softmax(logits[t, c] / tau_h)                      # [N, H, T, M]  (soft)  or one-hot argmax (hard)
    y[n,h] = sum_t s[n,h,t] * x[n,h] @ (sum_m p[n,h,t,m] W[h,m])

THE COLLAPSE. The whole path is linear in p, so the T tables of a head are reduced BEFORE any matrix is
touched:

    a[n,h,m] = sum_t s[n,h,t] * p[n,h,t,m]                      # [N, H, M]: one embedding_bag
    y[n,h]   = sum_m a[n,h,m] * (x[n,h] @ W[h,m])               # one menu application per (token, head)

``p`` depends on (table, cell) only, so ``P = softmax(logits / tau)`` is computed ONCE for the whole table
([n_tables, 2^NAP, M], 16.8M values at H8/tph128/nap8/M64) and ``a`` is exactly Light's fused bagged sum
(``F.embedding_bag`` with ``per_sample_weights = s``) over P instead of over the vector table. That is
T = 128x fewer matrix applications than the naive per-table form, and no [N, H, T, M] tensor is ever
materialised. The naive form is kept as the reference in test_matrix_menu_multi_head_lut.py.

The menu application has three equivalent implementations (``menu_impl``); see ``_apply_menu``.

HARD / SOFT. ``menu_forward="soft"`` uses the softmax in train and eval. ``menu_forward="hard"`` is the
inference-faithful mode: the forward value uses the one-hot argmax of each cell's logits, and in training
the softmax's gradient is injected with the codebase's straight-through convention
``hard + (soft - sg(soft))`` (the same zero-valued-term idiom LightMultiHeadLUT's forward_mode="hard" uses),
so the forward IS the index read and the logits / tau still learn -- which is what closes the soft/hard gap
(no temperature annealing).

INFERENCE ARTEFACT. ``export_indices()`` returns the menu plus ONE index per cell (argmax), i.e. what a
deployed layer stores: ceil(log2 M) bits per cell instead of d_out floats.
"""
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .light_multi_head_lut import LightMultiHeadLUT


MENU_IMPLS = ("mix", "expand", "outer")


# Default element std of the menu matrices. Chosen so the layer starts near zero like LightMultiHeadLUT's
# Uniform[-1e-3, 1e-3] tables: at H8/tph128/nap8/d48 on real compressed codes the pre-decompress rms is
# ~1e-3 (Light: ~1.9e-3), vs 1.45 with the norm-preserving 1/sqrt(d_in).
MENU_INIT_STD = 1e-4


class _MenuBank(nn.Module):
    """Holds the menu as `weight`. A child module (not a direct parameter of the LUT) so the trainers'
    tables_no_decay rule -- which exempts only a LUT module's own parameters -- leaves it weight-decayed."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)


class MatrixMenuMultiHeadLUT(LightMultiHeadLUT):
    """LightMultiHeadLUT with a matrix-menu cell payload (see the module docstring).

    Args (menu-specific; every other argument is LightMultiHeadLUT's and keeps its meaning):
        menu_size: M, the number of matrices in each head's menu (default 64).
        menu_tau_init: initial softmax temperature (default 1.0). With near-zero logits the initial softmax
            is ~uniform whatever tau is, so tau sets the gradient scale (d p / d logit ~ p / tau), not the
            starting distribution; 1.0 is the neutral choice.
        menu_tau_granularity: "head" (default: one learnable log-tau per head, [H]) or "global" (one scalar).
            Per head because the menus are per head, so each head's cells may need to commit at a different
            rate; the extra cost is H-1 scalars.
        menu_tau_learnable: learn log-tau (default True). False stores it as a buffer (same state_dict key).
        menu_forward: "soft" (default) or "hard" (one-hot argmax forward; straight-through softmax gradient
            in training; argmax at eval).
        menu_init: "normal" (default: N(0, std^2)) or "orthogonal" (each W[h, m] a (semi-)orthogonal matrix
            scaled to the same element std). Either way std = MENU_INIT_STD (1e-4) unless overridden: SMALL, so
            the layer output starts near zero; orthogonal keeps only the direction structure at that scale.
        menu_init_scale: overrides the element std (both inits).
        menu_logit_noise: cell logits ~ N(0, noise^2) (default 0.01): near-uniform, so no cell starts
            committed, with just enough noise to break the symmetry between cells.
        menu_impl: "mix" (default), "expand" or "outer" -- three exact implementations of the menu
            application, differing only in speed/memory (see ``_apply_menu``). "mix" was fastest and
            leanest at N=24576, H=8, d=48, M=64 on the RTX 5090 (fwd+bwd 53.0 ms / 5.8 GiB peak vs
            expand 53.4 ms / 6.9 GiB and outer 61.7 ms / 9.2 GiB; LightMultiHeadLUT 42.7 ms / 3.5 GiB).

    Supported layout: read_top_n == 1, forward_mode "scored", no quant_mode, output_heads == 1, anchor_mode
    "pair" or "single", multi_head_input True (per-head menus) or False (one head). Everything else is refused.
    """

    def __init__(
        self,
        input_dim: int,
        n_tables: int,
        output_dim: int,
        n_anchor_pairs: int,
        *,
        menu_size: int = 64,
        menu_tau_init: float = 1.0,
        menu_tau_granularity: str = "head",
        menu_tau_learnable: bool = True,
        menu_forward: str = "soft",
        menu_init: str = "normal",
        menu_init_scale: Optional[float] = None,
        menu_logit_noise: float = 0.01,
        menu_impl: str = "mix",
        cell_mode: str = "matrix_menu",
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        **light_kwargs,
    ):
        if cell_mode != "matrix_menu":
            raise ValueError(f"MatrixMenuMultiHeadLUT is cell_mode='matrix_menu' only, got {cell_mode!r}")
        bad = []
        if light_kwargs.get("read_top_n", 1) != 1:
            bad.append(f"read_top_n={light_kwargs['read_top_n']} (needs 1)")
        if light_kwargs.get("forward_mode", "scored") != "scored":
            bad.append(f"forward_mode={light_kwargs['forward_mode']!r} (needs 'scored')")
        if light_kwargs.get("quant_mode") is not None:
            bad.append(f"quant_mode={light_kwargs['quant_mode']!r} (needs None)")
        if light_kwargs.get("output_heads", 1) != 1:
            bad.append(f"output_heads={light_kwargs['output_heads']} (needs 1)")
        if light_kwargs.get("codebook_out_dim") is not None:
            bad.append("codebook_out_dim (codebook is a different cell_mode)")
        if bad:
            raise ValueError("MatrixMenuMultiHeadLUT is not implemented for: " + "; ".join(bad))
        if menu_size < 1:
            raise ValueError(f"menu_size must be >= 1, got {menu_size}")
        if not (menu_tau_init > 0):
            raise ValueError(f"menu_tau_init must be > 0, got {menu_tau_init!r}")
        if menu_tau_granularity not in ("head", "global"):
            raise ValueError(f"menu_tau_granularity must be 'head' or 'global', got {menu_tau_granularity!r}")
        if menu_forward not in ("soft", "hard"):
            raise ValueError(f"menu_forward must be 'soft' or 'hard', got {menu_forward!r}")
        if menu_init not in ("normal", "orthogonal"):
            raise ValueError(f"menu_init must be 'normal' or 'orthogonal', got {menu_init!r}")
        if menu_impl not in MENU_IMPLS:
            raise ValueError(f"menu_impl must be one of {MENU_IMPLS}, got {menu_impl!r}")

        # Build the plain Light layer: anchors, powers, table_offset, the native address kernel, the score
        # parameters and head dropout come out bit-identical to a LightMultiHeadLUT with the same arguments
        # (every draw uses its own seeded Generator, so nothing here perturbs them). Its vector `tables` is
        # then replaced by the menu payload below -- a one-off 50 MB allocation at H8, freed immediately.
        super().__init__(input_dim, n_tables, output_dim, n_anchor_pairs, cell_mode="constant",
                         random_seed=random_seed, device=device, **light_kwargs)
        del self.tables
        dev = self.anchor_c.device if self.anchor_mode == "single" else self.anchor_a.device

        H, M = self.n_heads, int(menu_size)
        self.menu_size = M
        self.menu_forward = menu_forward
        self.menu_impl = menu_impl
        self.menu_tau_granularity = menu_tau_granularity
        self.menu_tau_learnable = bool(menu_tau_learnable)

        # Cell logits replace `tables`: [n_tables, 2^NAP, M] in the SAME (table, cell) layout, so the flat
        # address `index + table_offset` indexes a row of logits exactly as it indexed a row of `tables`.
        g_l = torch.Generator(device=dev).manual_seed(random_seed + 101) if random_seed is not None else None
        logits = torch.randn(n_tables, self.table_size, M, device=dev, generator=g_l) * float(menu_logit_noise)
        self.menu_logits = nn.Parameter(logits)

        # Per-head menu [H, M, d_in, d_out], SMALL at init (element std MENU_INIT_STD unless overridden) so the
        # layer's output starts near zero, like Light's near-zero tables -- which matters most with
        # inner_out_dim=-1, where there is no zeroed decompress behind it.
        g_w = torch.Generator(device=dev).manual_seed(random_seed + 202) if random_seed is not None else None
        std = MENU_INIT_STD if menu_init_scale is None else float(menu_init_scale)
        if menu_init == "normal":
            W = torch.randn(H, M, input_dim, output_dim, device=dev, generator=g_w) * std
        else:
            # (semi-)orthogonal with its gain chosen so the ELEMENT std equals `std` too: a [d_in, d_out] matrix
            # with orthonormal columns (or rows) has element std 1/sqrt(max(d_in, d_out)).
            gain = std * math.sqrt(max(input_dim, output_dim))
            W = torch.empty(H, M, input_dim, output_dim, device=dev)
            for h in range(H):
                for m in range(M):
                    # orthogonal_ draws from the global RNG; seed it from our generator for reproducibility
                    s = int(torch.randint(0, 2**62, (1,), generator=g_w, device=dev).item()) if g_w is not None else None
                    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                        if s is not None:
                            torch.manual_seed(s)
                        nn.init.orthogonal_(W[h, m], gain=gain)
        # The menu lives in a child module so it is weight-decayed like an ordinary weight: every trainer's
        # tables_no_decay rule exempts `m.parameters(recurse=False)` of LightMultiHeadLUT instances, which
        # still covers menu_logits (and menu_log_tau is 1-D, no-decay by the ndim rule), but not this child.
        self.menu_bank = _MenuBank(W)

        # Temperature, stored as log-tau (positivity for free; multiplicative steps). A learnable Parameter
        # or a buffer under the same key, as LightMultiHeadLUT does for its read tau.
        n_tau = H if menu_tau_granularity == "head" else 1
        log_tau = torch.full((n_tau,), math.log(menu_tau_init), device=dev)
        if self.menu_tau_learnable:
            self.menu_log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("menu_log_tau", log_tau)

    @property
    def menu(self):
        """The per-head menu [H, M, d_in, d_out] (stored as menu_bank.weight)."""
        return self.menu_bank.weight

    # ------------------------------------------------------------------ temperature ---------------------------------
    @property
    def menu_tau(self):
        """Per-head temperature [H] (global granularity broadcasts one value to every head)."""
        tau = self.menu_log_tau.exp()
        return tau.expand(self.n_heads) if tau.numel() == 1 else tau

    # ------------------------------------------------------------------ cell distributions --------------------------
    def menu_probs(self, hard: Optional[bool] = None):
        """Per-cell menu weights [n_tables, 2^NAP, M] as used by the forward.

        soft: softmax(logits / tau_h). hard: one-hot argmax, carrying the softmax's gradient through the
        straight-through term when grad is enabled (value exactly one-hot)."""
        hard = (self.menu_forward == "hard") if hard is None else hard
        tau_t = self.menu_tau.repeat_interleave(self.tables_per_head)          # [n_tables]
        P = torch.softmax(self.menu_logits / tau_t.view(-1, 1, 1), dim=-1)
        if not hard:
            return P
        onehot = F.one_hot(self.menu_logits.argmax(dim=-1), self.menu_size).to(P.dtype)
        if not torch.is_grad_enabled():
            return onehot
        return onehot + (P - P.detach())

    # ------------------------------------------------------------------ menu application ----------------------------
    def _apply_menu(self, a, x):
        """y[n,h] = sum_m a[n,h,m] * (x[n,h] @ W[h,m]).   a [N,H,M], x [N,H,d_in] -> [N,H,d_out].

        Three exact implementations; all hold an [N, H, M*d] intermediate (the unavoidable size of the
        product), and differ in how the GEMMs are shaped:
          outer : u = a (x) x  ([H, N, M*d_in]), then ONE GEMM per head  u @ W.view(M*d_in, d_out)
                  (contraction length M*d_in = 3072 -> tensor-core friendly).
          expand: Z = x @ W.permute -> [H, N, M*d_out] (one GEMM per head), then a batched a . Z.
          mix   : per-token matrix Wmix = a @ W.view(M, d_in*d_out) ([N, H, d_in*d_out]), then N*H tiny
                  [1,d_in] @ [d_in,d_out] matmuls.
        """
        H, M = self.n_heads, self.menu_size
        W = self.menu.to(x.dtype)
        di, do = W.shape[-2], W.shape[-1]
        N = x.shape[0]
        a = a.to(x.dtype)
        if self.menu_impl == "outer":
            u = (a.permute(1, 0, 2).unsqueeze(-1) * x.permute(1, 0, 2).unsqueeze(-2))   # [H, N, M, di]
            y = torch.bmm(u.reshape(H, N, M * di), W.reshape(H, M * di, do))            # [H, N, do]
            return y.permute(1, 0, 2)
        if self.menu_impl == "expand":
            Z = torch.bmm(x.permute(1, 0, 2), W.permute(0, 2, 1, 3).reshape(H, di, M * do))  # [H, N, M*do]
            y = torch.matmul(a.permute(1, 0, 2).unsqueeze(-2), Z.view(H, N, M, do))          # [H, N, 1, do]
            return y.squeeze(-2).permute(1, 0, 2)
        Wmix = torch.bmm(a.permute(1, 0, 2), W.reshape(H, M, di * do))                  # [H, N, di*do]
        y = torch.matmul(x.permute(1, 0, 2).unsqueeze(-2), Wmix.view(H, N, di, do))     # [H, N, 1, do]
        return y.squeeze(-2).permute(1, 0, 2)

    # ------------------------------------------------------------------ forward -------------------------------------
    def _margins(self, x):
        """Light's margins d and flat input, for the per-head [N,H,d_in] or shared [N,d_in] layout."""
        B = x.shape[0]
        if self.multi_head_input:
            H, T, NAP = self.n_heads, self.tables_per_head, self.n_anchor_pairs
            if x.dim() != 3 or x.shape[1] != H or x.shape[2] != self.input_dim:
                raise ValueError(f"multi_head_input expects x of shape [B, {H}, {self.input_dim}], "
                                 f"got {tuple(x.shape)}")
            if self.anchor_mode == "single":
                idx_c = self.anchor_c.reshape(1, H, T * NAP).expand(B, H, T * NAP)
                d = torch.gather(x, 2, idx_c).view(B, H, T, NAP)
            else:
                idx_a = self.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
                idx_b = self.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
                d = (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(B, H, T, NAP)
            return d, x.reshape(B, H * self.input_dim), x
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        d = x[:, self.anchor_c] if self.anchor_mode == "single" else x[:, self.anchor_a] - x[:, self.anchor_b]
        return d.view(B, 1, self.n_tables, self.n_anchor_pairs), x, x.unsqueeze(1)

    def menu_weights(self, x):
        """a [N, H, M]: the confidence-weighted menu distribution summed over each head's tables."""
        B, H, T = x.shape[0], self.n_heads, self.tables_per_head
        d, x_flat, _ = self._margins(x)
        index = self._pack_index(x_flat, d.view(B, H * T, -1) if not self.multi_head_input else d)
        index = index.reshape(B, H, T)
        score = self._head_drop_score(self.confidence_score(d))                 # [B, H, T]
        P = self.menu_probs()                                                    # [n_tables, 2^NAP, M]
        flat = P.reshape(self.n_tables * self.table_size, self.menu_size)
        flat_idx = (index + self.table_offset.view(1, H, T)).reshape(-1)
        return self._bagged_sum(flat, flat_idx, score, B * H, T).view(B, H, self.menu_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, H, d_in] (multi_head_input) or [B, d_in] -> [B, H, d_out] or [B, d_out], as Light."""
        a = self.menu_weights(x)
        xh = x if self.multi_head_input else x.unsqueeze(1)
        y = self._apply_menu(a, xh)
        return y if self.multi_head_input else y.squeeze(1)

    # ------------------------------------------------------------------ regularisers / export ------------------------
    def cell_tv(self):
        """Hamming-1 TV on the cells' SOFT menu distributions (the analogue of Light's cell_tv on vectors):
        mean over Hamming-1-adjacent cell pairs and tables of ||p_c - p_c'||^2."""
        nap = self.n_anchor_pairs
        P = self.menu_probs(hard=False)
        t = P.view(self.n_tables, *([2] * nap), self.menu_size)
        tv = t.new_zeros(())
        for ax in range(1, nap + 1):
            dd = t.diff(dim=ax)
            tv = tv + (dd * dd).sum()
        return tv / (self.n_tables * nap * (1 << (nap - 1)))

    def som_penalty(self, sigma):
        raise NotImplementedError("som_penalty is not defined for the matrix-menu payload")

    def forward_int(self, x):
        raise NotImplementedError("forward_int is a quant_mode path; the matrix menu has no quantised read")

    @torch.no_grad()
    def export_indices(self):
        """The inference artefact: {"menu": [H, M, d_in, d_out], "index": [n_tables, 2^NAP] (argmax),
        "bits_per_cell": ceil(log2 M)}. A hard-mode forward is exactly the read of these."""
        idx = self.menu_logits.argmax(dim=-1)
        dtype = torch.uint8 if self.menu_size <= 256 else torch.int16
        return {"menu": self.menu.detach().clone(), "index": idx.to(dtype),
                "bits_per_cell": max(1, math.ceil(math.log2(self.menu_size)))}

    def extra_repr(self) -> str:
        return (super().extra_repr() + f", menu_size={self.menu_size}, menu_forward={self.menu_forward!r}, "
                f"menu_tau_granularity={self.menu_tau_granularity!r}, menu_impl={self.menu_impl!r}")
