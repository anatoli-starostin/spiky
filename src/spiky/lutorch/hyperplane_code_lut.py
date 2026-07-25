"""HyperplaneCodeLUT — a fully-soft, code-scoring hyperplane unembedder.

Maps a hidden vector x (dim E) directly to V = 2^nap logits (V must equal the
vocab, e.g. 32768 = 2^15 -> nap=15). Unlike HyperplaneMultiHeadLUT there are NO
stored per-cell output vectors and NO softmax inside: it emits a per-CODE score,
gated by a learned scalar, summed over T tables ("voting"). Fully differentiable
(no straight-through, no argmax) — the same soft-sign front-end as the sibling
class, but the "table" is replaced by the fixed +/-1 code matrix, so the output
width is 2^nap for free.

Math (N tokens):
  1. a     = x @ W_hyp^T + b_hyp          -> reshape [N, T, nap]     (one GEMM)
  2. p     = sign(a) * |a| / (T_soft + |a|)   (soft-sign, [N, T, nap])
  3. s_t   = p_t @ B^T                     per-code score, [N, V]; B[k,i]=+-1
  4. logits[n,k] = sum_t  w_cell[t,k] * s_t[n,k]
So logits[n,k] = <sum_t w_cell[t,:]-gated soft-signs, code k>: a confidence-
weighted Hamming-similarity between the input's soft sign-vector and each code k,
voted over T tables. Bit order is MSB-first, matching HyperplaneMultiHeadLUT's
index-pack convention (B[k,i] = +1 iff bit (nap-1-i) of k is set).

Parameters / param count:
  hyperplane_weight  [T*nap, E]   +  hyperplane_bias [T*nap]   -> T*nap*(E+1)
  w_cell             [T, V]        (multiplicative per-code gate) -> T*V
  code_matrix B      [V, nap]      fixed +/-1 buffer (NOT trainable)
  TOTAL trainable = T*nap*(E+1) + T*V.

Efficiency: never materialize [N, T, V] (~25 GB at N=12k/V=32k). The forward loops
over the T tables and accumulates into a single [N, V] logits tensor. In the
BACKWARD, autograd would otherwise retain each table's [N, V] score (O(T*N*V) —
OOMs for large T), so each per-table vote is wrapped in gradient checkpointing:
the [N, V] score is recomputed in backward instead of stored, bounding peak
activation memory to ~O([N, V]) at the cost of one extra matmul per table.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs

_HYPERPLANE_INITS = ("anchor_pairs", "random")


def _code_matrix(nap: int, device, dtype=torch.float32) -> torch.Tensor:
    """[V, nap] +/-1 code matrix, V = 2^nap, MSB-first:
    B[k, i] = +1 if (k >> (nap-1-i)) & 1 else -1.
    Same bit convention as HyperplaneMultiHeadLUT's _soft_bit_matrix_msb (B == its
    transpose), so codes line up with the sibling class's index packing."""
    V = 1 << nap
    k = torch.arange(V, device=device).unsqueeze(1)                 # [V, 1]
    shifts = torch.arange(nap - 1, -1, -1, device=device).unsqueeze(0)  # [1, nap] MSB-first
    bits = (k >> shifts) & 1                                        # [V, nap] in {0,1}
    return (bits.to(dtype) * 2.0 - 1.0)                            # {-1,+1}


class HyperplaneCodeLUT(nn.Module):
    """Fully-soft code-scoring hyperplane unembedder: E-dim hidden -> V=2^nap logits.

    Args:
      input_dim: E, hidden width.
      nap:       hyperplanes per table; the output width is V = 2^nap.
      n_tables:  T, number of voting tables summed in the output.
      n_outputs: V, MUST equal 2^nap (checked).
      T_soft:    soft-sign sharpness (fixed scalar, default 0.5, as the sibling).
      hyperplane_init: "anchor_pairs" (default; w_i = e_p1 - e_p2, b_i = 0) or "random".
      w_cell_init: constant initial value of the per-code gate (default 0.02, small
                   so step-1 logits are ~uniform -> loss ~ ln(V)).
    """

    def __init__(self, input_dim: int, nap: int, n_tables: int, n_outputs: int,
                 *, T_soft: float = 0.5, hyperplane_init: str = "anchor_pairs",
                 w_cell_init: float = 0.02, initial_weights_noise: float = 0.001,
                 anchor_sampling_policy=None, random_seed=None,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__()
        if hyperplane_init not in _HYPERPLANE_INITS:
            raise ValueError(f"hyperplane_init must be one of {_HYPERPLANE_INITS}, got {hyperplane_init!r}")
        if n_outputs != (1 << nap):
            raise ValueError(f"n_outputs (={n_outputs}) must equal 2^nap (=2^{nap}={1 << nap})")
        if not (1 <= nap <= 20):
            raise ValueError(f"nap must be in [1, 20], got {nap}")

        self.input_dim = input_dim
        self.nap = nap
        self.n_tables = n_tables
        self.n_outputs = n_outputs
        self.T_soft = float(T_soft)
        self.hyperplane_init = hyperplane_init

        dev = device or torch.device("cpu")
        policy = anchor_sampling_policy or AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE

        # --- Hyperplane params (mirror HyperplaneMultiHeadLUT._hyperplane_project) ---
        w_init = torch.zeros(n_tables, nap, input_dim, dtype=torch.float32, device=dev)
        b_init = torch.zeros(n_tables, nap, dtype=torch.float32, device=dev)
        if hyperplane_init == "anchor_pairs":
            anchor_a, anchor_b = get_balanced_anchor_pairs(
                n_tables=n_tables, n_anchor_pairs=nap, input_dim=input_dim,
                device=dev, random_seed=random_seed, policy=policy, n_heads=1)
            t_idx = torch.arange(n_tables, device=dev).view(-1, 1).expand(n_tables, nap)
            n_idx = torch.arange(nap, device=dev).view(1, -1).expand(n_tables, nap)
            w_init[t_idx, n_idx, anchor_a] += 1.0
            w_init[t_idx, n_idx, anchor_b] -= 1.0
        else:  # "random"
            gen = None
            if random_seed is not None:
                gen = torch.Generator(device=dev).manual_seed(random_seed + 2)
            w_init.normal_(mean=0.0, std=initial_weights_noise, generator=gen)

        # Stored flat as [T*nap, E] / [T*nap] for the one fused GEMM.
        self.hyperplane_weight = nn.Parameter(w_init.reshape(n_tables * nap, input_dim).to(dtype))
        self.hyperplane_bias = nn.Parameter(b_init.reshape(n_tables * nap).to(dtype))

        # Per-code multiplicative gate, small so init logits are ~uniform.
        self.w_cell = nn.Parameter(torch.full((n_tables, n_outputs), float(w_cell_init),
                                              dtype=dtype, device=dev))

        # Fixed +/-1 code matrix (non-trainable buffer), [V, nap].
        self.register_buffer("code_matrix", _code_matrix(nap, dev, dtype=torch.float32))

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, nap={self.nap}, n_tables={self.n_tables}, "
                f"n_outputs={self.n_outputs}, T_soft={self.T_soft}, init={self.hyperplane_init}")

    @staticmethod
    def _table_vote(p_t, w_cell_t, BT):
        """One table's gated per-code vote: [N,nap]@[nap,V] -> [N,V], gated by w_cell_t."""
        return w_cell_t.unsqueeze(0) * (p_t @ BT)                  # [N, V]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, input_dim] -> logits [N, n_outputs]. fp32 for stable logits."""
        x = x.float()
        N = x.shape[0]
        a = (x @ self.hyperplane_weight.float().t() + self.hyperplane_bias.float())
        a = a.view(N, self.n_tables, self.nap)                     # [N, T, nap]
        absa = a.abs()
        p = a.sign() * absa / (self.T_soft + absa)                 # soft-sign [N, T, nap]

        BT = self.code_matrix.float().t()                          # [nap, V]  (== B^T)
        w_cell = self.w_cell.float()                               # [T, V]
        logits = x.new_zeros(N, self.n_outputs, dtype=torch.float32)
        # Accumulate over tables; never materialize [N, T, V]. Under grad, each
        # per-table [N,V] score is recomputed in backward (checkpoint) so peak
        # activation memory stays ~O([N,V]) regardless of T (else autograd retains
        # T score tensors and OOMs for large T).
        use_ckpt = torch.is_grad_enabled() and p.requires_grad
        for t in range(self.n_tables):
            if use_ckpt:
                logits = logits + _ckpt.checkpoint(
                    self._table_vote, p[:, t, :], w_cell[t], BT, use_reentrant=False)
            else:
                logits = logits + self._table_vote(p[:, t, :], w_cell[t], BT)
        return logits
