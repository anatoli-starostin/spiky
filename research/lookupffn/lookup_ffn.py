"""
lookup_ffn.py -- a self-contained, teaching-oriented pure-PyTorch reference
implementation of LookupFFN.

Paper: "LookupFFN: Making Transformers Compute-lite for CPU Inference"
       Zhanpeng Zeng, Michael Davies, Pranav Pulijala, Karthikeyan Sankaralingam,
       Vikas Singh. ICML 2024. arXiv:2403.07221.
Reference code (compiled kernels; we do NOT use it here): github.com/mlpen/LookupFFN

WHY THIS FILE EXISTS
--------------------
This is NOT the paper's optimized code (their BH4 / gather run as compiled C++/CUDA
kernels). This is a clean, readable, correct-by-construction reference so a reader can
understand the *mechanism*. Everything is pure PyTorch and CPU-runnable. Equation
numbers below refer to the arXiv v1 of the paper.

THE ONE-SENTENCE IDEA
---------------------
A standard transformer FFN computes
        y = sum_i  sigma(<x, W_i>) * V_i                                  (dense, GEMM)
i.e. every hidden unit i is scored by a dot product <x, W_i>, passed through a
nonlinearity, and used to weight a stored vector V_i. That is O(d^2) multiply-adds.

LookupFFN replaces this with a set of learnable HASH TABLES addressed by a learnable
HASH of x:
        y = sum_k  T_k[ f_k(x) ]                                          (Eq 6)
Each head k hashes x to an integer address f_k(x) and *looks up* one stored row of a
table T_k; the rows are summed over the K heads. No d x d matmul: the work becomes a
cheap structured projection (BH4) + a table gather. You trade FLOPs for memory.

The two things that make it trainable and cheap:
  * BH4  (Eq ~19): a structured O(d log d) projection replacing the dense hash matrix.
  * a differentiable hash: hard argmax at inference (Eq 6), softmax relaxation while
    training (Eq 10), with a top-N neighbourhood truncation for efficiency (Eq 13).

Read top to bottom; each nn.Module has a docstring mapping it to the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Fast Walsh-Hadamard Transform (the "H" in BH4)
# ---------------------------------------------------------------------------
def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalised Fast Walsh-Hadamard Transform along the last dim.

    The Hadamard matrix H_n is an orthogonal +/-1 matrix. Multiplying by it mixes all
    coordinates, but instead of an O(n^2) matmul we use the butterfly algorithm in
    O(n log n) add/subtract operations (no multiplications at all). This is the cheap,
    dense-mixing "H" factor that BH4 interleaves with learnable block-diagonal matrices.

    Requires the last dimension `n` to be a power of two. We divide by sqrt(n) so the
    transform is orthonormal (H H^T = I), which keeps activations well-scaled.
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n & (n - 1) == 0, f"fwht needs a power-of-two length, got {n}"
    x = x.clone()
    h = 1
    # Standard in-place butterfly: at each stage, combine coordinate pairs 2h apart.
    while h < n:
        x = x.view(*orig_shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]        # "even" half of each pair-block
        b = x[..., 1, :]        # "odd"  half
        x = torch.stack([a + b, a - b], dim=-2)   # butterfly: (a+b, a-b)
        x = x.view(*orig_shape)
        h *= 2
    return x / math.sqrt(n)


# ---------------------------------------------------------------------------
# 2. BH4 projection  (Eq ~19):  R = (prod_{i=1}^{4} B_i H)
# ---------------------------------------------------------------------------
class BH4(nn.Module):
    """Structured O(d log d) projection replacing a dense d x d hash matrix.

    The hash needs a projection R that mixes all input coordinates so that different
    inputs land in different buckets. A dense R is d x d = O(d^2) params/FLOPs -- exactly
    the cost we are trying to avoid. BH4 instead builds R as a product of FOUR factors,
    each factor = a *learnable block-diagonal* matrix B_i followed by a fixed Hadamard
    mixing H:
            R = B_4 H B_3 H B_2 H B_1 H                                    (paper Eq ~19)
    (The "4" in BH4 is the number of B-H factors.)

    * B_i is block-diagonal: we split the d-vector into d/b blocks of size b and apply a
      learned b x b matrix to each block. Cost O(d*b), params O(d*b) -- linear in d.
    * H (fast Hadamard, above) then spreads information *across* blocks in O(d log d)
      adds, so after a few B-H stages every output coordinate depends on every input.

    Net: an expressive, learnable projection at O(d log d + d*b) instead of O(d^2). Block
    size b trades expressiveness (larger b) against cost (smaller b).
    """

    def __init__(self, dim: int, block: int = 16, n_factors: int = 4):
        super().__init__()
        assert dim % block == 0, "dim must be divisible by block size"
        assert dim & (dim - 1) == 0, "BH4 uses the Hadamard transform, so dim must be a power of two"
        self.dim, self.block, self.n_blocks = dim, block, dim // block
        # n_factors learnable block-diagonal matrices, shape [factor, n_blocks, b, b].
        # Initialised near-orthogonal (identity + small noise) so the product starts as a
        # well-behaved (roughly norm-preserving) mixing rather than exploding/vanishing.
        eye = torch.eye(block)
        w = eye.expand(n_factors, self.n_blocks, block, block).clone()
        w = w + 0.02 * torch.randn_like(w)
        self.blocks = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        lead = x.shape[:-1]
        for i in range(self.blocks.shape[0]):
            # Block-diagonal matmul: reshape to [..., n_blocks, block] and apply the
            # per-block b x b matrix via a batched matmul (einsum). This is the "B_i".
            xb = x.view(*lead, self.n_blocks, self.block)
            xb = torch.einsum("...nb,nkb->...nk", xb, self.blocks[i])
            x = xb.reshape(*lead, self.dim)
            # Then the fixed Hadamard mix "H" across the whole vector.
            x = fwht(x)
        return x


# ---------------------------------------------------------------------------
# 3. The learnable hash / addressing
# ---------------------------------------------------------------------------
class StructuredHash(nn.Module):
    """Turns a projected code z into a table address, hard or soft.

    After projecting x -> z (via BH4) and taking the first `n_bits` coordinates as the
    hash code, we address a table of C = 2^n_bits rows. The address is the "nearest
    structured binary code": we hold a code matrix S in {-1,+1}^(C x n_bits) and score
    each code by an inner product, then pick / weight codes by that score.

        score_i = <z, S_i>

    HARD path (inference, Eq 6):  address = argmax_i score_i, gather that one row.
    SOFT path (training,  Eq 10): weights = softmax(score), blend ALL rows -- a small,
                                  fully differentiable "attention over the codebook", so
                                  gradients flow to the projection with no straight-through
                                  estimator.
    TOP-N (efficiency, Eq 13):    keep only the N highest-scoring codes, softmax over
                                  those, blend N rows. N -> 1 recovers the hard gather;
                                  larger N is smoother. This is the bridge from soft to hard.

    KEY SPECIAL CASE (worth internalising): if S is the *full* hypercube {-1,+1}^n_bits,
    then argmax_i <z, S_i> is achieved by S_i = sign(z). So the hard address is simply the
    bit-pattern of sign(z) -- i.e. classic hyperplane locality-sensitive hashing. This is
    exactly the sign-of-projection routing our own LUT layers use (see README mapping). A
    smaller, "structured" S is just a chosen sub-codebook to shrink the table.
    """

    def __init__(self, n_bits: int, temp: float = 1.0):
        super().__init__()
        self.n_bits = n_bits
        self.n_codes = 1 << n_bits              # C = 2^n_bits
        self.temp = temp
        # Build S = all 2^n_bits sign vectors, ordered so code index i has bit j = (i>>j)&1.
        # With this ordering, argmax_i <z,S_i> == sum_j (z_j>0) * 2^j (the sign bit-code).
        idx = torch.arange(self.n_codes)
        bits = ((idx[:, None] >> torch.arange(n_bits)[None, :]) & 1).float()   # {0,1}
        S = 2.0 * bits - 1.0                                                   # {-1,+1}
        self.register_buffer("S", S)           # [C, n_bits], fixed (not learned)

    def scores(self, z_code: torch.Tensor) -> torch.Tensor:
        # <z, S_i> for every code i.  z_code: [..., n_bits] -> [..., C]
        return z_code @ self.S.t()

    def hard_index(self, z_code: torch.Tensor) -> torch.Tensor:
        """Integer address per token: argmax over codes (== sign bit-code for full S)."""
        return self.scores(z_code).argmax(dim=-1)          # [...]

    def soft_weights(self, z_code: torch.Tensor, top_n: int | None = None) -> torch.Tensor:
        """Differentiable weights over the C table rows.  [..., C]

        top_n=None -> softmax over all codes (Eq 10). top_n=k -> softmax over the k best
        codes only, others zero (Eq 13); with top_n=1 this is a soft-argmax that closely
        tracks the hard gather while still passing gradient.
        """
        s = self.scores(z_code) / self.temp
        if top_n is not None and top_n < self.n_codes:
            topv, topi = s.topk(top_n, dim=-1)
            w = torch.softmax(topv, dim=-1)
            out = torch.zeros_like(s)
            out.scatter_(-1, topi, w)          # place the N weights back, rest 0
            return out
        return torch.softmax(s, dim=-1)


# ---------------------------------------------------------------------------
# 4. One hash table (one "head")
# ---------------------------------------------------------------------------
class LookupTableHead(nn.Module):
    """A single learnable hash table T_k with its own learnable hash f_k.

    forward computes T_k[f_k(x)] (one term of the sum in Eq 6):
        1. z   = BH4(x)                         structured projection  (Eq ~19)
        2. code = first n_bits coords of z      the hash code
        3. hard:  row = T_k[argmax code]        gather one row         (Eq 6)
           soft:  row = softmax(code) @ T_k     differentiable blend   (Eq 10 / 13)

    T_k has shape [C, d_out]: its rows are the learnable "value" vectors V_i of the FFN
    it replaces -- but instead of scoring all of them with a matmul, we address one.
    """

    def __init__(self, d_in: int, d_out: int, n_bits: int, block: int = 16,
                 temp: float = 1.0, top_n: int | None = 4):
        super().__init__()
        self.proj = BH4(d_in, block=block)                 # x -> z
        self.hash = StructuredHash(n_bits, temp=temp)
        self.n_bits = n_bits
        self.top_n = top_n
        # The table: C rows, each a d_out vector. Small init keeps the summed output tame.
        self.table = nn.Parameter(0.02 * torch.randn(self.hash.n_codes, d_out))

    def forward(self, x: torch.Tensor, hard: bool | None = None) -> torch.Tensor:
        # Default: hard when in eval mode, soft when training. (Caller may override.)
        if hard is None:
            hard = not self.training
        z = self.proj(x)                        # [..., d_in]
        z_code = z[..., : self.n_bits]          # [..., n_bits]  (use n_bits coords as the code)
        if hard:
            idx = self.hash.hard_index(z_code)  # [...]
            # Gather one row of the table per token.
            return self.table[idx]              # [..., d_out]
        else:
            w = self.hash.soft_weights(z_code, top_n=self.top_n)   # [..., C]
            return w @ self.table               # [..., d_out]  weighted blend of rows


# ---------------------------------------------------------------------------
# 5. The LookupFFN layer:  y = sum_k T_k[f_k(x)]   (Eq 6)
# ---------------------------------------------------------------------------
class LookupFFN(nn.Module):
    """Drop-in replacement for a transformer FFN, built from K parallel hash tables.

    Standard FFN:   y = W2 @ GELU(W1 @ x)   ==  sum_i GELU(<x,W1_i>) * W2_i     (O(d^2))
    LookupFFN:      y = sum_{k=1}^{K} T_k[ f_k(x) ]                             (Eq 6)

    Each head hashes x independently and contributes one looked-up row of width d_model;
    the K contributions are summed (like summing the K nearest "experts"). More heads =
    more expressive, at more table memory but still no d^2 matmul.
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_bits: int = 8,
                 block: int = 16, temp: float = 1.0, top_n: int | None = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            LookupTableHead(d_model, d_model, n_bits, block=block, temp=temp, top_n=top_n)
            for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor, hard: bool | None = None) -> torch.Tensor:
        out = self.heads[0](x, hard=hard)
        for h in self.heads[1:]:
            out = out + h(x, hard=hard)         # sum_k T_k[f_k(x)]
        return out


# ---------------------------------------------------------------------------
# 6. A dense FFN baseline (for the FLOP/param comparison)
# ---------------------------------------------------------------------------
class DenseFFN(nn.Module):
    """The vanilla transformer FFN we are replacing: d -> 4d -> d with GELU."""

    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 7. A minimal causal transformer block + tiny GPT, so you see it end to end.
#    (The paper's repo is RoBERTa/encoder; we make ours causal to match nanochat.)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention (unchanged from a vanilla transformer;
    only the FFN is being replaced, so attention stays dense here)."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # causal mask
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Pre-norm transformer block: attention, then a LookupFFN instead of a dense FFN."""

    def __init__(self, d_model, n_heads=4, ffn_heads=4, n_bits=8, block=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = LookupFFN(d_model, n_heads=ffn_heads, n_bits=n_bits, block=block)

    def forward(self, x, hard=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x), hard=hard)
        return x


class TinyGPT(nn.Module):
    """A tiny GPT-style causal LM whose FFNs are LookupFFNs. For reading/demonstration."""

    def __init__(self, vocab=256, d_model=128, n_layers=2, n_heads=4,
                 ffn_heads=4, n_bits=8, block=16, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, ffn_heads, n_bits, block) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, hard=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x, hard=hard)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# 8. Rough FLOP / parameter accounting (per token) for the write-up.
# ---------------------------------------------------------------------------
def dense_ffn_cost(d_model, mult=4):
    params = 2 * d_model * mult * d_model + (mult * d_model + d_model)   # weights + biases
    macs = 2 * d_model * mult * d_model                                  # two matmuls, per token
    return params, macs


def lookup_ffn_cost(d_model, n_heads, n_bits, block):
    C = 1 << n_bits
    n_factors = 4
    bh4_params = n_heads * n_factors * (d_model // block) * block * block  # = n_heads*4*d*block
    table_params = n_heads * C * d_model
    params = bh4_params + table_params
    # per-token multiply-adds (hard path): per head = BH4 blockmuls (4*d*block) +
    # code scoring (n_bits*C) ; the Hadamard is adds-only and the gather is a memory read.
    macs = n_heads * (n_factors * d_model * block + n_bits * C)
    return params, macs


# ---------------------------------------------------------------------------
# 9. Smoke test (CPU-runnable): forward in soft and hard modes, print shapes + costs.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, n_bits, ffn_heads, block = 128, 8, 4, 16

    print("=== LookupFFN reference: smoke test (CPU) ===")
    print(f"d_model={d_model}  n_heads(FFN)={ffn_heads}  n_bits={n_bits} "
          f"(codes/table C={1<<n_bits})  bh4_block={block}\n")

    # --- (a) the FFN layer alone, both modes ---
    ffn = LookupFFN(d_model, n_heads=ffn_heads, n_bits=n_bits, block=block)
    x = torch.randn(2, 16, d_model)                       # [batch, seq, d_model]

    ffn.train()
    y_soft = ffn(x)                                       # training path: soft (Eq 10/13)
    ffn.eval()
    with torch.no_grad():
        y_hard = ffn(x)                                  # inference path: hard gather (Eq 6)
    print(f"[LookupFFN] input {tuple(x.shape)} -> soft {tuple(y_soft.shape)} "
          f"(train), hard {tuple(y_hard.shape)} (eval)")

    # gradient sanity: the soft path must be differentiable end to end
    loss = y_soft.pow(2).mean()
    loss.backward()
    g = ffn.heads[0].proj.blocks.grad
    print(f"[LookupFFN] soft path is differentiable: BH4 grad norm = {g.norm().item():.4f}")

    # a couple of BH4 / hadamard sanity checks
    h = fwht(torch.eye(8))
    print(f"[fwht] orthonormal? max|H H^T - I| = "
          f"{(h @ h.t() - torch.eye(8)).abs().max().item():.2e}")

    # --- (b) end-to-end tiny causal LM ---
    gpt = TinyGPT(vocab=256, d_model=d_model, n_layers=2,
                  ffn_heads=ffn_heads, n_bits=n_bits, block=block, max_len=64)
    idx = torch.randint(0, 256, (2, 16))
    gpt.train(); logits_soft = gpt(idx)
    gpt.eval()
    with torch.no_grad():
        logits_hard = gpt(idx)
    print(f"[TinyGPT] tokens {tuple(idx.shape)} -> logits {tuple(logits_soft.shape)} "
          f"(train/soft) and {tuple(logits_hard.shape)} (eval/hard)")

    # --- (c) FLOP / param comparison vs a dense FFN of the same width ---
    dp, dm = dense_ffn_cost(d_model)
    lp, lm = lookup_ffn_cost(d_model, ffn_heads, n_bits, block)
    print("\n=== per-token cost vs a dense d->4d->d FFN (same d_model) ===")
    print(f"dense  FFN : params={dp:>10,d}   MACs/token={dm:>10,d}")
    print(f"lookup FFN : params={lp:>10,d}   MACs/token={lm:>10,d}")
    print(f"ratio      : params x{lp/dp:6.2f}   MACs x{lm/dm:6.3f}  "
          f"(lookup trades ~{dm/lm:.0f}x fewer MACs for ~{lp/dp:.0f}x more params)")
    print("\nOK: forward ran in both soft (train) and hard (eval) modes on CPU.")
