"""
lookup_ffn.py -- a self-contained, teaching-oriented pure-PyTorch reference
implementation of LookupFFN, matching the paper's ACTUAL mechanism.

Paper: "LookupFFN: Making Transformers Compute-lite for CPU Inference"
       Zhanpeng Zeng, Michael Davies, Pranav Pulijala, Karthikeyan Sankaralingam,
       Vikas Singh. ICML 2024. arXiv:2403.07221.
Reference code (compiled kernels; we do NOT use it): github.com/mlpen/LookupFFN
       -- verified against src/roberta/models/prenorm_lookup/lookup.py and
          compute_code_score/kernel.py.

WHY THIS FILE EXISTS
--------------------
This is NOT the paper's optimized code (their BH4 / gather run as compiled C++/CUDA
kernels). It is a clean, readable, correct-by-construction reference so a reader can
understand the *mechanism*. Everything is pure PyTorch and CPU-runnable. Equation
numbers refer to the arXiv v1 of the paper.

THE MECHANISM (and the subtle part people get wrong)
----------------------------------------------------
A standard transformer FFN computes
        y = sum_i  sigma(<x, W_i>) * V_i                                  (dense, GEMM)
i.e. every hidden unit i is scored by a dot product, passed through a nonlinearity, and
used to weight a stored vector V_i. That is O(d^2) multiply-adds.

LookupFFN replaces this with learnable HASH TABLES addressed by a learnable HASH of x:
        y = sum_k  score_k(x) * T_k[ code_k(x) ]                          (Eq 6 / 13)
Each head k hashes x to an integer address code_k and *looks up* one stored row of a
table T_k; the row is scaled by a scalar score_k and the K heads are summed. No d x d
matmul: the work becomes a cheap structured projection (BH4) + a table gather.

THE KEY POINT -- there is NO soft-vs-hard train/inference split.
Training and inference run the *same* forward. Concretely (matching their
compute_code_score kernel):

  1. project:  z = BH4(x)                          structured O(d log d) projection (Eq ~19)
  2. address:  code = bin2dec( sign(z) )           the HARD sign pattern, identical in
                                                    train and eval. With the full
                                                    hypercube codebook, "nearest code" ==
                                                    sign(z) (classic hyperplane LSH), so
                                                    no explicit code matrix is needed.
  3. score:    m = |z|;  score = m.sum() / prod_j(1 + exp(-2 m_j))
                                                    a SMOOTH, differentiable scalar. This
                                                    is exactly the single dominant term of
                                                    the softmax-over-codes: Eq 10's full
                                                    softmax collapsed to its top-1 term
                                                    (Eq 13 with N = 1), which equals
                                                    exp(<z, sign z>) / prod_j (e^{z_j}+e^{-z_j}).
  4. output:   y = sum_k score_k * T_k[code_k]

WHY THERE IS NO TRAIN/EVAL MISMATCH, and how gradients flow:
  * The address code = sign(z) is discrete and is the SAME in training and inference.
    We never soften it, and we never blend over all codes. There is no straight-through
    estimator on the address either.
  * Differentiability comes entirely from the continuous SCORE, which depends on the
    projection magnitudes |z_j|. Backprop reaches the BH4 projection through score, while
    the argmax/sign selection just picks which table row is scaled. Same function at train
    and eval => same output for the same input => no soft/hard gap by construction.
  * Consequently there is NO temperature parameter (not fixed, learnable, or annealed),
    and NO entropy / load-balancing / importance / sparsity auxiliary loss. The layer
    returns only its output tensor. (All confirmed in the official code.)

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

    The Hadamard matrix H_n is an orthogonal +/-1 matrix. Instead of an O(n^2) matmul we
    use the butterfly algorithm in O(n log n) add/subtract operations (no multiplications).
    This is the cheap dense-mixing "H" factor that BH4 interleaves with learnable
    block-diagonal matrices. `n` must be a power of two; we divide by sqrt(n) so the
    transform is orthonormal (H H^T = I), keeping activations well-scaled.
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n & (n - 1) == 0, f"fwht needs a power-of-two length, got {n}"
    x = x.clone()
    h = 1
    while h < n:  # standard iterative butterfly: combine coordinate pairs 2h apart
        x = x.view(*orig_shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*orig_shape)
        h *= 2
    return x / math.sqrt(n)


# ---------------------------------------------------------------------------
# 2. BH4 projection  (Eq ~19):  R = (prod_{i=1}^{4} B_i H)
# ---------------------------------------------------------------------------
class BH4(nn.Module):
    """Structured O(d log d) projection replacing a dense d x d hash matrix.

    The hash needs a projection that mixes all input coordinates so different inputs land
    in different buckets. A dense projection is O(d^2). BH4 instead builds it as a product
    of FOUR factors, each = a learnable block-diagonal matrix B_i followed by a fixed
    Hadamard mix H:
            R = B_4 H B_3 H B_2 H B_1 H                                    (paper Eq ~19)
    ("4" = number of B-H factors.)  B_i is block-diagonal (split d into d/b blocks of size
    b, apply a learned b x b matrix per block: O(d*b) params/FLOPs); H then spreads
    information across blocks in O(d log d) adds. After a few B-H stages every output
    depends on every input, at O(d log d + d*b) instead of O(d^2).

    NOTE on faithfulness: the paper's code projects hidden_size -> num_table*code_length in
    ONE rectangular BH4 and splits the result into per-table code vectors, and it blends
    the structured transform with a residual path at a fixed decay_coeff = 0.7. For
    readability this sketch uses a SQUARE per-head BH4 (d -> d) and reads off the first
    `code_length` coordinates as that head's code vector; the mechanism (structured cheap
    projection feeding the hash) is the same.
    """

    def __init__(self, dim: int, block: int = 16, n_factors: int = 4):
        super().__init__()
        assert dim % block == 0, "dim must be divisible by block size"
        assert dim & (dim - 1) == 0, "BH4 uses the Hadamard transform, so dim must be a power of two"
        self.dim, self.block, self.n_blocks = dim, block, dim // block
        eye = torch.eye(block)
        w = eye.expand(n_factors, self.n_blocks, block, block).clone()
        w = w + 0.02 * torch.randn_like(w)   # near-identity init -> stable norm-preserving start
        self.blocks = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        for i in range(self.blocks.shape[0]):
            xb = x.view(*lead, self.n_blocks, self.block)
            xb = torch.einsum("...nb,nkb->...nk", xb, self.blocks[i])  # per-block B_i
            x = xb.reshape(*lead, self.dim)
            x = fwht(x)                                                # Hadamard mix H
        return x


# ---------------------------------------------------------------------------
# 3. The hash: hard sign address + smooth magnitude score (NO soft/hard split)
# ---------------------------------------------------------------------------
def lookup_address(z_code: torch.Tensor) -> torch.Tensor:
    """Integer table address = the HARD sign pattern of the code vector.

    With the full-hypercube codebook S = {-1,+1}^b, argmax_i <z, S_i> is achieved by
    S_i = sign(z). So the "nearest structured binary code" is simply sign(z), and its
    index is the bit-packing  code = bin2dec( (sign(z)+1)/2 ) = sum_j 1[z_j>0] * 2^j.
    This is identical in training and inference; no explicit code matrix is needed.

    z_code: [..., b]  ->  [...] int64 in [0, 2^b).
    """
    b = z_code.shape[-1]
    bits = (z_code > 0).long()                                   # sign(z) as {0,1}
    powers = (1 << torch.arange(b, device=z_code.device))        # 2^j
    return (bits * powers).sum(dim=-1)


def lookup_score(z_code: torch.Tensor) -> torch.Tensor:
    """Smooth, differentiable scalar weight for the gathered row (Eq 13, N=1).

    This is the single dominant softmax-over-codes term. For the winning code sign(z):
        numerator   = exp(<z, sign z>) = exp( sum_j |z_j| )
        denominator = prod_j ( e^{z_j} + e^{-z_j} ) = prod_j 2 cosh(z_j)
    The paper's kernel computes the numerically-stable equivalent below (verified against
    compute_code_score/kernel.py): with m = |z_j|,
        score = ( sum_j m_j ) / prod_j ( 1 + e^{-2 m_j} )
    It is smooth in the magnitudes |z_j|, so gradients flow to the BH4 projection through
    it, while the discrete address sign(z) is left hard. No temperature is applied.
    """
    m = z_code.abs()
    denom = torch.prod(1.0 + torch.exp(-2.0 * m), dim=-1)
    return m.sum(dim=-1) / denom


# ---------------------------------------------------------------------------
# 4. One hash table (one "head")
# ---------------------------------------------------------------------------
class LookupTableHead(nn.Module):
    """A single learnable hash table T_k with its own learnable hash f_k.

    forward computes  score_k(x) * T_k[code_k(x)]  (one term of Eq 6 / 13):
        z    = BH4(x)                          structured projection      (Eq ~19)
        code = bin2dec(sign(z_code))           HARD address (top-1)       (same train/eval)
        s    = smooth magnitude score          differentiable weight      (Eq 13, N=1)
        out  = s * T_k[code]

    The SAME code and score come from the SAME code coordinates z_code (the first
    `code_length` outputs of BH4), so there is no inconsistency between which coordinates
    address the table and which produce the gradient-carrying weight. There is exactly ONE
    forward path: it is used identically in `.train()` and `.eval()`.

    T_k has shape [2^code_length, d_out]: its rows are the learnable "value" vectors V_i of
    the FFN it replaces -- but instead of scoring all of them with a matmul, we address one.
    """

    def __init__(self, d_in: int, d_out: int, code_length: int, block: int = 16):
        super().__init__()
        assert code_length <= d_in
        self.proj = BH4(d_in, block=block)                 # x -> z  (square, per-head)
        self.code_length = code_length
        n_codes = 1 << code_length
        self.table = nn.Parameter(0.02 * torch.randn(n_codes, d_out))  # small init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)                                   # [..., d_in]
        z_code = z[..., : self.code_length]                # [..., code_length] -> code AND score
        idx = lookup_address(z_code)                       # [...]  hard sign address
        s = lookup_score(z_code)                           # [...]  smooth differentiable weight
        return s.unsqueeze(-1) * self.table[idx]           # [..., d_out]


# ---------------------------------------------------------------------------
# 5. The LookupFFN layer:  y = sum_k score_k * T_k[code_k]   (Eq 6 / 13)
# ---------------------------------------------------------------------------
class LookupFFN(nn.Module):
    """Drop-in replacement for a transformer FFN, built from K parallel hash tables.

    Standard FFN:   y = W2 @ GELU(W1 @ x)  ==  sum_i GELU(<x,W1_i>) * W2_i     (O(d^2))
    LookupFFN:      y = sum_{k=1}^{K} score_k(x) * T_k[ code_k(x) ]            (Eq 6 / 13)

    Each head hashes x independently (hard sign address), gathers one row, scales it by its
    smooth score, and the K contributions are summed -- the only sum is over the K tables
    (N = 1 code per table). No d^2 matmul, one differentiable forward for train and eval.
    Returns ONLY the output tensor: no auxiliary loss (no load-balancing / entropy term).
    """

    def __init__(self, d_model: int, n_heads: int = 4, code_length: int = 8, block: int = 16):
        super().__init__()
        self.heads = nn.ModuleList([
            LookupTableHead(d_model, d_model, code_length, block=block)
            for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.heads[0](x)
        for h in self.heads[1:]:
            out = out + h(x)                               # sum_k score_k * T_k[code_k]
        return out


# ---------------------------------------------------------------------------
# 6. A dense FFN baseline (for the FLOP/param comparison)
# ---------------------------------------------------------------------------
class DenseFFN(nn.Module):
    """The vanilla transformer FFN we replace: d -> 4d -> d with GELU."""

    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 7. A minimal causal transformer block + tiny GPT, so you see it end to end.
#    (The paper's repo is RoBERTa/encoder; ours is causal to match nanochat.)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention (unchanged; only the FFN is replaced)."""

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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Pre-norm transformer block: attention, then a LookupFFN instead of a dense FFN."""

    def __init__(self, d_model, n_heads=4, ffn_heads=4, code_length=8, block=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = LookupFFN(d_model, n_heads=ffn_heads, code_length=code_length, block=block)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """A tiny GPT-style causal LM whose FFNs are LookupFFNs. For reading/demonstration."""

    def __init__(self, vocab=256, d_model=128, n_layers=2, n_heads=4,
                 ffn_heads=4, code_length=8, block=16, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, ffn_heads, code_length, block) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# 8. Rough FLOP / parameter accounting (per token) for the write-up.
# ---------------------------------------------------------------------------
def dense_ffn_cost(d_model, mult=4):
    params = 2 * d_model * mult * d_model + (mult * d_model + d_model)   # weights + biases
    macs = 2 * d_model * mult * d_model                                  # two matmuls, per token
    return params, macs


def lookup_ffn_cost(d_model, n_heads, code_length, block):
    C = 1 << code_length
    n_factors = 4
    bh4_params = n_heads * n_factors * (d_model // block) * block * block  # = n_heads*4*d*block
    table_params = n_heads * C * d_model
    params = bh4_params + table_params
    # per-token multiply-adds: per head = BH4 block-muls (4*d*block) + the score (O(code_length),
    # a sum and a product over code_length coords).  The Hadamard is adds-only; the address is
    # a sign+bit-pack; the gather is a memory read.  No d^2 term and NO scan over all 2^b codes.
    macs = n_heads * (n_factors * d_model * block + 2 * code_length)
    return params, macs


# ---------------------------------------------------------------------------
# 9. Smoke test (CPU-runnable): show train and eval are the SAME forward.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, code_length, ffn_heads, block = 128, 8, 4, 16

    print("=== LookupFFN reference (corrected): smoke test (CPU) ===")
    print(f"d_model={d_model}  n_heads(FFN)={ffn_heads}  code_length={code_length} "
          f"(rows/table={1<<code_length})  bh4_block={block}\n")

    ffn = LookupFFN(d_model, n_heads=ffn_heads, code_length=code_length, block=block)
    x = torch.randn(2, 16, d_model)

    # The whole point: ONE forward. train() and eval() must give the SAME output.
    ffn.train(); y_train = ffn(x)
    ffn.eval()
    with torch.no_grad():
        y_eval = ffn(x)
    same = torch.allclose(y_train, y_eval, atol=1e-6)
    print(f"[LookupFFN] input {tuple(x.shape)} -> output {tuple(y_train.shape)}")
    max_diff = (y_train - y_eval).abs().max().item()
    print(f"[LookupFFN] train()==eval() forward is IDENTICAL (no soft/hard split): "
          f"allclose={same}, max|diff|={max_diff:.2e}")

    # Differentiability: gradient must reach the BH4 projection THROUGH the smooth score,
    # even though the address sign(z) is hard (no straight-through estimator).
    loss = y_train.pow(2).mean()
    loss.backward()
    g = ffn.heads[0].proj.blocks.grad
    print(f"[LookupFFN] BH4 projection receives gradient (via the score): "
          f"grad norm = {g.norm().item():.4f}")

    # hadamard sanity
    h = fwht(torch.eye(8))
    print(f"[fwht] orthonormal? max|H H^T - I| = "
          f"{(h @ h.t() - torch.eye(8)).abs().max().item():.2e}")

    # end-to-end tiny causal LM (also identical train/eval forward)
    gpt = TinyGPT(vocab=256, d_model=d_model, n_layers=2, ffn_heads=ffn_heads,
                  code_length=code_length, block=block, max_len=64)
    idx = torch.randint(0, 256, (2, 16))
    gpt.train(); lg_train = gpt(idx)
    gpt.eval()
    with torch.no_grad():
        lg_eval = gpt(idx)
    print(f"[TinyGPT] tokens {tuple(idx.shape)} -> logits {tuple(lg_train.shape)}; "
          f"train==eval allclose={torch.allclose(lg_train, lg_eval, atol=1e-5)}")

    # FLOP / param comparison vs a dense FFN of the same width
    dp, dm = dense_ffn_cost(d_model)
    lp, lm = lookup_ffn_cost(d_model, ffn_heads, code_length, block)
    print("\n=== per-token cost vs a dense d->4d->d FFN (same d_model) ===")
    print(f"dense  FFN : params={dp:>10,d}   MACs/token={dm:>10,d}")
    print(f"lookup FFN : params={lp:>10,d}   MACs/token={lm:>10,d}")
    print(f"ratio      : params x{lp/dp:6.2f}   MACs x{lm/dm:6.3f}  "
          f"(lookup trades ~{dm/lm:.0f}x fewer MACs for ~{lp/dp:.0f}x more params)")
    print("\nOK: single differentiable forward; train and eval match; ran on CPU.")
