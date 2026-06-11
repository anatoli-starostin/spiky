"""Verify torch.amp.autocast(bf16) actually casts vanilla matmuls (including
the unembedder head) to bf16. Builds a minimal MinimalGPT-shaped network and
inspects intermediate dtypes inside vs outside an autocast block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, D=384, H=6):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.mlp_up = nn.Linear(D, 4 * D, bias=False)
        self.mlp_down = nn.Linear(4 * D, D, bias=False)
        self.H, self.D = H, D

    def forward(self, x):
        h = self.ln1(x)
        qkv = self.qkv(h)
        print(f"  block.qkv         out dtype = {qkv.dtype}")
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(*q.shape[:-1], self.H, self.D // self.H).transpose(-2, -3)
        k = k.view(*k.shape[:-1], self.H, self.D // self.H).transpose(-2, -3)
        v = v.view(*v.shape[:-1], self.H, self.D // self.H).transpose(-2, -3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  block.sdpa        out dtype = {attn.dtype}")
        # attn: (B, H, T, d) -> (B, T, H*d=D)
        B, _, T, d = attn.shape
        attn = attn.transpose(-2, -3).reshape(B, T, self.D)
        x = x + self.proj(attn)
        h = self.ln2(x)
        u = self.mlp_up(h)
        print(f"  block.mlp_up      out dtype = {u.dtype}")
        x = x + self.mlp_down(F.gelu(u))
        return x


class Tiny(nn.Module):
    def __init__(self, V=32768, D=384):
        super().__init__()
        self.tok_emb = nn.Embedding(V, D)
        self.block = Block(D)
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx)
        print(f"  tok_emb           out dtype = {x.dtype}")
        x = self.block(x)
        x = self.ln_f(x)
        print(f"  ln_f              out dtype = {x.dtype}")
        logits = self.head(x)
        print(f"  HEAD (unembedder) out dtype = {logits.dtype}")
        return logits


def run(label, ctx):
    print(f"\n=== {label} ===")
    m = Tiny().to("cuda")
    idx = torch.randint(0, 32768, (2, 16), device="cuda")
    with ctx:
        logits = m(idx)
    print(f"  final logits      out dtype = {logits.dtype}")
    print(f"  head.weight       storage   = {m.head.weight.dtype}")


run("WITHOUT autocast (fp32 reference)",
    torch.amp.autocast("cuda", enabled=False))

run("WITH autocast(bf16) — should show bf16 on all matmuls + head",
    torch.amp.autocast("cuda", dtype=torch.bfloat16))
