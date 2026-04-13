"""
Quick vanilla baseline: 5000 steps, batch_size=32, constant lr=0.001.
Same architecture as exp002 (d_model=256, 4 heads, 6 layers, FFN 4x).
"""
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE

DEVICE = 'cuda:0'
N_STEPS = 5000
BATCH_SIZE = 32
LR = 0.001
torch.manual_seed(1)


class SDPATransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.permute(0,2,1,3).reshape(B, T, C)
        x = x + self.norm1(self.out_proj(attn))
        x = x + self.norm2(self.ffn(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, num_layers=6, ff_mult=4):
        super().__init__()
        self.token_emb = nn.Embedding(257, d_model)
        self.token_emb.weight.data.uniform_(-1, 1)
        self.register_buffer('pos_emb', torch.empty(CONTEXT_SIZE, d_model).uniform_(-1, 1))
        self.layers = nn.ModuleList([
            SDPATransformerLayer(d_model, n_heads, d_model * ff_mult)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, 257)

    def forward(self, x):
        h = self.token_emb(x) + self.pos_emb[:x.shape[1]]
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


model = CharTransformer().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Vanilla baseline: {n_params:,} params")

sampler = make_sampler(DEVICE, random_seed=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
ema = None

model.train()
t0 = time.time()
for step in range(N_STEPS):
    x = sampler.sample_training_batch(BATCH_SIZE).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lv = loss.item()
    ema = lv if ema is None else 0.99*ema + 0.01*lv

elapsed = time.time() - t0
print(f"loss@5k = {ema:.4f}, time = {elapsed:.0f}s")
