# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fully Permutational LUT Transformer
#
# All Q/K/V/out_proj are PermutationalLut. RankAttention on Q/K (pair-dominance projection).
# V outputs dominance directly (return_dominance=True). Positional embeddings concatenated.
# Concat all layer outputs + MLP unembedder (no residuals).

# %%
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision('high')
import warnings
import torch._dynamo
torch._dynamo.config.cache_size_limit = 128
warnings.filterwarnings("ignore", message=".*has_cuda.*deprecated.*")
warnings.filterwarnings("ignore", message=".*has_cudnn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*has_mps.*deprecated.*")
warnings.filterwarnings("ignore", message=".*has_mkldnn.*deprecated.*")

device = 'cuda:0'
torch.manual_seed(42)
print(torch.__version__)

# %% [markdown]
# ## Data preparation

# %%
import gdown
url = 'https://drive.google.com/file/d/1vWjyIpU6wvCPtx2OdV_4M3MXX6FsOpxV/view'
output = 'fineweb_texts.txt'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False, fuzzy=True)

# %%
CONTEXT_SIZE = 32
VOCAB_SIZE = 257
BOS_ID = 256
TESTING_LENGTH = 10_000

from spiky.util.text_snippet_sampler import TextSnippetSampler
sampler = TextSnippetSampler('fineweb_texts.txt', CONTEXT_SIZE, TESTING_LENGTH, device, random_seed=1)

batch = sampler.sample_training_batch(2)
print(sampler.batch_to_text(batch))

# %% [markdown]
# ## Model definition
#
# Fully permutational architecture:
# - Q/K: PermutationalLut (concat pos_emb) → LayerNorm → RankAttention
# - V: PermutationalLut (return_dominance=True) → directly to SDPA as dominance pairs
# - After SDPA: Borda-aggregate back to rank vectors
# - out_proj: PermutationalLut (H*d_v → E)
# - Concat all layer outputs → MLP unembedder

# %%
cfg = {
    'embedding_dim': 32,
    'pos_dim': 16,
    'num_layers': 6,
    'n_heads': 4,
    'd_qk': 16,
    'd_v': 8,
    'qk_input_nap': 6,
    'qk_output_nap': 16,
    'qk_tph': 256,
    'v_input_nap': 6,
    'v_output_nap': 4,
    'v_tph': 256,
    'out_input_nap': 10,
    'out_output_nap': 32,
    'out_tph': 1024,
    'soft_mode': 'ste',
    'temperature': 0.1,
    'rank_attn_temperature': 0.1,
}

E = cfg['embedding_dim']
POS_DIM = cfg['pos_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
D_V_P = d_v * (d_v - 1) // 2  # 28
N_LAYERS = cfg['num_layers']
SOFT_MODE = cfg['soft_mode']
TEMP = cfg['temperature']
RANK_ATTN_TEMP = cfg['rank_attn_temperature']

# %%
from spiky.lutorch.permutational_lut import PermutationalLut
from spiky.lutorch.ranking_tools import RankAttention

PERM_KWARGS = dict(
    pair_mode='scrambled',
    soft_mode=SOFT_MODE,
    temperature=TEMP,
    device=device,
    recompute_in_backward=True,
    initial_weights_noise=0.001,
)


def _make_qk_perm(seed_offset):
    return PermutationalLut(
        n_inputs=E + POS_DIM, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'],
        random_seed=42 + seed_offset,
        **PERM_KWARGS,
    )


def _make_v_perm(seed_offset):
    return PermutationalLut(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=cfg['v_tph'],
        return_dominance=True,
        random_seed=42 + seed_offset,
        **PERM_KWARGS,
    )


def _make_out_perm(seed_offset):
    return PermutationalLut(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=cfg['out_tph'],
        random_seed=42 + seed_offset,
        **PERM_KWARGS,
    )


class FullPermBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_perm = _make_qk_perm(layer_idx)
        self.k_perm = _make_qk_perm(100 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_perm = _make_v_perm(200 + layer_idx)
        self.rank_attn = RankAttention(
            d_qk=d_qk, d_v=D_V_P,
            smooth_mode=False,
            temperature=RANK_ATTN_TEMP,
            sdpa_temperature=1.0,
            sdpa_forward_temperature=1.0,
        )
        self.out_proj = _make_out_perm(400 + layer_idx)
        self.out_norm = nn.LayerNorm(E)
        self.register_buffer('borda_m', self.v_perm.dom_borda_m.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = torch.cat([x, pos_emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        q = self.q_perm(xp.reshape(B * T, -1))
        k = self.k_perm(xp.reshape(B * T, -1))
        q = self.q_norm(q.reshape(B, T, H, d_qk).permute(0, 2, 1, 3))
        k = self.k_norm(k.reshape(B, T, H, d_qk).permute(0, 2, 1, 3))

        v_dom = self.v_perm(x.reshape(B * T, _E))
        v_dom = v_dom.reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)

        attn_dom = self.rank_attn(q, k, v_dom, is_causal=True)
        attn = torch.einsum('bhtp,kp->bhtk', attn_dom, self.borda_m)

        out = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v))
        out = self.out_norm(out.squeeze(1).reshape(B, T, E))
        return out


class FullPermTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(CONTEXT_SIZE, POS_DIM) * 0.1) for _ in range(N_LAYERS)]
        )
        self.layers = nn.ModuleList([FullPermBlock(i) for i in range(N_LAYERS)])
        concat_dim = E * N_LAYERS
        self.unembedder = nn.Sequential(
            nn.Linear(concat_dim, concat_dim * 4),
            nn.ReLU(),
            nn.Linear(concat_dim * 4, VOCAB_SIZE),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        return self.unembedder(torch.cat(outs, dim=-1))

# %% [markdown]
# ## Evaluation & generation helpers

# %%
RAW_VOCAB_SIZE = 256


def generate_text(model, prefix, length=80):
    ctx = list(prefix.encode("utf-8"))
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            trunc = ctx[-(CONTEXT_SIZE - 1):]
            x = torch.zeros([1, CONTEXT_SIZE], dtype=torch.long, device=device)
            x[0, 0] = BOS_ID
            x[0, 1:1+len(trunc)] = torch.tensor(trunc, dtype=torch.long, device=device)
            pos = len(trunc)
            logits = model(x)
            probs = torch.softmax(logits[0, pos, :RAW_VOCAB_SIZE], dim=-1)
            ctx.append(torch.multinomial(probs, 1).item())
    model.train()
    return bytes(c for c in ctx if 0 <= c < 256).decode("utf-8", errors="replace")


def evaluate(model, batch_size=128):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(batch_size):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), batch.long().reshape(B * T))
            losses.append(loss.item())
    val_loss = sum(losses) / len(losses)
    print(f"[GEN]: {generate_text(model, 'Once upon a time ')}")
    model.train()
    return val_loss

# %% [markdown]
# ## Training

# %%
model = FullPermTransformer().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
print(f"Embedding: {E}, pos_dim: {POS_DIM} (concat)")
print(f"Q/K PermLut: in_nap={cfg['qk_input_nap']} out_nap={cfg['qk_output_nap']} tph={cfg['qk_tph']} d_qk={d_qk}")
print(f"V PermLut: in_nap={cfg['v_input_nap']} out_nap={cfg['v_output_nap']} tph={cfg['v_tph']} d_v={d_v} (dominance P={D_V_P})")
print(f"Out PermLut: in_nap={cfg['out_input_nap']} out_nap={cfg['out_output_nap']} tph={cfg['out_tph']}")

# %%
import math
from tqdm.notebook import tqdm

n_steps = 25000
batch_size = 64
lr = 0.001
warmup_fraction = 0.1

dense_params = [p for n, p in model.named_parameters() if 'unembedder' in n]
perm_params = [p for n, p in model.named_parameters() if 'unembedder' not in n]
optimizer = torch.optim.Adam([
    {'params': dense_params, 'lr': lr * 3},
    {'params': perm_params, 'lr': lr},
])


def get_lr_scale(step):
    warmup = int(warmup_fraction * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

train_losses, val_losses, val_steps = [], [], []
ema = None

model.train()
for step in tqdm(range(n_steps)):
    x = sampler.sample_training_batch(batch_size).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]

    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), x.reshape(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    lv = loss.item()
    ema = lv if ema is None else 0.99 * ema + 0.01 * lv

    if step % 100 == 0:
        print(f"step {step:6d} | loss={ema:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    if step % 1000 == 0:
        val = evaluate(model)
        print(f"[VAL] step {step}: {val:.4f}")
        train_losses.append(ema)
        val_losses.append(val)
        val_steps.append(step)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(val_steps, train_losses, label='train')
plt.plot(val_steps, val_losses, label='val')
plt.xlabel('steps')
plt.ylabel('loss')
plt.legend()
plt.ylim(top=2.0)
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Final val loss: {val_losses[-1]:.4f}")
