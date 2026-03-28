# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Global preparations

# %%
import sys
import os
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.notebook import tqdm

sys.path.insert(0, os.path.expanduser('~/spiky'))
device = 'cuda:6'
summation_dtype = torch.float32
random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.enabled = True
print(torch.__version__)

# %% [markdown]
# # Text data preparation
#
# Let's read tinyshakespeare.txt

# %%
CONTEXT_SIZE = 32

# %%
import gdown
url = 'https://drive.google.com/file/d/1vWjyIpU6wvCPtx2OdV_4M3MXX6FsOpxV/view'
output = 'fineweb_texts.txt'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False, fuzzy=True)
size = os.path.getsize(output)
if size < 233 * 1024 * 1024:
    raise RuntimeError(f'Download failed: file size {size/1024/1024:.1f} MB')

# %%
CONTEXT_SIZE = 32
RAW_VOCAB_SIZE = 256
BOS_ID = RAW_VOCAB_SIZE  # 256
VOCAB_SIZE = RAW_VOCAB_SIZE + 1  # 257

# %%
TESTING_LENGTH = 10_000

from spiky.util.text_snippet_sampler import TextSnippetSampler

snippet_sampler = TextSnippetSampler(
    'fineweb_texts.txt',
    CONTEXT_SIZE,
    TESTING_LENGTH,
    device,
    random_seed=random_seed,
)

# %%
snippet_sampler.sample_training_batch(2)

# %%
snippet_sampler.batch_to_text(snippet_sampler.sample_training_batch(4))

# %%
for test_batch in snippet_sampler.testing_batches_iterator(4):
    print(snippet_sampler.batch_to_text(test_batch))
    break

# %% [markdown]
# # Baseline character transformer

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


def build_sinusoidal_pe(max_len, d_model):
    position = torch.arange(max_len).float().unsqueeze(1)
    inv_freq = torch.exp(
        -torch.arange(0, d_model, 2).float() * (torch.log(torch.tensor(10000.0)) / d_model)
    )
    sinusoid = position * inv_freq
    pe = torch.empty(max_len, d_model)
    pe[:, 0::2] = torch.sin(sinusoid)
    pe[:, 1::2] = torch.cos(sinusoid)
    return pe


class SDPATransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.permute(0, 2, 1, 3).reshape(B, T, C)
        attn = self.out_proj(attn)
        x = x + self.norm1(self.dropout(attn))
        x = x + self.norm2(self.dropout(self.ffn(x)))
        return x


class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        d_model=256,
        n_heads=4,
        num_layers=6,
        max_len=512,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb.weight.data.uniform_(-1, 1)

        # self.register_buffer("pos_emb", build_sinusoidal_pe(max_len, d_model))
        self.register_buffer("pos_emb", torch.empty(max_len, d_model).uniform_(-1, 1))
        
        self.layers = nn.ModuleList([
            SDPATransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb[:T]
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


# %%
baseline_net = CharTransformer(max_len=CONTEXT_SIZE, vocab_size=VOCAB_SIZE)
baseline_net.to(device)
total = sum(p.numel() for p in baseline_net.parameters())
trainable = sum(p.numel() for p in baseline_net.parameters() if p.requires_grad)

print("Model: ", baseline_net)
print("total:", total)
print("trainable:", trainable)
print("frozen:", total - trainable)

# %%
import torch
import torch.nn.functional as F


def generate_text_lut(lut_model, prefix, length, device):
    # Model operates on byte IDs; decode final byte stream as UTF-8 for display.
    # Position 0 = BOS; positions 1..CONTEXT_SIZE-1 = context. We right-align context so
    # position -1 always holds the last byte (matches training where last position = last preceding byte).
    ctx = list(prefix.encode("utf-8"))

    for _ in range(length):
        x = torch.zeros([1, CONTEXT_SIZE], dtype=torch.long, device=device)
        x[0, 0] = BOS_ID
        trunc_ctx = ctx[-(CONTEXT_SIZE - 1):]  # at most CONTEXT_SIZE-1 bytes so we never overwrite BOS
        if len(trunc_ctx) > 0:
            # Right-align: last byte of context at position -1 so logits[:, -1, :] is conditioned correctly
            x[0, -len(trunc_ctx):] = torch.tensor(trunc_ctx, dtype=torch.long, device=device)

        logits = lut_model(x)
        # Exclude BOS from sampling: generated stream should be UTF-8 bytes only.
        probs = torch.softmax(logits[:, -1, :RAW_VOCAB_SIZE], dim=-1)[0]
        next_id = torch.multinomial(probs, 1).item()
        ctx.append(next_id)

    ctx_bytes = [c for c in ctx if 0 <= c < 256]
    return bytes(ctx_bytes).decode("utf-8", errors="replace")

def evaluate_model(model, sampler, B):
    """Full-sequence mean cross-entropy loss (standard language modeling)."""
    model.eval()
    losses = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(B):   # [B, C]
            # Prepend BOS at position 0
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            tgt = batch.long()

            logits = model(inp)   # [B, 32, 257]

            B_, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B_ * T, V),
                tgt.reshape(B_ * T),
                reduction='mean',
            )
            losses.append(loss.item())

    # ---- small generation demo ----
    prefix = "Once upon a time "
    gen = generate_text_lut(model, prefix, length=80, device=device)
    print("\n[GEN]:", gen, "\n")

    model.train()
    return sum(losses) / len(losses) #  / (CONTEXT_SIZE * B)


# %%
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

batch_size = 128
test_batch_size = 256

optimizer = optim.Adam(baseline_net.layers.parameters(), lr=0.0001)

n_steps=500000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_steps
)

# %%
test_loss = evaluate_model(baseline_net, snippet_sampler, test_batch_size)
test_loss

# %%
train_losses = []
test_losses = []

# %%
test_every=1000
train_loss = None
alpha = 0.01

pbar = tqdm(total=n_steps)
baseline_net.train()

for step in range(n_steps):
    x = snippet_sampler.sample_training_batch(batch_size).long()   # [B, 32] bytes
    inp = torch.empty(batch_size, x.shape[1], dtype=torch.long, device=x.device)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    tgt = x

    logits = baseline_net(inp)      # [B, C-1, 256]
    B, T, V = logits.shape

    # Train on full sequence with sum reduction (gradient from all positions)
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        tgt.reshape(B * T),
        reduction="mean"
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    loss_value = loss.item()
    train_loss = loss_value if train_loss is None else (1 - alpha) * train_loss + alpha * loss_value
    pbar.update(1)
    if step % 10 == 0:
        pbar.set_description(f"loss={train_loss:.4f}, lr {sched.get_last_lr()[0]:.6f}")

    if step % test_every == 0:
        test_loss = evaluate_model(baseline_net, snippet_sampler, test_batch_size)
        if len(train_losses) == 0 or step > 0:
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        print(f"[TEST] step {step}: loss={test_loss:.4f}")

# %%
import matplotlib.pyplot as plt

# assume train_losses and test_losses are Python lists of equal length

st = [i * 1000 for i in range(len(train_losses))]

plt.figure(figsize=(6,4))
plt.plot(st, train_losses, label="train")
plt.plot(st, test_losses, label="test")

plt.xlabel("steps")
plt.ylabel("loss")
plt.ylim(top=2.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Plot for CONTEXT_SIZE=32

# %%
import matplotlib.pyplot as plt

# assume train_losses and test_losses are Python lists of equal length

st = [i * 1000 for i in range(len(train_losses))]

plt.figure(figsize=(6,4))
plt.plot(st, train_losses, label="train")
plt.plot(st, test_losses, label="test")

plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Plot for CONTEXT_SIZE=64

# %%
import matplotlib.pyplot as plt

# assume train_losses and test_losses are Python lists of equal length

st = [i * 1000 for i in range(len(train_losses))]

plt.figure(figsize=(6,4))
plt.plot(st, train_losses, label="train")
plt.plot(st, test_losses, label="test")

plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
