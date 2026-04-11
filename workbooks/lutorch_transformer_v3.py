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
# # LUT Transformer V3 (Softmax Attention)
#
# Based on exp184: V3 architecture with LUT score attention + value LUT + softmax@V + out_proj LUT.

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
warnings.filterwarnings(
    "ignore",
    message=".*has_cuda.*deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*has_cudnn.*deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*has_mps.*deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*has_mkldnn.*deprecated.*"
)

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
# V3 Softmax architecture per block:
# - Score LUT (LUTAttentionV3) → raw attention scores [B, T, T, H, 1]
# - Value LUT (per-token) → values [B, T, H, d_v]
# - softmax(scores) @ V → weighted values
# - LayerNorm → Out-proj LUT (H*d_v → E)
# - Residual + LayerNorm

# %%
cfg = {
    'embedding_dim': 32,
    'positional_dim': 16,
    'num_layers': 6,
    'n_heads': 4,
    'd_v': 16,
    'attention_nap': 10,
    'attention_tph': 32,
    'value_nap': 6,
    'value_tph': 128,
    'out_proj_nap': 8,
    'out_proj_tph': 128,
    'anchor_sampling_policy': 'full_coverage',
    'initial_weights_noise': 0.001,
}

# %%
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.lut_attention import LUTAttentionV3


def make_score_attn(seed_offset=0):
    E, P, H = cfg['embedding_dim'], cfg['positional_dim'], cfg['n_heads']
    lut = MultiHeadLut(
        input_dim=2*E+P, n_heads=H, n_outputs=1,
        n_anchor_pairs=cfg['attention_nap'], tables_per_head=cfg['attention_tph'],
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=42+seed_offset, device=device, recompute_in_backward=True,
    )
    return LUTAttentionV3(lut, seq_len=CONTEXT_SIZE, causal=True, include_diagonal=True)


def make_value_lut(seed_offset=200):
    E, H, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_v']
    return MultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['value_nap'], tables_per_head=cfg['value_tph'],
        smooth_mode=False, n_alternatives=3,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=42+seed_offset, device=device, recompute_in_backward=True,
    )


def make_out_proj(seed_offset=400):
    E, H, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_v']
    return MultiHeadLut(
        input_dim=H*d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_proj_nap'], tables_per_head=cfg['out_proj_tph'],
        smooth_mode=False, n_alternatives=3,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=42+seed_offset, device=device, recompute_in_backward=True,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        H, d_v = cfg['n_heads'], cfg['d_v']
        self.score_attn = make_score_attn(seed_offset=layer_idx)
        self.value_lut = make_value_lut(seed_offset=200+layer_idx)
        self.out_proj = make_out_proj(seed_offset=400+layer_idx)
        self.attn_norm = nn.LayerNorm(H*d_v)
        self.norm = nn.LayerNorm(cfg['embedding_dim'])
        self.H, self.d_v = H, d_v

    def forward(self, x, rel_pe):
        B, T, E = x.shape
        H, d_v = self.H, self.d_v
        raw_scores = self.score_attn(x, rel_pe).squeeze(-1).permute(0, 3, 1, 2)
        attn_weights = F.softmax(raw_scores, dim=-1)
        v = self.value_lut(x.reshape(B*T, E)).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn_out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, T, H*d_v)
        attn_out = self.attn_norm(attn_out)
        proj = self.out_proj(attn_out.reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, E)
        return x + self.norm(proj)


class LUTTransformerV3(nn.Module):
    def __init__(self):
        super().__init__()
        E, P = cfg['embedding_dim'], cfg['positional_dim']
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.rel_pe = nn.Parameter(torch.randn(CONTEXT_SIZE, P) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(cfg['num_layers'])])
        self.unembedder = nn.Linear(E, VOCAB_SIZE, bias=False)

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer in self.layers:
            x = layer(x, self.rel_pe)
        return self.unembedder(x)

# %% [markdown]
# ## Evaluation & generation helpers

# %%
def generate_text(model, prefix, length=80):
    ctx = list(prefix.encode("utf-8"))
    for _ in range(length):
        x = torch.zeros([1, CONTEXT_SIZE], dtype=torch.long, device=device)
        x[0, 0] = BOS_ID
        trunc = ctx[-(CONTEXT_SIZE-1):]
        if trunc:
            x[0, -len(trunc):] = torch.tensor(trunc, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits[:, -1, :256], dim=-1)[0]
        ctx.append(torch.multinomial(probs, 1).item())
    return bytes([c for c in ctx if 0 <= c < 256]).decode("utf-8", errors="replace")


def evaluate(model):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(128):
            inp = torch.empty_like(batch, dtype=torch.long)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch.long().reshape(B*T))
            losses.append(loss.item())
    val_loss = sum(losses) / len(losses)
    print(f"[GEN]: {generate_text(model, 'Once upon a time ')}")
    model.train()
    return val_loss

# %% [markdown]
# ## Training

# %%
model = LUTTransformerV3().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# %%
import math
from tqdm.notebook import tqdm

n_steps = 50000
batch_size = 64
lr = 0.001
warmup_fraction = 0.1

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# def get_lr_scale(step):
#     warmup = int(warmup_fraction * n_steps)
#     if step < warmup:
#         return step / max(warmup, 1)
#     progress = (step - warmup) / max(n_steps - warmup, 1)
#     return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

def get_lr_scale(step):
    return 1.0

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
    loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))

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

# %% [raw]
# # exp184, constant lr=0.001
#
# step      0 | loss=6.6045 | lr=1.00e-03
# [GEN]: Once upon a time   L     �  � `   �  L    d         j                  L                �    
# [VAL] step 0: 8.6556
# step    100 | loss=4.5302 | lr=1.00e-03
# step    200 | loss=3.2807 | lr=1.00e-03
# step    300 | loss=2.6523 | lr=1.00e-03
# step    400 | loss=2.3470 | lr=1.00e-03
# step    500 | loss=2.1804 | lr=1.00e-03
# step    600 | loss=2.0894 | lr=1.00e-03
# step    700 | loss=2.0403 | lr=1.00e-03
# step    800 | loss=1.9974 | lr=1.00e-03
# step    900 | loss=1.9616 | lr=1.00e-03
# step   1000 | loss=1.9376 | lr=1.00e-03
# [GEN]: Once upon a time is a sin system of this on inlefic can armo distry, atzioklome and supplier surq
# [VAL] step 1000: 1.9089
# step   1100 | loss=1.9175 | lr=1.00e-03
# step   1200 | loss=1.9039 | lr=1.00e-03
# step   1300 | loss=1.8859 | lr=1.00e-03
# step   1400 | loss=1.8684 | lr=1.00e-03
# step   1500 | loss=1.8625 | lr=1.00e-03
# step   1600 | loss=1.8423 | lr=1.00e-03
# step   1700 | loss=1.8370 | lr=1.00e-03
# step   1800 | loss=1.8311 | lr=1.00e-03
# step   1900 | loss=1.8141 | lr=1.00e-03
# step   2000 | loss=1.8089 | lr=1.00e-03
# [GEN]: Once upon a time there studiety on the death ord, and (the brequilved by developmently we enlves 
# [VAL] step 2000: 1.8036
# step   2100 | loss=1.8025 | lr=1.00e-03
# step   2200 | loss=1.8026 | lr=1.00e-03
# step   2300 | loss=1.7904 | lr=1.00e-03
# step   2400 | loss=1.7855 | lr=1.00e-03
# step   2500 | loss=1.7802 | lr=1.00e-03
# step   2600 | loss=1.7764 | lr=1.00e-03
# step   2700 | loss=1.7685 | lr=1.00e-03
# step   2800 | loss=1.7660 | lr=1.00e-03
# step   2900 | loss=1.7685 | lr=1.00e-03
# step   3000 | loss=1.7581 | lr=1.00e-03
# [GEN]: Once upon a time – whichemed the product that st tima worksheime. Mray used to grounded by the 
# [VAL] step 3000: 1.7527
# step   3100 | loss=1.7525 | lr=1.00e-03
# step   3200 | loss=1.7538 | lr=1.00e-03
# step   3300 | loss=1.7437 | lr=1.00e-03
# step   3400 | loss=1.7425 | lr=1.00e-03
# step   3500 | loss=1.7425 | lr=1.00e-03
# step   3600 | loss=1.7370 | lr=1.00e-03
# step   3700 | loss=1.7323 | lr=1.00e-03
# step   3800 | loss=1.7294 | lr=1.00e-03
# step   3900 | loss=1.7322 | lr=1.00e-03
# step   4000 | loss=1.7348 | lr=1.00e-03
# [GEN]: Once upon a time store gerstand, had expercented on stores in stand concluwer for yensors are sal
# [VAL] step 4000: 1.7253
# step   4100 | loss=1.7222 | lr=1.00e-03
# step   4200 | loss=1.7228 | lr=1.00e-03
# step   4300 | loss=1.7204 | lr=1.00e-03
# step   4400 | loss=1.7235 | lr=1.00e-03
# step   4500 | loss=1.7126 | lr=1.00e-03
# step   4600 | loss=1.7141 | lr=1.00e-03
# step   4700 | loss=1.7061 | lr=1.00e-03
# step   4800 | loss=1.7020 | lr=1.00e-03
# step   4900 | loss=1.6979 | lr=1.00e-03
# step   5000 | loss=1.7022 | lr=1.00e-03
# [GEN]: Once upon a time of Goog!pophape some during soil, Chocolom occunic modeling
# Alcohold unpuscent s
# [VAL] step 5000: 1.6986
# step   5100 | loss=1.6985 | lr=1.00e-03
# step   5200 | loss=1.6982 | lr=1.00e-03
# step   5300 | loss=1.6967 | lr=1.00e-03
# step   5400 | loss=1.6974 | lr=1.00e-03
# step   5500 | loss=1.6948 | lr=1.00e-03
# step   5600 | loss=1.6921 | lr=1.00e-03
# step   5700 | loss=1.6861 | lr=1.00e-03
# step   5800 | loss=1.6802 | lr=1.00e-03
# step   5900 | loss=1.6818 | lr=1.00e-03
# step   6000 | loss=1.6784 | lr=1.00e-03
# [GEN]: Once upon a time of work in at the endor proofslit piece, though it expate to thing on the balls 
# [VAL] step 6000: 1.6751
# step   6100 | loss=1.6819 | lr=1.00e-03
# step   6200 | loss=1.6806 | lr=1.00e-03
# step   6300 | loss=1.6775 | lr=1.00e-03
# step   6400 | loss=1.6688 | lr=1.00e-03
# step   6500 | loss=1.6639 | lr=1.00e-03
# step   6600 | loss=1.6698 | lr=1.00e-03
# step   6700 | loss=1.6677 | lr=1.00e-03
# step   6800 | loss=1.6678 | lr=1.00e-03
# step   6900 | loss=1.6722 | lr=1.00e-03
# step   7000 | loss=1.6607 | lr=1.00e-03
# [GEN]: Once upon a time color. In the intract is cut all it is used a fixed as in the flow that thisos t
# [VAL] step 7000: 1.6648
# step   7100 | loss=1.6640 | lr=1.00e-03
# step   7200 | loss=1.6663 | lr=1.00e-03
# step   7300 | loss=1.6604 | lr=1.00e-03
# step   7400 | loss=1.6573 | lr=1.00e-03
# step   7500 | loss=1.6564 | lr=1.00e-03
# step   7600 | loss=1.6617 | lr=1.00e-03
# step   7700 | loss=1.6596 | lr=1.00e-03
# step   7800 | loss=1.6580 | lr=1.00e-03
# step   7900 | loss=1.6554 | lr=1.00e-03
# step   8000 | loss=1.6523 | lr=1.00e-03
# [GEN]: Once upon a time soden are expicted to Applications/ort. They deader vest the moistriated thinkid
# [VAL] step 8000: 1.6587
# step   8100 | loss=1.6523 | lr=1.00e-03
# step   8200 | loss=1.6511 | lr=1.00e-03
# step   8300 | loss=1.6518 | lr=1.00e-03
# step   8400 | loss=1.6481 | lr=1.00e-03
# step   8500 | loss=1.6522 | lr=1.00e-03
# step   8600 | loss=1.6521 | lr=1.00e-03
# step   8700 | loss=1.6509 | lr=1.00e-03
# step   8800 | loss=1.6481 | lr=1.00e-03
# step   8900 | loss=1.6401 | lr=1.00e-03
# step   9000 | loss=1.6466 | lr=1.00e-03
# [GEN]: Once upon a time of eyellate negative larger).
# There are because fills answer face-igwa, which ca
# [VAL] step 9000: 1.6435
# step   9100 | loss=1.6368 | lr=1.00e-03
# step   9200 | loss=1.6365 | lr=1.00e-03
# step   9300 | loss=1.6364 | lr=1.00e-03
# step   9400 | loss=1.6431 | lr=1.00e-03
# step   9500 | loss=1.6348 | lr=1.00e-03
# step   9600 | loss=1.6354 | lr=1.00e-03
# step   9700 | loss=1.6395 | lr=1.00e-03
# step   9800 | loss=1.6355 | lr=1.00e-03
# step   9900 | loss=1.6332 | lr=1.00e-03
# step  10000 | loss=1.6343 | lr=1.00e-03
# [GEN]: Once upon a time of the meteralizething, form Brokes in the Icellen’s knowledge authority suica
# [VAL] step 10000: 1.6347
#

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

# %% [raw]
# cfg = {
#     'embedding_dim': 32,
#     'positional_dim': 16,
#     'num_layers': 6,
#     'n_heads': 4,
#     'd_v': 16,
#     'attention_nap': 10,
#     'attention_tph': 32,
#     'value_nap': 6,
#     'value_tph': 128,
#     'out_proj_nap': 8,
#     'out_proj_tph': 128,
#     'anchor_sampling_policy': 'full_coverage',
#     'initial_weights_noise': 0.001,
# }
#
# n_alternatives=1
#
# step      0 | loss=6.3837 | lr=1.00e-03
# [GEN]: Once upon a time ee             �        e e�   �  e�         � e/   ef        e     e� / �e   
# [VAL] step 0: 6.9000
# step    100 | loss=4.2812 | lr=1.00e-03
# step    200 | loss=3.1286 | lr=1.00e-03
# step    300 | loss=2.5705 | lr=1.00e-03
# step    400 | loss=2.2860 | lr=1.00e-03
# step    500 | loss=2.1316 | lr=1.00e-03
# step    600 | loss=2.0437 | lr=1.00e-03
# step    700 | loss=1.9798 | lr=1.00e-03
# step    800 | loss=1.9342 | lr=1.00e-03
# step    900 | loss=1.9076 | lr=1.00e-03
# step   1000 | loss=1.8761 | lr=1.00e-03
# [GEN]: Once upon a time it usther from the five use development the ares and oraym fet in more the a che
# [VAL] step 1000: 1.8572
# step   1100 | loss=1.8557 | lr=1.00e-03
# step   1200 | loss=1.8398 | lr=1.00e-03
# step   1300 | loss=1.8280 | lr=1.00e-03
# step   1400 | loss=1.8129 | lr=1.00e-03
# step   1500 | loss=1.8038 | lr=1.00e-03
# step   1600 | loss=1.7910 | lr=1.00e-03
# step   1700 | loss=1.7764 | lr=1.00e-03
# step   1800 | loss=1.7693 | lr=1.00e-03
# step   1900 | loss=1.7671 | lr=1.00e-03
# step   2000 | loss=1.7539 | lr=1.00e-03
# [GEN]: Once upon a time and: Boundant in my directed that a massey soy, by purployees, you conceptial tr
# [VAL] step 2000: 1.7546
# step   2100 | loss=1.7551 | lr=1.00e-03
# step   2200 | loss=1.7542 | lr=1.00e-03
# step   2300 | loss=1.7414 | lr=1.00e-03
# step   2400 | loss=1.7383 | lr=1.00e-03
# step   2500 | loss=1.7400 | lr=1.00e-03
# step   2600 | loss=1.7263 | lr=1.00e-03
# step   2700 | loss=1.7179 | lr=1.00e-03
# step   2800 | loss=1.7075 | lr=1.00e-03
# step   2900 | loss=1.7109 | lr=1.00e-03
# step   3000 | loss=1.7114 | lr=1.00e-03
# [GEN]: Once upon a time takage and the opensive the wount of than taught lighttial freen in Varily the r
# [VAL] step 3000: 1.7108
# step   3100 | loss=1.7089 | lr=1.00e-03
# step   3200 | loss=1.7089 | lr=1.00e-03
# step   3300 | loss=1.7040 | lr=1.00e-03
# step   3400 | loss=1.7004 | lr=1.00e-03
# step   3500 | loss=1.6935 | lr=1.00e-03
# step   3600 | loss=1.6963 | lr=1.00e-03
# step   3700 | loss=1.6875 | lr=1.00e-03
# step   3800 | loss=1.6872 | lr=1.00e-03
# step   3900 | loss=1.6898 | lr=1.00e-03
# step   4000 | loss=1.6906 | lr=1.00e-03
# [GEN]: Once upon a time Quron stone have the and systems are moderaterage numbership of labbasic of the 
# [VAL] step 4000: 1.6819
# step   4100 | loss=1.6892 | lr=1.00e-03
# step   4200 | loss=1.6863 | lr=1.00e-03
# step   4300 | loss=1.6804 | lr=1.00e-03
# step   4400 | loss=1.6761 | lr=1.00e-03
# step   4500 | loss=1.6812 | lr=1.00e-03
# step   4600 | loss=1.6793 | lr=1.00e-03
# step   4700 | loss=1.6696 | lr=1.00e-03
# step   4800 | loss=1.6651 | lr=1.00e-03
# step   4900 | loss=1.6637 | lr=1.00e-03
# step   5000 | loss=1.6669 | lr=1.00e-03
# [GEN]: Once upon a time kinsic the pops.
# Pajesi way I can any Cyprit Sraua, Deal,
# the Became goddes Rela
# [VAL] step 5000: 1.6645
# step   5100 | loss=1.6673 | lr=1.00e-03
# step   5200 | loss=1.6653 | lr=1.00e-03
# step   5300 | loss=1.6684 | lr=1.00e-03
# step   5400 | loss=1.6634 | lr=1.00e-03
# step   5500 | loss=1.6559 | lr=1.00e-03
# step   5600 | loss=1.6634 | lr=1.00e-03
# step   5700 | loss=1.6557 | lr=1.00e-03
# step   5800 | loss=1.6507 | lr=1.00e-03
# step   5900 | loss=1.6532 | lr=1.00e-03
# step   6000 | loss=1.6509 | lr=1.00e-03
# [GEN]: Once upon a time Xraw flevis swylery care no delg on what will be found through you couse while b
# [VAL] step 6000: 1.6487
# step   6100 | loss=1.6540 | lr=1.00e-03
# step   6200 | loss=1.6514 | lr=1.00e-03
# step   6300 | loss=1.6496 | lr=1.00e-03
# step   6400 | loss=1.6475 | lr=1.00e-03
# step   6500 | loss=1.6444 | lr=1.00e-03
# step   6600 | loss=1.6398 | lr=1.00e-03
# step   6700 | loss=1.6394 | lr=1.00e-03
# step   6800 | loss=1.6425 | lr=1.00e-03
# step   6900 | loss=1.6381 | lr=1.00e-03
# step   7000 | loss=1.6387 | lr=1.00e-03
# [GEN]: Once upon a time of blog the schild set with when the instant stopation worresners to know can st
# [VAL] step 7000: 1.6393
# step   7100 | loss=1.6434 | lr=1.00e-03
# step   7200 | loss=1.6382 | lr=1.00e-03
# step   7300 | loss=1.6382 | lr=1.00e-03
# step   7400 | loss=1.6433 | lr=1.00e-03
# step   7500 | loss=1.6427 | lr=1.00e-03
# step   7600 | loss=1.6345 | lr=1.00e-03
# step   7700 | loss=1.6380 | lr=1.00e-03
# step   7800 | loss=1.6327 | lr=1.00e-03
# step   7900 | loss=1.6276 | lr=1.00e-03
# step   8000 | loss=1.6226 | lr=1.00e-03
# [GEN]: Once upon a time shootes also both charge on their crawear is between temp within the first perma
# [VAL] step 8000: 1.6310
# step   8100 | loss=1.6272 | lr=1.00e-03
# step   8200 | loss=1.6290 | lr=1.00e-03
# step   8300 | loss=1.6307 | lr=1.00e-03
# step   8400 | loss=1.6337 | lr=1.00e-03
# step   8500 | loss=1.6337 | lr=1.00e-03
# step   8600 | loss=1.6287 | lr=1.00e-03
# step   8700 | loss=1.6281 | lr=1.00e-03
# step   8800 | loss=1.6273 | lr=1.00e-03
# step   8900 | loss=1.6223 | lr=1.00e-03
# step   9000 | loss=1.6220 | lr=1.00e-03
# [GEN]: Once upon a time data X ADisic natural Your Feature and cover apiking 51). “Ocal medicine findi
# [VAL] step 9000: 1.6205
# step   9100 | loss=1.6236 | lr=1.00e-03
# step   9200 | loss=1.6217 | lr=1.00e-03
# step   9300 | loss=1.6238 | lr=1.00e-03
# step   9400 | loss=1.6182 | lr=1.00e-03
# step   9500 | loss=1.6197 | lr=1.00e-03
# step   9600 | loss=1.6163 | lr=1.00e-03
# step   9700 | loss=1.6176 | lr=1.00e-03
# step   9800 | loss=1.6147 | lr=1.00e-03
# step   9900 | loss=1.6143 | lr=1.00e-03
# step  10000 | loss=1.6164 | lr=1.00e-03
# [GEN]: Once upon a time of around their people.org.
# I, Recent Reducial Size Amission `more to treas the 
# [VAL] step 10000: 1.6096
# step  10100 | loss=1.6161 | lr=1.00e-03
# step  10200 | loss=1.6180 | lr=1.00e-03
# step  10300 | loss=1.6134 | lr=1.00e-03
# step  10400 | loss=1.6170 | lr=1.00e-03
# step  10500 | loss=1.6177 | lr=1.00e-03
# step  10600 | loss=1.6107 | lr=1.00e-03
# step  10700 | loss=1.6035 | lr=1.00e-03
# step  10800 | loss=1.6053 | lr=1.00e-03
# step  10900 | loss=1.6023 | lr=1.00e-03
# step  11000 | loss=1.6091 | lr=1.00e-03
# [GEN]: Once upon a time concessoristics the psychotherapism undermarkets, it from the argun numbendyle t
# [VAL] step 11000: 1.6101
# step  11100 | loss=1.6105 | lr=1.00e-03
# step  11200 | loss=1.6017 | lr=1.00e-03
# step  11300 | loss=1.6056 | lr=1.00e-03
# step  11400 | loss=1.6078 | lr=1.00e-03
# step  11500 | loss=1.6063 | lr=1.00e-03
# step  11600 | loss=1.6022 | lr=1.00e-03
# step  11700 | loss=1.6017 | lr=1.00e-03
# step  11800 | loss=1.6058 | lr=1.00e-03
# step  11900 | loss=1.5997 | lr=1.00e-03
# step  12000 | loss=1.6036 | lr=1.00e-03
# [GEN]: Once upon a time just easy stre, development, rats, with Africa.
#
# (Nosting) short billiance Russi
# [VAL] step 12000: 1.6029
# step  12100 | loss=1.6031 | lr=1.00e-03
# step  12200 | loss=1.6045 | lr=1.00e-03
# step  12300 | loss=1.6047 | lr=1.00e-03
# step  12400 | loss=1.5989 | lr=1.00e-03
# step  12500 | loss=1.5966 | lr=1.00e-03
# step  12600 | loss=1.5968 | lr=1.00e-03
# step  12700 | loss=1.5963 | lr=1.00e-03
# step  12800 | loss=1.5958 | lr=1.00e-03
# step  12900 | loss=1.5990 | lr=1.00e-03
# step  13000 | loss=1.5978 | lr=1.00e-03
# [GEN]: Once upon a time lifew! Allenges adolescards, secondall knowledge is on. Conchanization to enter 
# [VAL] step 13000: 1.5992
# step  13100 | loss=1.5929 | lr=1.00e-03
# step  13200 | loss=1.5925 | lr=1.00e-03
# step  13300 | loss=1.5972 | lr=1.00e-03
# step  13400 | loss=1.5938 | lr=1.00e-03
# step  13500 | loss=1.5992 | lr=1.00e-03
# step  13600 | loss=1.5935 | lr=1.00e-03
# step  13700 | loss=1.5937 | lr=1.00e-03
# step  13800 | loss=1.5928 | lr=1.00e-03
# step  13900 | loss=1.5892 | lr=1.00e-03
# step  14000 | loss=1.5906 | lr=1.00e-03
# [GEN]: Once upon a time infall dystma (ppless in two payer was nearly actual publishment includeds.org/h
# [VAL] step 14000: 1.5920
# step  14100 | loss=1.5892 | lr=1.00e-03
# step  14200 | loss=1.5872 | lr=1.00e-03
# step  14300 | loss=1.5896 | lr=1.00e-03
# step  14400 | loss=1.5885 | lr=1.00e-03
# step  14500 | loss=1.5897 | lr=1.00e-03
# step  14600 | loss=1.5867 | lr=1.00e-03
# step  14700 | loss=1.5875 | lr=1.00e-03
# step  14800 | loss=1.5880 | lr=1.00e-03
# step  14900 | loss=1.5825 | lr=1.00e-03
# step  15000 | loss=1.5870 | lr=1.00e-03
# [GEN]: Once upon a time do with insult, to be substanted, last means to set as have a resulting in emiss
# [VAL] step 15000: 1.5899

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

# %%

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(val_steps, train_losses, label='train')
plt.plot(val_steps, val_losses, label='val')
plt.xlabel('steps')
plt.ylabel('loss')
plt.legend()
plt.ylim(top=2.0)
plt.xlim(left=0.0, right=14000.0)
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Final val loss: {val_losses[-1]:.4f}")

# %%
