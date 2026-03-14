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
# # Global preparations

# %%
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image
from tqdm.notebook import tqdm

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

device = 'cuda:7'
summation_dtype = torch.float32
random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.enabled = True
print(torch.__version__)

# %% [markdown]
# # Text data preparation
#
# Let's read small fineweb fragment

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
# Match spike_QK: BOS at index 256, vocab 257 (bytes 0-255 + BOS)
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
batch = snippet_sampler.sample_training_batch(4)
batch.shape
snippet_sampler.batch_to_text(batch)

# %%
for test_batch in snippet_sampler.testing_batches_iterator(4):
    print(test_batch.shape)
    print(snippet_sampler.batch_to_text(test_batch))
    break

# %% [markdown]
# # LUTTransformer (lutorch sketch)

# %%
# LUTTransformer sketch using lutorch: MultiHeadLut, LUTAttention
import torch.nn as nn
from dataclasses import dataclass
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_attention import LUTAttention, PairProcessingConfig, PairProcessingMode
from spiky.lutorch.lut_helpers import UncertaintyMode


@dataclass(frozen=True)
class LUTTransformerConfig:
    """Full configuration for LUTTransformer (embedding, layers, block params, temperatures)."""
    vocab_size: int = VOCAB_SIZE
    embedding_dim: int = 32
    num_layers: int = 6
    num_heads: int = 4
    n_anchor_pairs_attn: int = 14
    n_anchor_pairs_ffn: int = 14
    n_positional_buckets: int = 8
    tables_per_head_attn: int = 32
    tables_per_head_value: int = 16
    ffn_tables: int = 16
    dropout: float = 0.0
    smooth_mode: bool = True
    device: object = device
    connected_anchors_mode: bool = False
    random_seed: object = 42
    attention_temperature: float = 0.25
    embedding_temperature: float = 0.1
    initial_weights_noise: float = 0.001
    uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1
    pair_config: PairProcessingConfig = PairProcessingConfig(c1=1.0, c2=-2.0)

    def __post_init__(self):
        assert (self.embedding_dim % self.num_heads) == 0


class LUTTransformer(nn.Module):
    """Transformer with lutorch primitives: MultiHeadLut + LUTAttention."""

    class Block(nn.Module):
        """Single transformer block: LUT cross-attention + value projection + FFN."""

        def __init__(self, c: LUTTransformerConfig):
            super().__init__()
            self.cross_attn = LUTAttention(
                MultiHeadLut(
                    input_dim=c.embedding_dim,
                    n_heads=c.num_heads,
                    n_outputs=1,
                    n_anchor_pairs=c.n_anchor_pairs_attn,
                    tables_per_head=c.tables_per_head_attn,
                    n_buckets=c.n_positional_buckets,
                    smooth_mode=c.smooth_mode,
                    device=c.device,
                    connected_anchors_mode=c.connected_anchors_mode,
                    random_seed=c.random_seed,
                    initial_weights_noise=c.initial_weights_noise,
                    uncertainty_mode=c.uncertainty_mode,
                ),
                causal=True,
                include_diagonal=False,
                attention_temperature=c.attention_temperature,
                n_positional_buckets=c.n_positional_buckets,
                pair_config=c.pair_config,
            )
            self.value_lut = MultiHeadLut(
                input_dim=c.embedding_dim,
                n_heads=c.num_heads,
                n_outputs=c.embedding_dim // c.num_heads,
                n_anchor_pairs=c.n_anchor_pairs_attn,
                tables_per_head=c.tables_per_head_value,
                smooth_mode=c.smooth_mode,
                device=c.device,
                connected_anchors_mode=c.connected_anchors_mode,
                random_seed=c.random_seed,
                initial_weights_noise=c.initial_weights_noise,
                uncertainty_mode=c.uncertainty_mode,
            )
            self.attn_dropout = nn.Dropout(c.dropout)
            self.ffn = MultiHeadLut(
                input_dim=c.embedding_dim,
                n_heads=1,
                n_outputs=c.embedding_dim,
                n_anchor_pairs=c.n_anchor_pairs_ffn,
                tables_per_head=c.ffn_tables,
                smooth_mode=c.smooth_mode,
                device=c.device,
                connected_anchors_mode=c.connected_anchors_mode,
                random_seed=c.random_seed,
                initial_weights_noise=c.initial_weights_noise,
                uncertainty_mode=c.uncertainty_mode,
            )
            self.ffn_dropout = nn.Dropout(c.dropout)

        def forward(self, z):
            B, S, E = z.shape
            attn_weights = self.cross_attn(z, z)  # [B, S, S, H]
            v = self.value_lut(z.reshape(-1, E))  # [B*S, H, E//H]
            H = v.shape[1]
            v = v.reshape(B, S, H, -1)  # [B, S, H, E//H]
            attn_out = attn_weights.permute(0, 3, 1, 2) @ v.permute(0, 2, 1, 3)  # [B, H, S, E//H]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, E)  # [B, S, E]
            z = z + self.attn_dropout(attn_out)  # [B, S, E]
            ffn_out = self.ffn(z.reshape(-1, E)).reshape(B, S, -1)  # [B, S, E]
            z = z + self.ffn_dropout(ffn_out)  # [B, S, E]
            return z

    def __init__(self, c: LUTTransformerConfig = LUTTransformerConfig()):
        super().__init__()
        self.config = c
        with torch.no_grad():
            self.token_embedder = nn.Embedding(c.vocab_size, c.embedding_dim, device=c.device)
            self.token_embedder.weight.copy_(torch.randn(self.token_embedder.weight.shape, device=c.device) * 0.1)
        self.layers = nn.ModuleList([LUTTransformer.Block(c) for _ in range(c.num_layers)])

    def forward(self, tokens):
        z = self.token_embedder(tokens)  # [B, S, E]
        for layer in self.layers:
            z = layer(z)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        logits = z @ self.token_embedder.weight.T / self.config.embedding_temperature  # [B, S, E] @ [E, V] -> [B, S, V]
        return logits


# %%
lut_transformer = None
optimizer = None
sched = None
if device != 'cpu':
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
lut_transformer = LUTTransformer()
print(lut_transformer)

# %%
total = sum(p.numel() for p in lut_transformer.parameters())
trainable = sum(p.numel() for p in lut_transformer.parameters() if p.requires_grad)

print("total:", total)
print("trainable:", trainable)
print("frozen:", total - trainable)

# %%
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

SCALE = 5.0
MAX_RATE = 0.01
def lr_func(t):
    return min(MAX_RATE, SCALE / (1 + t)**0.5)

print(f"Crossover point for LR: {(SCALE / MAX_RATE )**2:,}")

lr = 1.0
optimizer = optim.SGD(lut_transformer.layers.parameters(), lr=lr)
# Match spike_QK: embedder uses global LR scale (0.01); 0.001 was 10x too small
optimizer_embedder = optim.Adam(lut_transformer.token_embedder.parameters(), lr=0.01)

steps=1000000
# sched = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=steps
# )
# sched = None
sched = LambdaLR(optimizer, lr_lambda=lr_func)
# sched_embedder = LambdaLR(optimizer_embedder, lr_lambda=lr_func)
# LUTTransformerLutorch has no set_external_learning_rate_hook

# %%
lut_train_losses = []
lut_test_losses = []

# %%
import torch
import torch.nn.functional as F

test_batch_size = 128

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
            # Prepend BOS at position 0 (match spike_QK)
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
test_loss = evaluate_model(lut_transformer, snippet_sampler, test_batch_size)
test_loss

# %%
test_every=1000
train_loss = None
alpha = 0.01
batch_size = 128

pbar = tqdm(total=steps)
lut_transformer.train()

for step in range(0, steps + 1):
    x = snippet_sampler.sample_training_batch(batch_size)   # [B, 32] bytes
    x = x.long() if x.dtype != torch.long else x
    # Match spike_QK: position 0 = BOS, positions 1..31 = x[:,0..30]; targets = all 32 bytes
    inp = torch.empty(batch_size, x.shape[1], dtype=torch.long, device=x.device)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    tgt = x

    logits = lut_transformer(inp)      # [B, C-1, 256]
    B, T, V = logits.shape

    # Train on full sequence with sum reduction (gradient from all positions)
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        tgt.reshape(B * T),
        reduction="sum",
    )

    optimizer.zero_grad()
    optimizer_embedder.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer_embedder.step()
    if sched is not None:
        for _ in range(x.shape[0]):
            sched.step()  # once per batch (was range(x.shape[0]) = 128 steps/batch, making cosine decay 128x too fast)

    # Report full-sequence mean loss (same as above, just mean instead of sum)
    loss_value = loss.item() / (B * T)
    train_loss = loss_value if train_loss is None else (1 - alpha) * train_loss + alpha * loss_value
    pbar.update(1)
    if step % 10 == 0:
        pbar.set_description(f"loss={train_loss:.4f}, lr {lr if sched is None else sched.get_last_lr()[0]:.8f}")

    if step % test_every == 0:
        test_loss = evaluate_model(lut_transformer, snippet_sampler, test_batch_size)
        if len(lut_train_losses) == 0 or step > 0:
            lut_train_losses.append(train_loss)
            lut_test_losses.append(test_loss)
        print(f"[TEST] step {step}: loss={test_loss:.4f}")
#         if step > 0 and batch_size < 384:
#             print(f"batch_size {batch_size} -> {batch_size + 32}")
#             batch_size += 32

# %%
import matplotlib.pyplot as plt

# assume train_losses and test_losses are Python lists of equal length

steps = [i * 1000 for i in range(len(lut_train_losses))]

plt.figure(figsize=(6,4))
plt.plot(steps, lut_train_losses, label="train")
plt.plot(steps, lut_test_losses, label="test")
plt.ylim(top=2.0)
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
test_loss

# %%
import torch
from torch.profiler import profile, record_function, ProfilerActivity

profile_steps = 200

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(profile_steps):

        with record_function("sample_batch"):
            x = snippet_sampler.sample_training_batch(batch_size)
            x = x.long() if x.dtype != torch.long else x
            inp = torch.empty(batch_size, x.shape[1], dtype=torch.long, device=x.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = x[:, :-1]
            tgt = x

        with record_function("forward"):
            logits = lut_transformer(inp)

        with record_function("loss"):
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                tgt.reshape(B * T),
                reduction="none"
            ).sum()

        with record_function("backward+step"):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with record_function("scheduler"):
            if sched is not None:
                for _ in range(x.shape[0]):
                    sched.step()

        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
prof.export_chrome_trace("trace.json")

# %%
steps=1000000


# %%
from spiky.lutorch.lut_helpers import logarithmic_pe_buckets, rpe_matrix

# %%
pe_buckets = logarithmic_pe_buckets(8, 32, device)

# %%
pe_buckets

# %%
rpe_matrix(pe_buckets, 32, device).T

# %%
causal_mask = torch.tril(torch.ones(32, 32, device=device), diagonal=-1).unsqueeze(0).unsqueeze(-1)
causal_mask == 0

# %%
attention_scores = torch.zeros([2, 32, 32, 1], device=device)

# %%
attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf')).squeeze(3)    

# %%
F.softmax(attention_scores[:, 0])

# %%
attention_scores.shape

# %%
seq_len = 32
batch_size = 3
rows_local, cols_local = torch.tril_indices(
    seq_len, seq_len, offset=-1, device=device
)  # [num_pairs_single]

offsets = torch.arange(batch_size, device=device) * seq_len  # [B]
_cached_batched_rows = (
    rows_local.unsqueeze(0) + offsets.unsqueeze(1)
).reshape(-1)  # [P], where P = B * num_pairs_single
_cached_batched_cols = (
    cols_local.unsqueeze(0).expand(batch_size, -1)
).reshape(-1)  # [P]

# Within-sequence key indices for scattering into [B*S, S, H]
_cached_key_indices = (_cached_batched_cols % seq_len).contiguous()  # [P]

pe_buckets = logarithmic_pe_buckets(8, seq_len, device)
rpe = rpe_matrix(pe_buckets, seq_len, device)  # [S, S]
rpe_pairs = rpe[rows_local, cols_local]  # [num_pairs_single]
_cached_bucket_indices = rpe_pairs.repeat(batch_size).contiguous()  # [P]

# %%
_cached_batched_rows[:16], _cached_batched_cols[:16]

# %%
_cached_batched_rows[-16:], _cached_batched_cols[-16:]

# %%
H = 28
W = 28
kH = 5
kW = 5
sH = 1
sW = 1
index_grid = torch.arange(H * W, device=device, dtype=torch.long).view(1, 1, H, W)

# Unfold directly (zero padding, dilation=1).
# Result shape: [1, K, n_patches] where K = kH * kW
patches = F.unfold(
    index_grid.to(dtype=torch.float32),
    kernel_size=(kH, kW),
    dilation=1,
    stride=(sH, sW),
)
patches = patches.to(dtype=torch.long).transpose(1, 2).contiguous()  # [n_patches, K]

# anchor_candidates: [n_heads * tables_per_head, K] (all indices are valid)
K = patches.shape[1]
anchor_candidates = patches.repeat_interleave(4, dim=0).to(device=device)
anchor_candidates.shape


# %%
576 ** 0.5

# %%
patches.shape

# %%
index_grid.shape

# %%
anchor_candidates[0][1], anchor_candidates[0][2]

# %%
