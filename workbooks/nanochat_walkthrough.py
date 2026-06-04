# %% [markdown]
# # Nanochat Walkthrough
#
# End-to-end walkthrough of the nanochat pipeline:
# 1. **Dataset** — inspect ClimbMix parquet shards, understand the train/val split
# 2. **Tokenisation** — train a BPE tokenizer, then apply it on the fly with BOS-aligned best-fit packing
# 3. **Training** — train a small GPT model with tqdm progress, periodic validation (bits-per-byte), and a text generation demo

# %%
import os, sys, math, time
import torch
import torch._dynamo
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# Disable torch.compile for the notebook — the fused optimizer kernel requires
# extra VRAM during compilation warmup which can OOM on a busy / small GPU.
# Eager mode is slightly slower but perfectly fine for this small demo model.
torch._dynamo.config.disable = True

# Make sure we run from the nanochat project root
PROJECT_ROOT = os.path.dirname(os.path.abspath("walkthrough.py"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nanochat.common import get_base_dir, COMPUTE_DTYPE, get_dist_info, print0
from nanochat.dataset import list_parquet_files, DATA_DIR, parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_tokenizer, get_token_bytes

BASE_DIR = get_base_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  |  Compute dtype: {COMPUTE_DTYPE}")
print(f"Base dir: {BASE_DIR}")

# %% [markdown]
# ---
# ## Part 1 — Dataset
#
# ClimbMix-400B is stored as a collection of parquet shards.
# The **last shard** is always reserved as the validation set; all others are training data.

# %%
parquet_paths = list_parquet_files()
train_paths = parquet_paths[:-1]
val_paths   = parquet_paths[-1:]

print(f"Total shards on disk : {len(parquet_paths)}")
print(f"Train shards         : {len(train_paths)}")
print(f"Val shard            : {os.path.basename(val_paths[0])}")

# %%
# Inspect the first training shard
pf = pq.ParquetFile(train_paths[0])
print(f"Shard : {os.path.basename(train_paths[0])}")
print(f"Row groups : {pf.num_row_groups}")
rg0 = pf.read_row_group(0)
print(f"Rows in first row-group : {rg0.num_rows}")
print(f"Columns : {rg0.schema.names}")

# %%
# Show a few sample documents
docs = rg0.column("text").to_pylist()
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Document {i} ({len(doc):,} chars) ---")
    print(doc[:400])
    print("...")

# %% [markdown]
# ---
# ## Part 2 — Tokenisation
#
# ### 2a. Train a BPE tokenizer
#
# The tokenizer is GPT-4-style BPE, trained with `rustbpe` (fast Rust implementation)
# and wrapped in `tiktoken` for efficient inference.
#
# > **Note:** if a tokenizer is already saved at `~/.cache/nanochat/tokenizer/`, this step
# > is skipped and the cached one is loaded instead.

# %%
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
VOCAB_SIZE     = 32768   # 2^15 — sweet spot for models of this size
MAX_CHARS      = 50_000_000  # 50 M chars is enough to produce a decent tokenizer quickly
DOC_CAP        = 10_000  # truncate very long documents before training

if os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.pkl")):
    print("Tokenizer already on disk — loading.")
    tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
else:
    print(f"Training tokenizer on up to {MAX_CHARS:,} chars  (vocab_size={VOCAB_SIZE:,}) ...")

    def text_iterator():
        nchars = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc = doc[:DOC_CAP]
                nchars += len(doc)
                yield doc
                if nchars >= MAX_CHARS:
                    return

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), VOCAB_SIZE)
    print(f"Tokenizer trained in {time.time() - t0:.1f}s")
    tokenizer.save(TOKENIZER_DIR)

# Generate token_bytes.pt if missing (needed for bits-per-byte evaluation).
# This maps each token id to the number of UTF-8 bytes it represents (0 for special tokens).
token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
if not os.path.exists(token_bytes_path):
    print("Generating token_bytes.pt ...")
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes_list = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes_tensor, f)
    print(f"Saved token_bytes.pt  ({vocab_size:,} entries)")

print(f"Vocab size: {tokenizer.get_vocab_size():,}")
print(f"Special tokens: {tokenizer.get_special_tokens()}")

# %% [markdown]
# ### 2b. Encode / decode round-trip

# %%
sample = "Hello world! The quick brown fox jumps over the lazy dog. 2024-03-26"
ids = tokenizer.encode(sample)
decoded = tokenizer.decode(ids)
print(f"Text   : {sample!r}")
print(f"Token IDs ({len(ids)} tokens): {ids}")
print(f"Decoded: {decoded!r}")
assert decoded == sample, "Round-trip mismatch!"
print("Round-trip OK ✓")

# %% [markdown]
# ### 2c. BOS-aligned best-fit packing
#
# The dataloader packs multiple variable-length documents into fixed-length rows of `T+1` tokens.
#
# **Algorithm per row:**
# 1. Every row starts with `<|bos|>`
# 2. Pick the **largest** document from the buffer that still fits in the remaining space
# 3. Repeat until nothing fits, then **crop** the shortest document to fill exactly
#
# Result: 100% utilisation (no padding), ~35% of tokens come from cropped documents.

# %%
# Build a single batch to inspect the packing
B, T = 2, 128   # tiny sizes just to visualise
loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, B, T, split="train", device="cpu")
x, y = next(loader)

BOS_ID = tokenizer.get_bos_token_id()
print(f"Input shape : {x.shape}  (B={B}, T={T})")
print(f"BOS token id: {BOS_ID}")
for row_idx in range(B):
    bos_positions = (x[row_idx] == BOS_ID).nonzero(as_tuple=True)[0].tolist()
    print(f"\nRow {row_idx}: BOS at positions {bos_positions}  => {len(bos_positions)} documents packed")
    print("  Tokens[0:20]:", x[row_idx, :20].tolist())

# %% [markdown]
# ---
# ## Part 3 — Training
#
# ### 3a. GPT model definition
#
# The full architecture inlined for reference. Key design choices:
# - **RoPE** positional encoding (no learned position embeddings)
# - **QK-Norm** on queries and keys for training stability
# - **ReLU²** activation in MLP
# - **Value residual** (ResFormer): raw token value embeddings mixed into V on alternating layers
# - **Sliding window attention**: alternating short/long context pattern across layers
# - **Smear gate**: cheap bigram — bleeds previous token embedding into current position
# - **x0 residual**: each layer blends initial embedding back in (decaying per-layer scalars)
# - **Backout**: subtracts mid-network residual before final projection
# - **Softcap**: tanh-squashes logits to ±15 before loss
# - Untied `wte` / `lm_head` weights

# %%
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanochat.flash_attention import flash_attn
from nanochat.optim import MuonAdamW, DistMuonAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    # Sliding window pattern tiled across layers. L=full context, S=quarter context.
    # Final layer is always L regardless of pattern.
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Master weights stay fp32 for optimizer precision; matmuls run in activation dtype."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """True if this layer gets a Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.n_embd // config.n_head
        self.c_q    = Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.c_k    = Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v    = Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        # Value residual gate: learned per-head scalar in (0, 3)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, config.n_kv_head, bias=False) \
                       if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual: mix raw token value embedding into V
        if ve is not None:
            ve   = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B,T,n_kv_head)
            v    = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)   # QK-Norm
        q, k = q * 1.2, k * 1.2  # sharpen attention (split scale between Q and K)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())  # ReLU²


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h":   nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head      = Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate    = Linear(24, 1, bias=False)
        self.smear_lambda  = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, std=0.001)
        n, s = self.config.n_embd, 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(n_layer - 1, 1)
            self.x0_lambdas.data[i]    = 0.20 - 0.15 * i / max(n_layer - 1, 1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.config.n_embd // self.config.n_head)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = freqs.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w  = config.sequence_len
        short_w = -(-long_w // 4 // 128) * 128  # ceil to FA3 tile size
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)  # final layer always full context
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5):
        ddp, rank, local_rank, world_size = get_dist_info()
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        matrix_params       = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params    = list(self.transformer.wte.parameters())
        lm_head_params      = list(self.lm_head.parameters())
        resid_params        = [self.resid_lambdas]
        x0_params           = [self.x0_lambdas]
        smear_params        = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]

        param_groups = [
            dict(kind='adamw', params=lm_head_params,      lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params,    lr=embedding_lr   * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr   * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params,        lr=scalar_lr * 0.01, betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params,           lr=scalar_lr,        betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params,        lr=0.2,              betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))

        # Smear: bleed previous token's embedding into current position
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x
        n_layer = self.config.n_layer
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x  = block(x, ve, cos_sin, self.window_sizes[i])
            if i == n_layer // 2:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = torch.Generator(device=device).manual_seed(seed) if temperature > 0 else None
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1, :]
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                next_id = torch.multinomial(F.softmax(logits / temperature, dim=-1),
                                            num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            yield next_id.item()

# %% [markdown]
# ### 3b. Build the model

# %%
DEPTH = 6
SEQ_LEN = 512
DEVICE_BATCH_SIZE = 8   # micro-batch size (sequences per step)
TOTAL_BATCH_SIZE = 4096  # tokens per gradient step (determines grad accumulation)
NUM_ITERATIONS = 8000
EVAL_EVERY = 50          # evaluate on val set every N steps
EVAL_STEPS = 10          # how many val batches to average

model_dim = DEPTH * 64
num_heads = model_dim // 64

config = GPTConfig(
    sequence_len=SEQ_LEN,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=DEPTH,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim,
    window_pattern="SSSL",
)

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=DEVICE)
model.init_weights()
model.train()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model  : depth={DEPTH}, dim={model_dim}, heads={num_heads}")
print(f"Params : {total_params:,}")
print(f"Device : {DEVICE}  |  dtype: {COMPUTE_DTYPE}")

# %% [markdown]
# ### 3b. Optimizer and LR schedule
#
# Uses the built-in `setup_optimizer()` which creates a mixed Muon (matrix params)
# + AdamW (embeddings, scalars) optimizer.

# %%
optimizer = model.setup_optimizer()

tokens_per_step = DEVICE_BATCH_SIZE * SEQ_LEN
grad_accum_steps = max(1, TOTAL_BATCH_SIZE // tokens_per_step)
print(f"Tokens/micro-batch : {tokens_per_step:,}")
print(f"Grad accum steps   : {grad_accum_steps}")
print(f"Effective batch    : {grad_accum_steps * tokens_per_step:,} tokens")

def get_lr_scale(step, num_steps, warmup=0.1):
    """Linear warmup + cosine decay to 10% of peak."""
    w = int(warmup * num_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(num_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

# %% [markdown]
# ### 3c. Dataloaders

# %%
train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, split="train", device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, split="val", device=DEVICE
)

token_bytes = get_token_bytes(device=DEVICE)

# %% [markdown]
# ### 3d. Training loop

# %%
train_losses = []
val_bpbs     = []
val_steps    = []

pbar = tqdm(range(1, NUM_ITERATIONS + 1), desc="Training", unit="step")

for step in pbar:
    # ---- LR update ----
    lr_scale = get_lr_scale(step, NUM_ITERATIONS)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_scale

    # ---- Gradient accumulation ----
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(grad_accum_steps):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum_steps).backward()
        accum_loss += loss.item() / grad_accum_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    train_losses.append(accum_loss)

    pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr_scale:.3f}")

    # ---- Periodic validation ----
    if step % EVAL_EVERY == 0 or step == NUM_ITERATIONS:
        model.eval()
        val_loader = val_loader_factory()
        bpb = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
        val_bpbs.append(bpb)
        val_steps.append(step)
        tqdm.write(f"  Step {step:04d} | train_loss={accum_loss:.4f} | val_bpb={bpb:.4f}")
        model.train()

print("Training complete.")

# %%
# Plot loss curves (if matplotlib is available)
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, alpha=0.6, label="train loss")
    ax1.set(xlabel="step", ylabel="cross-entropy loss", title="Training Loss")
    ax1.legend()

    ax2.plot(val_steps, val_bpbs, "o-", color="tab:orange", label="val bpb")
    ax2.set(xlabel="step", ylabel="bits per byte", title="Validation BPB")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.show()
    print("Saved training_curves.png")
except ImportError:
    print("matplotlib not installed — skipping plot.")

# %% [markdown]
# ### 3e. Text generation demo
#
# Use the trained model to autoregressively generate text from a short prompt.
# Even a small model trained for 8000 steps will show meaningful learned structure.

# %%
model.eval()

PROMPT = "The history of artificial intelligence began"
GENERATE_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 40

prompt_ids = tokenizer.encode(PROMPT, prepend="<|bos|>")
print(f"Prompt ({len(prompt_ids)} tokens): {PROMPT!r}\n")
print("Generated text:")
print("-" * 60)
print(PROMPT, end="", flush=True)

generated = []
for token_id in model.generate(prompt_ids, max_tokens=GENERATE_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    generated.append(token_id)
    token_str = tokenizer.decode([token_id])
    print(token_str, end="", flush=True)

print("\n" + "-" * 60)
print(f"\nGenerated {len(generated)} tokens.")

# %% [markdown]
# ---
# ## Part 4 — Minimal GPT (vanilla baseline)
#
# The same experiment with all optimisations stripped out.
# Vanilla GPT-2 style: learned positional embeddings, standard SDPA, LayerNorm, GELU, AdamW.
# No RoPE, no QK-Norm, no sliding window, no value residual, no smear, no backout, no softcap.

# %%
class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MinimalBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.blocks  = nn.ModuleList([MinimalBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # Zero-init output projections so residual branches start as identity
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.mlp[-1].weight)

    def get_device(self):
        return self.tok_emb.weight.device

    def setup_optimizer(self, lr=3e-4, weight_decay=0.1):
        # Two AdamW groups: weight matrices (with decay) and everything else (no decay).
        # Embeddings, LayerNorm params, and 1-D tensors generally should not be decayed.
        decay_params  = [p for p in self.parameters() if p.ndim >= 2]
        nodecay_params = [p for p in self.parameters() if p.ndim < 2]
        param_groups = [
            dict(params=decay_params,   lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
            dict(params=nodecay_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        ]
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction)
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seq_len=512, seed=42):
        rng = torch.Generator(device=tokens.device).manual_seed(seed)
        ids = tokens
        for _ in range(max_tokens):
            ids_cond = ids[:, -seq_len:]
            logits = self.forward(ids_cond)[:, -1, :]
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            next_id = torch.multinomial(F.softmax(logits / temperature, dim=-1),
                                        num_samples=1, generator=rng)
            ids = torch.cat((ids, next_id), dim=1)
            yield next_id.item()

# %%
MIN_DEPTH  = 6
MIN_DIM    = 384
MIN_HEADS  = 6
MIN_ITERS  = 8000

mini = MinimalGPT(
    vocab_size=tokenizer.get_vocab_size(),
    n_embd=MIN_DIM,
    n_head=MIN_HEADS,
    n_layer=MIN_DEPTH,
    seq_len=SEQ_LEN,
).to(DEVICE)

mini_params = sum(p.numel() for p in mini.parameters())
print(f"MinimalGPT params: {mini_params:,}")

mini_optimizer = mini.setup_optimizer(lr=3e-4, weight_decay=0.1)

mini_train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, split="train", device=DEVICE
)

# %%
mini_train_losses = []
mini_val_bpbs     = []
mini_val_steps    = []

pbar = tqdm(range(1, MIN_ITERS + 1), desc="MinimalGPT", unit="step")
for step in pbar:
    lr_scale = get_lr_scale(step, MIN_ITERS)
    for group in mini_optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_scale

    mini_optimizer.zero_grad(set_to_none=True)
    x, y = next(mini_train_loader)
    loss = mini(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mini.parameters(), 1.0)
    mini_optimizer.step()
    mini_train_losses.append(loss.item())

    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scale:.3f}")

    if step % EVAL_EVERY == 0 or step == MIN_ITERS:
        mini.eval()
        val_loader = val_loader_factory()
        bpb = evaluate_bpb(mini, val_loader, EVAL_STEPS, token_bytes)
        mini_val_bpbs.append(bpb)
        mini_val_steps.append(step)
        tqdm.write(f"  Step {step:04d} | train_loss={loss.item():.4f} | val_bpb={bpb:.4f}")
        mini.train()

print("MinimalGPT training complete.")

# %%
# Compare the two models
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(train_losses,       alpha=0.4, label=f"Nanochat GPT (depth={DEPTH})")
    ax1.plot(mini_train_losses,  alpha=0.4, label=f"Minimal GPT (depth={MIN_DEPTH})")
    ax1.set(xlabel="step", ylabel="cross-entropy loss", title="Training Loss")
    ax1.legend()

    ax2.plot(val_steps,      val_bpbs,      "o-", label=f"Nanochat GPT (depth={DEPTH})")
    ax2.plot(mini_val_steps, mini_val_bpbs, "s-", label=f"Minimal GPT (depth={MIN_DEPTH})")
    ax2.set(xlabel="step", ylabel="bits per byte", title="Validation BPB")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("comparison_curves.png", dpi=120)
    plt.show()
except ImportError:
    print("matplotlib not installed — skipping plot.")

# %%
# Generation demo for MinimalGPT
mini.eval()
prompt_ids_tensor = torch.tensor(
    [tokenizer.encode(PROMPT, prepend="<|bos|>")], dtype=torch.long, device=DEVICE
)
print(f"MinimalGPT generation:\n{'-'*60}")
print(PROMPT, end="", flush=True)
for token_id in mini.generate(prompt_ids_tensor, max_tokens=200, temperature=TEMPERATURE,
                               top_k=TOP_K, seq_len=SEQ_LEN):
    print(tokenizer.decode([token_id]), end="", flush=True)
print(f"\n{'-'*60}")

# %% [markdown]
# ---
# ## Part 5 — LUTGPT (publish recipe)
#
# Tiny LUT-language-model from
# [`examples/lutgpt/`](../examples/lutgpt/), 84.94 M parameters.
# Every attention projection and every per-layer residual is a
# `TinyMultiHeadLut` table — there are no dense matmuls in the network outside
# the token embedding and the unembedder.
#
# **Dual residual streams**:
#  * **E-stream** (width `E = 96`) carries the attention working state;
#    each block reads it through a `MeanAbsNorm`, runs the LUT-driven
#    attention (`qk_lut`, `v_lut`), and writes back via `out_proj`.
#  * **D-stream** (width `D = 384`) is a pure accumulator. The token embedding
#    is renormed and fed through `emb_resid_lut` to seed the D-stream; every
#    block then adds a `residual_lut(MeanAbsNorm(x_lut))` contribution. The
#    final `ln_final(x_resid)` goes to the unembedder.
#
# ```
#   tokens -> tok_emb_E [V, E]
#             |
#             MeanAbsNorm(E) -> emb_resid_lut (NAP=5, tph=256) -> D   --.
#             |                                                         |
#             |   per layer x N_LAYERS:                                 |
#             |     MeanAbsNorm(E)                                      |
#             |     qk_lut       (NAP=4, tph=256, n_outputs=2*d_qk)     |
#             |     v_lut        (NAP=6, tph=256, n_outputs=d_v)        |
#             |     scaled-dot-product attention (RoPE on q,k)          |
#             |     out_proj     (NAP=6, tph=512, n_outputs=E)          |
#             |     residual_lut (NAP=5, tph=256, n_outputs=D) ---------+
#             |     x_lut += out_proj(attn)                             |
#             v                                                         v
#          E-stream                                                  D-stream
#                                                                    ln_final
#                                                                    unembedder -> logits
# ```
#
# **Two-phase training**: `forward_mode='hybrid_smooth'` for the first half,
# then a runtime flip to `forward_mode='hard'`. Backward is the *same* soft
# K-row surrogate in both phases; only the forward path and the weight-grad
# scatter change. See
# [`paper/tinymhl_hybrid_smooth.tex`](../paper/tinymhl_hybrid_smooth.tex)
# for the math.
#
# This notebook runs a scaled-down 4 000-step demo (2 000 + 2 000 phase A/B).
# The publish recipe in `examples/lutgpt/` uses 16 000 steps with a wider
# batch — switch to that for a real run.

# %%
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

# Architecture (mirrors examples/lutgpt/config.json)
LUT_E             = 96
LUT_D             = 384
LUT_H             = 6
LUT_D_QK          = 64
LUT_D_V           = 16
LUT_N_LAYERS      = 6
LUT_ROPE_BASE     = 10000.0

# Per-LUT (NAP, tables_per_head). NAP=k means each table has K = 2^k rows.
LUT_QK_NAP   = 4;  LUT_QK_TPH   = 256
LUT_V_NAP    = 6;  LUT_V_TPH    = 256
LUT_OUT_NAP  = 6;  LUT_OUT_TPH  = 512
LUT_RES_NAP  = 5;  LUT_RES_TPH  = 256
LUT_EMB_NAP  = 5;  LUT_EMB_TPH  = 256

# Common kwargs for every TinyMultiHeadLut in the model.
_LUT_KWARGS = dict(
    weight_dtype=torch.float32,                       # fp32 master weights
    use_bf16=True,                                    # bf16 autocast for compute
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    forward_mode="hybrid_smooth",                     # phase A; flipped to "hard" at the switch
    soft_score_temp=0.5,
    select_temp=0.5,
    learnable_temps=True,
    initial_weights_noise=0.001,
)

LUT_SEED = 42

def _make_lut(input_dim, n_heads, n_outputs, tables_per_head, n_anchor_pairs, seed_offset):
    return TinyMultiHeadLut(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs,
        tables_per_head=tables_per_head,
        random_seed=LUT_SEED + seed_offset,
        device=torch.device(DEVICE),
        **_LUT_KWARGS,
    )

def _make_qk(seed_offset):
    # n_outputs=2*d_qk: same LUT emits Q and K concatenated.
    return _make_lut(LUT_E, LUT_H, 2 * LUT_D_QK, LUT_QK_TPH, LUT_QK_NAP, seed_offset)

def _make_v(seed_offset):
    return _make_lut(LUT_E, LUT_H, LUT_D_V, LUT_V_TPH, LUT_V_NAP, 200 + seed_offset)

def _make_out(seed_offset):
    # Input dim = H * d_v (concat'd value-head outputs); single LUT-head -> E.
    return _make_lut(LUT_H * LUT_D_V, 1, LUT_E, LUT_OUT_TPH, LUT_OUT_NAP, 400 + seed_offset)

def _make_residual_lut(seed_offset):
    # Per-layer residual LUT into the D-stream.
    return _make_lut(LUT_E, 1, LUT_D, LUT_RES_TPH, LUT_RES_NAP, 600 + seed_offset)

def _make_emb_resid_lut(seed_offset):
    # Bare-embedding seed contribution into the D-stream (placed once,
    # before the layer stack).
    return _make_lut(LUT_E, 1, LUT_D, LUT_EMB_TPH, LUT_EMB_NAP, 800 + seed_offset)


# %%
# --- MeanAbsNorm: x / (mean(|x|) + eps), no centering, no affine -------------
class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


# --- RoPE on (q, k) -----------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)


def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


# --- Lion optimizer (sign-momentum) for LUT params ----------------------------
class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum. Single momentum buffer, sign-based update."""
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp["lr"], grp["betas"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                update = (m * b1 + g * (1.0 - b1)).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1.0 - b2)


# %%
# --- LUTBlock: attention + D-stream contribution (dual-stream) ----------------
class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_lut       = _make_qk(layer_idx)
        self.v_lut        = _make_v(layer_idx)
        self.out_proj     = _make_out(layer_idx)
        self.residual_lut = _make_residual_lut(layer_idx)
        self.q_norm  = nn.LayerNorm(LUT_D_QK)
        self.k_norm  = nn.LayerNorm(LUT_D_QK)
        self.ln_pre  = MeanAbsNorm(LUT_E)
        self.ln_resid = MeanAbsNorm(LUT_E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, LUT_E)
        x_pre  = self.ln_pre(x_flat)

        qk_out = self.qk_lut(x_pre)                                  # [B*T, H, 2*d_qk]
        q_vec = self.q_norm(qk_out[..., :LUT_D_QK])
        k_vec = self.k_norm(qk_out[..., LUT_D_QK:2 * LUT_D_QK])
        q = q_vec.reshape(B, T, LUT_H, LUT_D_QK).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, LUT_H, LUT_D_QK).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_vec = self.v_lut(x_pre)                                    # [B*T, H, d_v]
        v = v_vec.reshape(B, T, LUT_H, LUT_D_V).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, LUT_H * LUT_D_V)
        out_e  = self.out_proj(out_in).squeeze(1)                    # [B*T, E]

        x_lut_next_flat = x_flat + out_e                             # E-stream update

        # D-stream contribution from this layer.
        r_in  = self.ln_resid(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, LUT_D)

        return x_lut_next_flat.reshape(B, T, LUT_E), r_out


# --- LUTGPT: dual-stream model ------------------------------------------------
class LUTGPT(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        torch.manual_seed(LUT_SEED)
        self.tok_emb_E      = nn.Embedding(vocab_size, LUT_E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.emb_resid_lut  = _make_emb_resid_lut(0)
        self.ln_emb_resid   = MeanAbsNorm(LUT_E)
        self.unembedder     = nn.Linear(LUT_D, vocab_size, bias=False)
        self.rope = RotaryEmbedding(LUT_D_QK, max_seq_len=seq_len,
                                    base=LUT_ROPE_BASE, device=torch.device(DEVICE))
        self.layers   = nn.ModuleList([LUTBlock(i) for i in range(LUT_N_LAYERS)])
        self.ln_final = nn.LayerNorm(LUT_D)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def setup_optimizer(self, adam_lr=3e-4, lut_lr=2e-4,
                        lut_betas=(0.9, 0.95), weight_decay=0.1):
        """Lion for the 3-D LUT weight tensors, AdamW for everything else."""
        lut_params, tok_emb_params, decay_params, nodecay_params = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 3:                        lut_params.append(p)
            elif name.startswith("tok_emb_E."):    tok_emb_params.append(p)
            elif p.ndim == 2:                      decay_params.append(p)
            else:                                  nodecay_params.append(p)
        adam_groups = [
            dict(params=decay_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=weight_decay),
            dict(params=tok_emb_params + nodecay_params, lr=adam_lr,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        ]
        adam_opt = torch.optim.AdamW(adam_groups)
        lion_opt = Lion([dict(params=lut_params, lr=lut_lr, weight_decay=0.0)],
                        lr=lut_lr, betas=lut_betas)
        opts = [adam_opt, lion_opt]
        for opt in opts:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return opts

    def flip_to_hard_forward(self):
        """Phase-B switch: forward_mode 'hybrid_smooth' -> 'hard' across all LUTs."""
        n = 0
        for mod in self.modules():
            if isinstance(mod, TinyMultiHeadLut):
                mod.forward_mode = "hard"
                n += 1
        return n

    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.shape
        x_lut = self.tok_emb_E(idx)                                  # (B, T, E)
        # Seed the D-stream from the bare embedding.
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B * T, LUT_E))
        x_resid   = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, LUT_D)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid  = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits  = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seq_len=512, seed=42):
        rng = torch.Generator(device=tokens.device).manual_seed(seed)
        ids = tokens
        for _ in range(max_tokens):
            ids_cond = ids[:, -seq_len:]
            logits = self.forward(ids_cond)[:, -1, :]
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            next_id = torch.multinomial(
                F.softmax(logits / temperature, dim=-1),
                num_samples=1, generator=rng,
            )
            ids = torch.cat((ids, next_id), dim=1)
            yield next_id.item()


# %% [markdown]
# ### 5a. Build the model

# %%
LUT_ITERS         = 4000
LUT_PHASE_SWITCH  = 2000     # bs_switch_step == hard_switch_step
LUT_EVAL_EVERY    = 200
LUT_BS_A          = 4        # phase A device batch (hybrid_smooth)
LUT_BS_B          = 8        # phase B device batch (hard)

lutgpt = LUTGPT(vocab_size=tokenizer.get_vocab_size(), seq_len=SEQ_LEN).to(DEVICE)
lut_param_count = sum(p.numel() for p in lutgpt.parameters())
print(f"LUTGPT params: {lut_param_count:,}")
print(f"Two-phase schedule: hybrid_smooth bs={LUT_BS_A} for {LUT_PHASE_SWITCH} steps, "
      f"then hard bs={LUT_BS_B} for {LUT_ITERS - LUT_PHASE_SWITCH} steps")

lut_optimizers = lutgpt.setup_optimizer()
lut_train_loader_a = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, LUT_BS_A, SEQ_LEN, split="train", device=DEVICE,
)
lut_train_loader_b = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, LUT_BS_B, SEQ_LEN, split="train", device=DEVICE,
)

# %% [markdown]
# ### 5b. Two-phase training loop
#
# Phase A: `forward_mode='hybrid_smooth'`, smaller batch (`bs=4`).
# At `LUT_PHASE_SWITCH`, flip every LUT to `forward_mode='hard'` and swap to
# the `bs=8` data loader. The LR schedule is continuous across both phases.

# %%
lut_train_losses = []
lut_val_bpbs     = []
lut_val_steps    = []

train_loader_lut = lut_train_loader_a
pbar = tqdm(range(1, LUT_ITERS + 1), desc="LUTGPT", unit="step")
for step in pbar:
    if step == LUT_PHASE_SWITCH + 1:
        n_flipped = lutgpt.flip_to_hard_forward()
        train_loader_lut = lut_train_loader_b
        tqdm.write(f"  [SWITCH] step {step}: forward_mode -> 'hard' on {n_flipped} LUTs, bs -> {LUT_BS_B}")

    lr_scale = get_lr_scale(step, LUT_ITERS)
    for opt in lut_optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale

    for opt in lut_optimizers:
        opt.zero_grad(set_to_none=True)
    x, y = next(train_loader_lut)
    loss = lutgpt(x, y)
    loss.backward()
    for opt in lut_optimizers:
        opt.step()
    lut_train_losses.append(loss.item())

    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scale:.3f}")

    if step % LUT_EVAL_EVERY == 0 or step == LUT_ITERS:
        lutgpt.eval()
        val_loader = val_loader_factory()
        bpb = evaluate_bpb(lutgpt, val_loader, EVAL_STEPS, token_bytes)
        lut_val_bpbs.append(bpb)
        lut_val_steps.append(step)
        tqdm.write(f"  Step {step:04d} | train_loss={loss.item():.4f} | val_bpb={bpb:.4f}")
        lutgpt.train()

print("LUTGPT training complete.")

# %%
# Comparison plot: LUTGPT vs MinimalGPT vs Nanochat GPT
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_steps,      val_bpbs,      "o-", label=f"Nanochat GPT (depth={DEPTH})")
    ax.plot(mini_val_steps, mini_val_bpbs, "s-", label=f"Minimal GPT (depth={MIN_DEPTH})")
    ax.plot(lut_val_steps,  lut_val_bpbs,  "^-", label=f"LUTGPT (E={LUT_E}, D={LUT_D}, L={LUT_N_LAYERS})")
    ax.axvline(LUT_PHASE_SWITCH, color="grey", linestyle="--", alpha=0.5,
               label="LUTGPT phase switch")
    ax.set(xlabel="step", ylabel="bits per byte", title="Validation BPB")
    ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig("comparison_curves_lutgpt.png", dpi=120)
    plt.show()
except ImportError:
    print("matplotlib not installed — skipping plot.")

# %%
# Generation demo for LUTGPT
lutgpt.eval()
prompt_ids_tensor = torch.tensor(
    [tokenizer.encode(PROMPT, prepend="<|bos|>")], dtype=torch.long, device=DEVICE,
)
print(f"LUTGPT generation:\n{'-'*60}")
print(PROMPT, end="", flush=True)
for token_id in lutgpt.generate(prompt_ids_tensor, max_tokens=200,
                                 temperature=TEMPERATURE, top_k=TOP_K, seq_len=SEQ_LEN):
    print(tokenizer.decode([token_id]), end="", flush=True)
print(f"\n{'-'*60}")
