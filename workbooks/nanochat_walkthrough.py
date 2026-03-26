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

    def get_device(self):
        return self.tok_emb.weight.device

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

mini_optimizer = torch.optim.AdamW(mini.parameters(), lr=3e-4, weight_decay=0.1)

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
        group["lr"] = 3e-4 * lr_scale

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
