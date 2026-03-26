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
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# Make sure we run from the nanochat project root
PROJECT_ROOT = os.path.dirname(os.path.abspath("walkthrough.py"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nanochat.common import get_base_dir, COMPUTE_DTYPE
from nanochat.dataset import list_parquet_files, DATA_DIR, parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig
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
# We train a **tiny** model (depth=4) so the notebook completes quickly.
# The architecture is identical to the full model — just smaller.

# %% [markdown]
# ### 3a. Build the model

# %%
DEPTH = 4
SEQ_LEN = 512
DEVICE_BATCH_SIZE = 8   # micro-batch size (sequences per step)
TOTAL_BATCH_SIZE = 4096  # tokens per gradient step (determines grad accumulation)
NUM_ITERATIONS = 200
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
# Even a tiny model trained for only 200 steps will show some learned structure.

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
