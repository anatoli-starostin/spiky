"""Load exp738 checkpoint, cast every parameter EXCEPT head.weight to bf16,
then evaluate val_bpb. Tests the deployment scenario where the backbone is
quantized to bf16 but the unembedder stays fp32.
"""
import sys, os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get("NANOCHAT_ROOT", "/home/starost/nanochat")
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

DEVICE = torch.device("cuda:0")
EXP_DIR = "/home/starost/spiky/nanochat_exps/exp738_vanilla_bf16_bs48_4k"
with open(os.path.join(EXP_DIR, "config.json")) as f:
    cfg = json.load(f)

DEPTH = cfg["depth"]
N_EMBD = cfg["n_embd"]
N_HEAD = cfg["n_head"]
SEQ_LEN = cfg["seq_len"]
DEVICE_BS = cfg["device_batch_size"]
EVAL_STEPS = cfg["eval_steps"]

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, "tokenizer"))
VOCAB_SIZE = tokenizer.get_vocab_size()
token_bytes = get_token_bytes(device=DEVICE)


# Identical model classes as exp738/train.py ---------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1); return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # Cast cos/sin to q's dtype so RoPE doesn't upcast to fp32 when q is bf16.
    cos = cos.to(q.dtype)[None, None, :, :]
    sin = sin.to(q.dtype)[None, None, :, :]
    return (q*cos + _rotate_half(q)*sin, k*cos + _rotate_half(k)*sin)


class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MinimalBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd, bias=False),
        )
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        head_dim = n_embd // n_head
        self.rope = RotaryEmbedding(head_dim, seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    def get_device(self): return self.tok_emb.weight.device
    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.size()
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        h = self.ln_f(x)
        # Cast to head weight's dtype so we can mix bf16 backbone with fp32 head
        h = h.to(self.head.weight.dtype)
        logits = self.head(h)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1)
        return logits


model = MinimalGPT(VOCAB_SIZE, N_EMBD, N_HEAD, DEPTH, SEQ_LEN).to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, "checkpoint.pt"), weights_only=False, map_location="cpu")
model.load_state_dict(ckpt)
print(f"Loaded exp738 checkpoint (n_params={sum(p.numel() for p in model.parameters()):,})")


def make_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BS, SEQ_LEN, split="val", device=DEVICE)


# Baseline: full fp32 ----------------------------------------------------------
model.eval()
with torch.no_grad():
    val_loader = make_val_loader()
    bpb_fp32 = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
print(f"\n[fp32 storage everywhere]               val_bpb = {bpb_fp32:.4f}")


# Cast everything EXCEPT head.weight to bf16 ----------------------------------
n_cast = 0
n_skipped = 0
for name, p in model.named_parameters():
    if name == "head.weight":
        n_skipped += 1
        continue
    p.data = p.data.to(torch.bfloat16)
    n_cast += 1
print(f"\nCast {n_cast} params to bf16, kept {n_skipped} param ('head.weight') as fp32")


# Verify (sample) -------------------------------------------------------------
for n in ["tok_emb.weight", "blocks.0.attn.qkv.weight", "blocks.0.mlp.0.weight",
         "blocks.0.mlp.2.weight", "blocks.0.ln1.weight", "ln_f.weight", "head.weight"]:
    p = dict(model.named_parameters())[n]
    print(f"  {n}: dtype={p.dtype}")


# Eval with bf16 backbone + fp32 head -----------------------------------------
with torch.no_grad():
    val_loader = make_val_loader()
    bpb_mixed = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
print(f"\n[bf16 backbone + fp32 head]            val_bpb = {bpb_mixed:.4f}")
print(f"Delta vs fp32 everywhere: {(bpb_mixed - bpb_fp32)*1000:+.2f} mb")


# Also try ALL bf16 (head too) for reference ----------------------------------
model.head.weight.data = model.head.weight.data.to(torch.bfloat16)
print(f"\nHead now cast to bf16 too: dtype={model.head.weight.dtype}")
with torch.no_grad():
    val_loader = make_val_loader()
    bpb_all_bf16 = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
print(f"[bf16 EVERYTHING]                       val_bpb = {bpb_all_bf16:.4f}")
print(f"Delta vs fp32 everywhere: {(bpb_all_bf16 - bpb_fp32)*1000:+.2f} mb")
