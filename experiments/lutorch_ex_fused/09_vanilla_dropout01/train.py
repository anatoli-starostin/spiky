#!/usr/bin/env python
"""09/10 vanilla + dropout — the dense-transformer vanilla baseline (exp_n_0216 arch) WITH standard
dropout, on the 170-shard corpus. Shared trainer for 09_vanilla_dropout01 (dropout 0.1) and
10_vanilla_dropout02 (dropout 0.2); the ONLY difference from 00_vanilla is one nn.Dropout.

Self-contained single-GPU trainer. The 6-block MinimalGPT is defined INLINE in this file
(RoPE + SDPA causal attention, prenorm residual blocks, untied unembedder; n_embd 384,
n_head 6, seq_len 512, vocab 32768) — no model factory, no hidden abstraction. Every block's
FFN is a plain dense 4x GELU MLP (Linear 384->1536 + GELU + Linear 1536->384, bias=False) with a
standard nn.Dropout(cfg["dropout"]) INSIDE the MLP, right after the GELU (attention untouched — the
closest analogue to the fused arms' table dropout, which also acts inside the FFN substitute). NO LUT,
NO cell-TV, NO table dropout.

Derived from the verified 6ebb49ec fused reference trainer with the LUT machinery dropped; the whole
training loop, optimizer grouping, LR schedule, fixed_eval, wandb and ckpt are byte-identical to 01-08.
geometry/schedule come from `config.json`. (Historical, now inapplicable, note:) cell-TV λ=10 is a
SEPARATE backward (as in the original abl47 trainer), table dropout 0.2, faithful OLD-matching
init (the sole cause of the earlier bpb gap), AdamW with a no-weight-decay group for the LUT
tables + all ndim<2 params, cosine LR with a 0.1·lr floor, fixed_eval (bs48×100, skip12,
abl47-comparable bits/byte), single-run checkpoint/resume. Corpus: nanochat ClimbMix.

Launch (from this folder, on a CUDA box with the lutorch_ex extension built):
    python train.py                 # reads ./config.json, writes artifacts here
    SMOKE_STEPS=40 python train.py   # short sanity (overrides n_steps + eval cadence)

NOTE (precision): these runs are plain FP32 throughout (PyTorch default; no mixed precision, no
explicit dtype knob); this matches abl47's actual numerics. The LUT stays an fp32 island. NOTE (wandb): ONLINE-or-die — wandb.init runs FIRST (before data/model/GPU) with
project "Spiky", group "lutorch_ex_fused"; no offline fallback, so a bad launch (no network /
not logged in — e.g. inside the sbox cage) hard-fails in seconds. Launch as a BARE `python
train.py` on a networked, wandb-logged-in box. (The old wandb_tracking helper was not vendored —
it pulls spiky.util.wandb_integration + metric_glossary.)
"""
import os, sys, json, math, time, csv

sys.path.insert(0, os.environ.get("NANOCHAT_ROOT", "/home/astarostin/projects/nanochat"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))  # vendored fixed_eval

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import get_base_dir
from fixed_eval import evaluate_bpb_fixed, eval_config
# VANILLA baseline: no lutorch_ex cartridge import — the FFN is a plain dense MLP (defined inline below).

# --------------------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "config.json")))
OUT_DIR = os.environ.get("OUT_DIR", HERE)
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda"
torch.manual_seed(cfg["random_seed"])

SEQ, DEVICE_BS, TOTAL_BS = cfg["seq_len"], cfg["device_batch_size"], cfg["total_batch_size"]
N_STEPS = int(os.environ.get("SMOKE_STEPS", cfg["n_steps"]))
EVAL_EVERY = (int(os.environ["SMOKE_EVAL_EVERY"]) if "SMOKE_EVAL_EVERY" in os.environ
              else (max(1, N_STEPS // 4) if "SMOKE_STEPS" in os.environ else cfg["eval_every"]))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", cfg.get("ckpt_every", 2000)))
LR, WD, WARM = cfg["lr"], cfg["weight_decay"], cfg["lr_warmup_fraction"]
# VANILLA: no LUT knobs (no cell-TV, no table dropout, no cartridge seed/noise).

# ------------------------------------------------- wandb (ONLINE-or-die; init FIRST, before any data/model/GPU work)
RUN_TAG = os.environ.get("RUN_TAG") or time.strftime("%m%d_%H%M%S")   # fresh unique name per launch (no resume)
run_name = f"{cfg['exp_name']}_{RUN_TAG}"
import wandb
try:
    wb = wandb.init(project="Spiky", group="lutorch_ex_fused", name=run_name, id=run_name,
                    config=cfg, tags=["lutorch_ex", "vanilla", cfg.get("ffn_type", "dense"), "dropout"])
except Exception as e:                                  # ONLINE or die -- no offline fallback
    raise RuntimeError(
        f"wandb.init failed ({type(e).__name__}: {e}). This trainer is ONLINE-or-die: no offline "
        "fallback, and init runs FIRST so a bad launch dies in seconds (before data/model/GPU). "
        "Likely causes: (1) no network -- e.g. launched inside the sbox cage; run a BARE "
        "`python train.py` on a networked box instead. (2) not logged in to wandb (`wandb login`). "
        "Fix and relaunch."
    ) from e

# --------------------------------------------------------------------------- data
tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), "tokenizer"))
VOCAB = tok.get_vocab_size()
assert VOCAB == cfg["tokenizer_vocab_size"], (VOCAB, cfg["tokenizer_vocab_size"])
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DEVICE_BS, SEQ, split="train", device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)
EVAL = eval_config(cfg)   # fixed protocol: bs48 x 100 steps, skip 12 rows

# ===========================================================================
# MODEL (inline, explicit): 6-block MinimalGPT, every FFN a fused LUT cartridge
# ===========================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MinimalBlock(nn.Module):
    def __init__(self, cfg, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        # VANILLA+DROPOUT dense FFN: the exp_n_0216 baseline FFN (4x GELU MLP, bias=False) with a
        # standard nn.Dropout INSIDE the MLP, right after the GELU (on the 1536-dim hidden units,
        # before the down-projection) — the dense analogue of the fused arms' table dropout. Rate
        # from cfg["dropout"] (0.0 => behaviour byte-identical to 00_vanilla). Attention untouched.
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                                 nn.Dropout(cfg.get("dropout", 0.0)),
                                 nn.Linear(4 * n_embd, n_embd, bias=False))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class MinimalGPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        E, H, L, S = cfg["n_embd"], cfg["n_head"], cfg["depth"], cfg["seq_len"]
        self.tok_emb = nn.Embedding(vocab_size, E)
        self.rope = RotaryEmbedding(E // H, max_seq_len=S)
        self.blocks = nn.ModuleList([MinimalBlock(cfg, E, H, i) for i in range(L)])
        self.ln_f = nn.LayerNorm(E)
        self.head = nn.Linear(E, vocab_size, bias=False)   # untied unembedder
        self.apply(self._init_weights)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)              # zero attn out-proj (as in abl47 backbone)
            nn.init.zeros_(b.mlp[-1].weight)                # zero the MLP's last Linear -> FFN starts ~0 (exp_n_0216)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, loss_reduction="mean"):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.rope.cos, self.rope.sin)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits

    def get_device(self):                 # nanochat-GPT interface used by fixed_eval.evaluate_bpb_fixed
        return next(self.parameters()).device


model = MinimalGPT(cfg, VOCAB).to(DEVICE)

# VANILLA: no cartridge faithful-init needed. The dense MLP + backbone use the standard init in
# MinimalGPT.__init__ (normal_(0, 0.02) on every Linear/Embedding; attn out-proj and the MLP's last
# Linear zeroed, so each block starts as ~identity — the exp_n_0216 vanilla scheme).
total_params = sum(p.numel() for p in model.parameters())
print(f"ffn_type=dense dropout={cfg.get('dropout', 0.0)} blocks={len(model.blocks)} params={total_params:,} "
      f"mlp_in_std(b0)={model.blocks[0].mlp[0].weight.std().item():.4f} "
      f"mlp_last_zero(b0)={bool((model.blocks[0].mlp[-1].weight == 0).all().item())} "
      f"attn_proj_zero(b0)={bool((model.blocks[0].attn.proj.weight == 0).all().item())} "
      f"n_steps={N_STEPS}")

# --------------------------------------------------------------------------- optimizer + schedule
decay, nodecay = [], []
for p in model.parameters():
    if p.requires_grad:
        (nodecay if p.ndim < 2 else decay).append(p)   # no-WD group = all ndim<2 (LayerNorms); no LUT tables
opt = torch.optim.AdamW([dict(params=decay, weight_decay=WD), dict(params=nodecay, weight_decay=0.0)],
                        lr=LR, betas=(0.9, 0.95), eps=1e-8)
grad_accum = max(1, TOTAL_BS // (DEVICE_BS * SEQ))


def lr_scale(step):
    w = int(WARM * N_STEPS)
    if step < w:
        return step / max(w, 1)
    p = (step - w) / max(N_STEPS - w, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))   # cosine to a 0.1*lr floor


# --------------------------------------------------------------------------- single-run checkpoint / resume
CKPT = os.path.join(OUT_DIR, "ckpt.pt")
start_step = 1
if os.path.exists(CKPT):
    st = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start_step = int(st["step"]) + 1
    print(f"[resume] ckpt.pt at step {st['step']} -> resuming at {start_step} "
          f"(dataloader restarts from token 0; resume is approximate)")

# --------------------------------------------------------------------------- csv (wandb initialised earlier, ONLINE-or-die)
csv_f = open(os.path.join(OUT_DIR, "metrics.csv"), "a", newline="")
csv_w = csv.writer(csv_f)
if start_step == 1:
    csv_w.writerow(["step", "train_loss", "val_bpb", "step_ms"]); csv_f.flush()

# --------------------------------------------------------------------------- train loop
model.train(); ema = None; best = float("inf"); t_run = time.time()
for step in range(start_step, N_STEPS + 1):
    s = lr_scale(step)
    for g in opt.param_groups:
        g["lr"] = LR * s
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    acc = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum).backward()
        acc += loss.item() / grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # VANILLA: no cell-TV penalty / separate backward
    opt.step()
    torch.cuda.synchronize(); step_ms = (time.perf_counter() - t0) * 1e3
    ema = acc if ema is None else 0.99 * ema + 0.01 * acc
    wb.log({"step": step, "train_loss": acc, "train_ema": ema,
           "lr": LR * s, "step_ms": step_ms}, step=step)
    if step % 20 == 0 or step <= 5:
        print(f"step {step:6d} loss={acc:.4f} ema={ema:.4f} lr={LR*s:.2e} ms={step_ms:.0f}")
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb_fixed(model, tok, token_bytes, SEQ, DEVICE, **EVAL)
        model.train()
        best = min(best, bpb)
        print(f"[VAL] step {step}: bpb={bpb:.4f} (best {best:.4f})")
        wb.log({"step": step, "val_bpb": bpb}, step=step)
        csv_w.writerow([step, f"{acc:.6f}", f"{bpb:.6f}", f"{step_ms:.1f}"]); csv_f.flush()
    if step % CKPT_EVERY == 0 or step == N_STEPS:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT)

csv_f.close()
peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
summary = {"exp_name": cfg["exp_name"], "ffn_type": cfg.get("ffn_type", "dense"), "best_val_bpb": best,
           "total_params": total_params, "peak_gb": peak_gb,
           "hours": round((time.time() - t_run) / 3600, 4), "n_steps": N_STEPS, "run_name": run_name}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
print("=== DONE ===", json.dumps(summary))
wb.summary.update(summary); wb.finish()
