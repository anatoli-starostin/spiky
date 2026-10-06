"""Quantised Confidence n=1 at the lutorch_ex sweep's champion topology, 48k steps -- one self-contained trainer.

The model is fully defined in this file: a 6-block pre-LayerNorm GPT with rotary attention (the sweep's MinimalGPT),
whose feed-forward layer in every block is lutorch_ex's ProjectionMHL around a QuantisedConfidenceLUT
(quant_mode="p2_int8", read_top_n=1). nanochat supplies only the tokenizer, the token-byte table, its data directory and
the ClimbMix training/validation stream. Everything else -- hyperparameters, optimiser, schedule, regularisers,
evaluation, checkpointing -- is written out below as literals.

Matches the seven-run lutorch_ex sweep (experiments/lutorch_ex/lutorch_ex_abl47_*_1004_*) in every respect except the
cartridge's read_top_n (the sweep's Quantised arm used 2):
  * cell TV regularisation at a CONSTANT lambda = 10 every step, and table dropout 0.2 (what every sweep run used);
  * the whole model in fp32 (the sweep's trainers never cast to bf16, whatever their configs' compute_dtype said);
  * the fixed validation window: rows [12, 4800) of the val stream read from token 0 at batch 48 x 100 steps.

Outputs, written into this folder: metrics.csv, summary.json, ckpt.pt (gitignored), train.log (from the launcher).

Checkpoint + resume: ckpt.pt (model, optimiser, step, EMA loss, best val_bpb) is written atomically every CKPT_EVERY
steps and at the end; re-running this script resumes from it, and metrics.csv is trimmed back to the checkpoint's step
so no row is duplicated. Resume does NOT restore the data-loader position or the RNG state: after a resume the
training stream restarts from the start of the shards and table-dropout masks differ, so a resumed run is continuous
in its weights and optimiser but not bit-identical to an uninterrupted one.

Run (from anywhere; W&B is optional and never fatal -- off unless WANDB_BASE_URL is set):
    python -u train.py >> train.log 2>&1
Environment: NANOCHAT_ROOT (default ~/projects/nanochat). For a short smoke test: SMOKE_STEPS, SMOKE_EVAL_EVERY,
CKPT_EVERY. The CUDA p2_int8 op is JIT-built on first use (needs a CUDA toolkit: CUDA_HOME or /usr/local/cuda).
"""
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))           # this checkout's root
sys.path.insert(0, os.path.join(REPO, "src"))                            # lutorch_ex from THIS checkout
NANOCHAT_ROOT = os.environ.get("NANOCHAT_ROOT", os.path.expanduser("~/projects/nanochat"))
sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir                                  # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes         # noqa: E402
from spiky.lutorch_ex import LUTSpec, ProjectionMHL, QuantisedConfidenceLUT, cell_tv_penalty  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Hyperparameters
EXP_NAME = os.path.basename(HERE)
SEED = 1

# model
DEPTH, N_EMBD, N_HEAD, SEQ_LEN, VOCAB = 6, 384, 6, 512, 32768        # untied unembedding

# LUT feed-forward layer (the sweep's champion topology): 8 heads x 64 tables x 2^8 cells, 48-wide per head
LUT_HEADS, LUT_TPH, LUT_NAP, LUT_D_IN, LUT_D_OUT = 8, 64, 8, 48, 48
LUT_TABLE_INIT = 1e-3              # tables ~ Uniform[-1e-3, 1e-3]
LUT_SEED_BASE = 1000               # cartridge of block i is seeded LUT_SEED_BASE + i
QUANT_MODE, READ_TOP_N = "p2_int8", 1
BETA_INIT, GAMMA_INIT = 2.0, 1.0   # confidence score s = (sum m) * exp(gamma * sum log sigmoid(beta m)); both learned
READ_TAU_INIT, READ_TAU_LEARNABLE = 0.5, True   # the n=2 champion's values; tau only enters the two-cell read
TABLE_DROPOUT = 0.2
TV_LAMBDA = 10.0                   # constant, every step

# optimisation
N_STEPS = 48_000
DEVICE_BATCH = 12                  # rows per micro-batch
GRAD_ACCUM = 4                     # -> 48 rows x 512 = 24,576 tokens per optimiser step
LR, BETAS, EPS, WEIGHT_DECAY = 3e-4, (0.9, 0.95), 1e-8, 0.1   # AdamW; LUT tables and 1-D params get no decay
WARMUP_FRAC, LR_FLOOR = 0.1, 0.1  # linear warmup over 10%, then cosine down to 0.1 x LR
GRAD_CLIP = 1.0

# evaluation: rows [EVAL_SKIP_ROWS, EVAL_BATCH * EVAL_STEPS) of the val stream from token 0 = 4,788 rows x 512 tokens
EVAL_EVERY = 500
EVAL_BATCH, EVAL_STEPS, EVAL_SKIP_ROWS = 48, 100, 12

# checkpoints (~165 ms/step on an RTX 5090 -> ~3 min of training per checkpoint, well under 15 min)
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 1000))

if "SMOKE_STEPS" in os.environ:    # a short sanity run; the schedule then spans SMOKE_STEPS
    N_STEPS = int(os.environ["SMOKE_STEPS"])
    EVAL_EVERY = int(os.environ.get("SMOKE_EVAL_EVERY", EVAL_EVERY))

DEVICE = "cuda"
CKPT = os.path.join(HERE, "ckpt.pt")
METRICS = os.path.join(HERE, "metrics.csv")


# ---------------------------------------------------------------------------------------------------------------------
# Model
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in self.qkv(x).split(C, dim=2))
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


def make_ffn(layer):
    spec = LUTSpec(h_in=LUT_HEADS, h_out=LUT_HEADS, tph=LUT_TPH, nap=LUT_NAP, d_in=LUT_D_IN, d_out=LUT_D_OUT)
    cart = QuantisedConfidenceLUT(spec, quant_mode=QUANT_MODE, read_top_n=READ_TOP_N, seed=LUT_SEED_BASE + layer,
                                  weight_init_std=LUT_TABLE_INIT, table_dropout_rate=TABLE_DROPOUT,
                                  beta_init=BETA_INIT, gamma_init=GAMMA_INIT,
                                  read_tau_init=READ_TAU_INIT, read_tau_learnable=READ_TAU_LEARNABLE)
    # ProjectionMHL defaults: compress ~ N(0, 0.02), decompress zero-initialised, so each block starts as identity.
    return ProjectionMHL(cart, d_model=N_EMBD)


class Block(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = Attention(N_EMBD, N_HEAD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.ffn = make_ffn(layer)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        B, T, C = h.shape
        return x + self.ffn(h.reshape(B * T, C)).reshape(B, T, C)


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.rope = RotaryEmbedding(N_EMBD // N_HEAD, SEQ_LEN)
        self.blocks = nn.ModuleList([Block(i) for i in range(DEPTH)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB, bias=False)
        # Backbone init: N(0, 0.02) for every Linear / Embedding outside the LUT layers, and each attention output
        # projection zeroed. The LUT layers keep their own init (above).
        ffn_modules = {id(m) for b in self.blocks for m in b.ffn.modules()}
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and id(m) not in ffn_modules:
                nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)

    def forward(self, idx, targets=None, loss_reduction="mean"):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=loss_reduction,
                               ignore_index=-1)


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation: bits per byte over the fixed validation window
@torch.no_grad()
def evaluate_bpb(model, tokenizer, token_bytes):
    """Bits per byte over rows [EVAL_SKIP_ROWS, EVAL_BATCH * EVAL_STEPS) of the val stream, read from token 0.

    Identical to the sweep's fixed eval: the batches are drained (cloned) before scoring, because the loader reuses
    one device buffer; target tokens with no bytes (special tokens) and ignore_index -1 targets do not count; the
    first EVAL_SKIP_ROWS rows are dropped (they are anomalously easy)."""
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, EVAL_BATCH, SEQ_LEN, split="val", device=DEVICE)
    batches = [tuple(t.clone() for t in next(loader)) for _ in range(EVAL_STEPS)]
    total_nats = torch.zeros((), dtype=torch.float32, device=DEVICE)
    total_bytes = torch.zeros((), dtype=torch.int64, device=DEVICE)
    for step, (x, y) in enumerate(batches):
        B, T = x.shape
        loss = model(x, y, loss_reduction="none").view(B, T)
        y_safe = torch.where(y >= 0, y, torch.zeros_like(y))
        nbytes = torch.where(y >= 0, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype))
        counts = nbytes > 0
        row0 = step * EVAL_BATCH
        if row0 < EVAL_SKIP_ROWS:
            counts[:min(EVAL_BATCH, EVAL_SKIP_ROWS - row0)] = False
        total_nats += (loss * counts).sum()
        total_bytes += torch.where(counts, nbytes, torch.zeros_like(nbytes)).sum()
    return total_nats.item() / (math.log(2) * total_bytes.item())


def lr_scale(step):
    warmup = int(WARMUP_FRAC * N_STEPS)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(N_STEPS - warmup, 1)
    return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------------------------------------------------
# W&B (optional, never fatal): project Spiky, group lutorch_ex, run name/id = EXP_NAME (resumes the same run)
def start_tracker(model):
    try:
        from spiky.util.wandb_integration.glossary import DictGlossary
        from spiky.util.wandb_integration.tracker import Tracker
    except Exception as e:                       # tracking is optional
        print(f"[wandb] off: {type(e).__name__}: {e}")
        return None
    glossary = DictGlossary({
        "train/loss": dict(unit="nats/tok", section="train", desc="Cross-entropy of the optimiser step (mean over its "
                           f"{GRAD_ACCUM} micro-batches)."),
        "train/loss_ema": dict(unit="nats/tok", section="train", desc="Exponential moving average (0.99) of train/loss."),
        "train/lr": dict(unit="lr", section="train", desc="Learning rate applied at this step."),
        "time/sec_per_step": dict(unit="s/step", section="train", desc="Wall seconds per optimiser step."),
        "val_bpb": dict(unit="bits/byte", section="eval", desc=f"Bits per byte on val rows [{EVAL_SKIP_ROWS}, "
                        f"{EVAL_BATCH * EVAL_STEPS}) read from token 0."),
        "train_loss": dict(unit="nats/tok", section="eval", desc="train/loss_ema at this eval."),
        "lut_tv": dict(unit="L2^2", section="eval", desc="Cell TV penalty (mean over the 6 LUT layers) at this step; "
                       f"the loss adds {TV_LAMBDA:g} x this."),
        "best_val_bpb": dict(unit="bits/byte", section="summary", desc="Lowest val_bpb of the run."),
        "total_params": dict(unit="params", section="summary", desc="Parameter count of the model."),
        "peak_gb": dict(unit="GB", section="summary", desc="Peak CUDA memory allocated."),
        "hours": dict(unit="h", section="summary", desc="Wall-clock hours of this process's training loop."),
        "n_steps": dict(unit="steps", section="summary", desc="Optimiser steps of the run."),
        "exp_name": dict(unit="name", section="summary", desc="The run folder name."),
    }, source=f"experiments/lutorch_ex/{EXP_NAME}/train.py")
    paths = {"HERE", "REPO", "NANOCHAT_ROOT", "CKPT", "METRICS", "DEVICE"}           # no local paths in the config
    cfg = {k: v for k, v in globals().items()
           if k.isupper() and k not in paths and isinstance(v, (int, float, str, tuple))}
    cfg["exp_name"] = EXP_NAME
    cfg["description"] = ("QuantisedConfidenceLUT (p2_int8, read_top_n=1) in a ProjectionMHL in every block of a "
                          "6-block GPT, the lutorch_ex sweep's champion topology h8/tph64/nap8/d48, cell-TV lambda=10, "
                          "table dropout 0.2, 48k steps, seed 1.")
    return Tracker.start(cfg, HERE, project="Spiky", group="lutorch_ex", name=EXP_NAME,
                         tags=["lutorch_ex", "quant_n1", f"tv:{TV_LAMBDA:g}", f"drop:{TABLE_DROPOUT:g}"],
                         glossary=glossary)


# ---------------------------------------------------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    tokenizer = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), "tokenizer"))
    assert tokenizer.get_vocab_size() == VOCAB
    token_bytes = get_token_bytes(device=DEVICE)
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BATCH, SEQ_LEN, split="train",
                                                                 device=DEVICE)

    model = GPT().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    tables = {id(b.ffn.cartridge.weights) for b in model.blocks}
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2 and id(p) not in tables]
    no_decay = [p for p in model.parameters() if p.requires_grad and (p.ndim < 2 or id(p) in tables)]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)],
                            lr=LR, betas=BETAS, eps=EPS)
    print(f"{EXP_NAME}: params={total_params:,} | {READ_TOP_N=} {QUANT_MODE=} {TABLE_DROPOUT=} {TV_LAMBDA=} | "
          f"{N_STEPS=} batch={DEVICE_BATCH}x{GRAD_ACCUM}x{SEQ_LEN} | {CKPT_EVERY=}")

    start, ema, best = 0, None, float("inf")
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, ema, best = ck["step"], ck["ema"], ck["best"]
        print(f"[resume] from step {start} (ema={ema:.4f}, best={best:.4f})")
        if os.path.exists(METRICS):        # drop rows logged after the checkpoint: they are re-run now
            with open(METRICS) as f:
                rows = list(csv.reader(f))
            with open(METRICS, "w", newline="") as f:
                csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if int(r[0]) <= start])

    def save_ckpt(step):
        torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "ema": ema, "best": best},
                   CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)    # atomic: a crash mid-save leaves the previous checkpoint intact

    tracker = start_tracker(model)
    new_file = not os.path.exists(METRICS) or start == 0
    csv_f = open(METRICS, "w" if new_file else "a", newline="")
    csv_w = csv.writer(csv_f)
    if new_file:
        csv_w.writerow(["step", "train_loss", "val_bpb", "lut_tv", "step_ms"])

    model.train()
    t_run = time.time()
    for step in range(start + 1, N_STEPS + 1):
        s = lr_scale(step)
        for g in opt.param_groups:
            g["lr"] = LR * s
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_step = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = next(train_loader)
            loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            loss_step += loss.item() / GRAD_ACCUM
        tv = cell_tv_penalty(model)
        (TV_LAMBDA * tv).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1e3
        ema = loss_step if ema is None else 0.99 * ema + 0.01 * loss_step
        if tracker:
            tracker.train_step(step, {"train/loss": loss_step, "train/loss_ema": ema, "train/lr": LR * s})
        if step % 100 == 0 or step <= 5:
            print(f"step {step:6d} loss={loss_step:.4f} ema={ema:.4f} tv={tv.item():.3e} lr={LR * s:.2e} "
                  f"step_ms={step_ms:.0f}", flush=True)
        if step % EVAL_EVERY == 0 or step == N_STEPS:
            model.eval()
            bpb = evaluate_bpb(model, tokenizer, token_bytes)
            model.train()
            best = min(best, bpb)
            print(f"[VAL] step {step}: bpb={bpb:.4f}", flush=True)
            # metrics.csv columns as in the sweep's runs: train_loss is this step's loss (not the EMA)
            csv_w.writerow([step, f"{loss_step:.6f}", f"{bpb:.6f}", f"{tv.item():.6e}", f"{step_ms:.1f}"])
            csv_f.flush()
            if tracker:
                tracker.eval_step(step, {"val_bpb": bpb, "train_loss": ema, "lut_tv": float(tv.item())})
        if step % CKPT_EVERY == 0 or step == N_STEPS:
            save_ckpt(step)
    csv_f.close()

    summary = {"exp_name": EXP_NAME, "best_val_bpb": best, "total_params": total_params,
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
               "hours": round((time.time() - t_run) / 3600, 4), "n_steps": N_STEPS}
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("=== DONE ===", json.dumps(summary), flush=True)
    if tracker:
        tracker.finish(summary)


if __name__ == "__main__":
    main()
