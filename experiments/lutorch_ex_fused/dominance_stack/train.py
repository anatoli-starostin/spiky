#!/usr/bin/env python
"""[dominance_stack] COPY of ../01_fmh_rational/train.py (same recipe, same everything) with ONLY these changes:
  * config path from $CONFIG (default ./config.json) and the cartridge may be a Dominance stack (`dom_fmh`) or the
    depth control (`fmh_fmh`) -- see build_cartridge; `conf_n2` / `fmh_rational` are unchanged from 05 / 01;
  * the faithful init, the no-weight-decay table group and the table count are written over "the tables of every
    LUT layer" instead of `cart.weights` (identical for the single-layer cartridges);
  * wandb group/tags, mean step time / peak memory in summary.json.
Original docstring follows.

Fused-cartridge abl47 reproduction on the refactored lutorch_ex (current `main` API).

Self-contained single-GPU trainer. The 6-block MinimalGPT is defined INLINE in this file
(RoPE + SDPA causal attention, prenorm residual blocks, untied unembedder; n_embd 384,
n_head 6, seq_len 512, vocab 32768) — no model factory, no hidden abstraction. Every block's
FFN is `ProjectionMHL(<fused cartridge>).float()` (an fp32 island), replacing the dense MLP.

The cartridge is chosen by `config.json["cartridge"]` (one of the 6 fused ablations); all other
geometry/schedule come from `config.json` (the abl47 champion topology). cell-TV λ=10 is a
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # dominance_stack modules

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import get_base_dir
from fixed_eval import evaluate_bpb_fixed, eval_config
from spiky.lutorch_ex import (LUTSpec, ProjectionMHL, cell_tv_penalty,
                              FusedManifestoHardLUT, FusedManifestoSoftLUT, FusedConfidenceLUT)
from stack import DominanceStackLUT, LUTStack

# --------------------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.environ.get("CONFIG", os.path.join(HERE, "config.json"))))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "runs", cfg["exp_name"]))
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda"
torch.manual_seed(cfg["random_seed"])

SEQ, DEVICE_BS, TOTAL_BS = cfg["seq_len"], cfg["device_batch_size"], cfg["total_batch_size"]
N_STEPS = int(os.environ.get("SMOKE_STEPS", cfg["n_steps"]))
EVAL_EVERY = (int(os.environ["SMOKE_EVAL_EVERY"]) if "SMOKE_EVAL_EVERY" in os.environ
              else (max(1, N_STEPS // 4) if "SMOKE_STEPS" in os.environ else cfg["eval_every"]))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", cfg.get("ckpt_every", 2000)))
LR, WD, WARM = cfg["lr"], cfg["weight_decay"], cfg["lr_warmup_fraction"]
TV_LAMBDA = float(cfg["lut_cell_smoothness"])          # 10.0
DROP = float(cfg["lut_head_dropout_rate"])             # 0.2 (whole-table dropout in the lib)
NOISE = float(cfg["lut_init_weights_noise"])           # 1e-3
BASE_SEED = int(cfg["lut_base_seed"])                  # 1000

# ------------------------------------------------- wandb (ONLINE-or-die; init FIRST, before any data/model/GPU work)
RUN_TAG = os.environ.get("RUN_TAG") or time.strftime("%m%d_%H%M%S")   # fresh unique name per launch (no resume)
run_name = f"{cfg['exp_name']}_{RUN_TAG}"
import wandb
try:
    wb = wandb.init(project="Spiky", group="lutorch_ex_fused_dominance_stack", name=run_name, id=run_name,
                    config=cfg, tags=["lutorch_ex", "fused", "dominance_stack", cfg["cartridge"], f"steps{cfg['n_steps']}"])
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


def build_cartridge(cfg, seed):
    """Explicit choice of one of the 6 fused cartridges (no registry/factory)."""
    spec = LUTSpec(h_in=cfg["lut_n_heads"], h_out=cfg["lut_n_heads"],
                   tph=cfg["lut_tables_per_head"], nap=cfg["lut_n_anchor_pairs"],
                   d_in=cfg["lut_inner_in_dim"], d_out=cfg["lut_inner_out_dim"],
                   anchor_mode="pairs")
    common = dict(seed=seed, weight_init_std=NOISE, table_dropout_rate=DROP)
    name = cfg["cartridge"]
    if name == "fmh_rational":
        return FusedManifestoHardLUT(spec, weight="rational", **common)
    if name == "fmh_softsign":
        return FusedManifestoHardLUT(spec, weight="softsign", **common)
    if name == "fms_rational":
        return FusedManifestoSoftLUT(spec, weight="rational", **common)
    if name == "fms_softsign":
        return FusedManifestoSoftLUT(spec, weight="softsign", **common)
    if name == "dom_fmh":     # B: Dominance (hard read, 'same' anchors, nop=nap) -> FusedManifestoHard rational (as A)
        second = FusedManifestoHardLUT(spec, weight="rational", **common)
        return DominanceStackLUT(spec, second, dom_seed=seed + int(cfg.get("dom_seed_offset", 1000)),
                                 dom_weight_init_std=cfg.get("dom_weight_init_std"),
                                 post_norm=cfg.get("post_norm", "none"))   # None -> sqrt(d_out/(2*tph*nop))
    if name in ("domconf_fmh", "domconf_conf1"):   # E1/E2: Dominance on ConfidenceLUT n=1 -> LN -> layer 2 (hard rational / conf n=1)
        if name == "domconf_fmh":
            second = FusedManifestoHardLUT(spec, weight="rational", **common)
        else:
            second = FusedConfidenceLUT(spec, read_top_n=1, beta_init=cfg["lut_learned_margin_beta_init"],
                                        gamma_init=cfg["lut_learned_margin_gamma_init"], **common)
        return DominanceStackLUT(spec, second, dom_seed=seed + int(cfg.get("dom_seed_offset", 1000)),
                                 dom_weight_init_std=cfg.get("dom_weight_init_std"),
                                 post_norm=cfg.get("post_norm", "none"), dom_conf=True,
                                 beta_init=cfg["lut_learned_margin_beta_init"],
                                 gamma_init=cfg["lut_learned_margin_gamma_init"])
    if name == "fmh_fmh":     # D: FusedManifestoHard (small tph, ~table budget of B's Dominance layer) -> FusedManifestoHard
        tph1 = int(cfg["ctrl_tph1"])
        spec1 = LUTSpec(h_in=spec.h_in, h_out=spec.h_out, tph=tph1, nap=spec.nap, d_in=spec.d_in, d_out=spec.d_out,
                        anchor_mode="pairs")
        first = FusedManifestoHardLUT(spec1, weight="rational", seed=seed + int(cfg.get("dom_seed_offset", 1000)),
                                      weight_init_std=1.0 / math.sqrt(tph1), table_dropout_rate=0.0)
        second = FusedManifestoHardLUT(spec, weight="rational", **common)
        return LUTStack(spec, first, second)
    if name in ("conf_n2", "conf_n1"):
        tau = dict(read_tau_init=cfg["lut_read_tau"], read_tau_learnable=bool(cfg["lut_read_tau_learnable"])) \
            if name == "conf_n2" else {}            # tau exists only for n=2 (n=1 raises if given)
        return FusedConfidenceLUT(spec, read_top_n=(2 if name == "conf_n2" else 1),
                                  beta_init=cfg["lut_learned_margin_beta_init"],
                                  gamma_init=cfg["lut_learned_margin_gamma_init"], **tau, **common)
    raise ValueError(f"unknown cartridge {name!r}")


class LUTFFN(nn.Module):
    """ProjectionMHL(cartridge) as an fp32 island: compress 384->r, lookup, decompress r->384."""
    def __init__(self, cfg, n_embd, seed):
        super().__init__()
        self.cart = build_cartridge(cfg, seed)
        self.proj = ProjectionMHL(self.cart, d_model=n_embd).float()

    def forward(self, x):          # x: [N, n_embd]
        return self.proj(x.float())


class MinimalBlock(nn.Module):
    def __init__(self, cfg, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = LUTFFN(cfg, n_embd, BASE_SEED + layer_idx)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        B, T, C = h.shape
        o = self.ffn(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)
        return x + o


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
            nn.init.zeros_(b.ffn.proj.decompress.weight)    # FFN starts ~0 (re-zeroed in faithful init too)

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


def final_lut(cart):
    """The ManifestoLUT whose tables get the faithful OLD init (the whole cartridge for 01/05; the 2nd layer of a stack)."""
    return getattr(cart, "second", cart)


model = MinimalGPT(cfg, VOCAB).to(DEVICE)

# --------------------------------------------------------------------------- faithful OLD-matching init
# (task 6286aa75 in the original: the lutorch_ex defaults silently diverged from OLD and were the
# sole cause of the earlier bpb gap.) Re-init the cartridge + projections to the OLD scheme exactly:
#   tables     -> Uniform[-noise, +noise], drawn per-head from a CPU torch.Generator seeded base+h+1,
#                 head-major (bit-exact to the OLD per-head CPU draw);
#   decompress -> weight zeroed (so the FFN starts ~0), bias left at default;
#   compress   -> Normal(0, 0.02) (OLD's scale), seeded base+101.
with torch.no_grad():
    for i, b in enumerate(model.blocks):
        cart, proj, base = final_lut(b.ffn.cart), b.ffn.proj, BASE_SEED + i   # the layer that is the baseline's FFN LUT
        tph, K, d_out = cart.weights.shape[1], cart.weights.shape[2], cart.weights.shape[3]
        for h in range(cfg["lut_n_heads"]):
            g = torch.Generator().manual_seed(base + h + 1)
            blk = (torch.rand(tph, K, d_out, generator=g) - 0.5) * (2.0 * NOISE)
            cart.weights[h] = blk.to(cart.weights.device, cart.weights.dtype)
        proj.decompress.weight.zero_()
        gC = torch.Generator().manual_seed(base + 101)
        proj.compress.weight.copy_(torch.empty(proj.compress.weight.shape)
                                   .normal_(0.0, 0.02, generator=gC)
                                   .to(proj.compress.weight.device, proj.compress.weight.dtype))

# DominanceConf: the score s (~0.2 at init) shrinks the Borda output; rescale the delta tables of each Dominance layer so the
# pre-LN hidden std is 1 at init. Calibrated on N(0,1) tokens (= the LN'd FFN input) through the layer's own (init) compress
# projection, with a private generator (global RNG untouched). Only the init scale; LN makes the scale unimportant.
if cfg.get("dom_calibrate", False):
    with torch.no_grad():
        gcal = torch.Generator(device=DEVICE).manual_seed(12345)
        xcal = torch.randn(4096, cfg["n_embd"], device=DEVICE, generator=gcal)
        for i, b in enumerate(model.blocks):
            st, proj = b.ffn.cart, b.ffn.proj
            hh = st.dom(proj.compress(xcal).reshape(4096, cfg["lut_n_heads"], cfg["lut_inner_in_dim"]))
            sd = hh.std(dim=-1, unbiased=False).mean().item()
            st.dom.weights.mul_(1.0 / sd)
            hh2 = st.dom(proj.compress(xcal).reshape(4096, cfg["lut_n_heads"], cfg["lut_inner_in_dim"]))
            print(f"[calib] block {i}: dom hidden std at init {sd:.4f} -> rescaled x{1.0/sd:.3f} -> {hh2.std(dim=-1, unbiased=False).mean().item():.4f}")

total_params = sum(p.numel() for p in model.parameters())
print(f"cartridge={cfg['cartridge']} blocks={len(model.blocks)} params={total_params:,} "
      f"tables_std(b0)={final_lut(model.blocks[0].ffn.cart).weights.std().item():.6f} "
      f"compress_std(b0)={model.blocks[0].ffn.proj.compress.weight.std().item():.4f} "
      f"decompress_zero(b0)={bool((model.blocks[0].ffn.proj.decompress.weight == 0).all().item())} "
      f"TV={TV_LAMBDA} drop={DROP} n_steps={N_STEPS}")

# --------------------------------------------------------------------------- optimizer + schedule
lut_table_ids = {id(p) for b in model.blocks for n, p in b.ffn.cart.named_parameters() if n.split(".")[-1] == "weights"}
n_table_params = sum(p.numel() for b in model.blocks for n, p in b.ffn.cart.named_parameters() if n.split(".")[-1] == "weights")
print(f"LUT table params: {n_table_params:,}")
decay, nodecay = [], []
for p in model.parameters():
    if p.requires_grad:
        (nodecay if (id(p) in lut_table_ids or p.ndim < 2) else decay).append(p)
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
    csv_w.writerow(["step", "train_loss", "val_bpb", "lut_tv", "step_ms", "hidden_std_pre_ln", "score_mean"]); csv_f.flush()

# --------------------------------------------------------------------------- train loop
step_times = []
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
    tv = cell_tv_penalty(model)
    if TV_LAMBDA > 0:
        (TV_LAMBDA * tv).backward()                    # SEPARATE backward for the cell-TV penalty
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize(); step_ms = (time.perf_counter() - t0) * 1e3
    step_times.append(step_ms)
    ema = acc if ema is None else 0.99 * ema + 0.01 * acc
    wb.log({"step": step, "train_loss": acc, "train_ema": ema, "lut_tv": float(tv.item()),
           "lr": LR * s, "step_ms": step_ms}, step=step)
    if step % 20 == 0 or step <= 5:
        print(f"step {step:6d} loss={acc:.4f} ema={ema:.4f} tv={tv.item():.3e} lr={LR*s:.2e} ms={step_ms:.0f}")
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb_fixed(model, tok, token_bytes, SEQ, DEVICE, **EVAL)
        model.train()
        stacks = [b.ffn.cart for b in model.blocks if isinstance(b.ffn.cart, DominanceStackLUT)]
        hstd = sum(c.mean_hidden_std() for c in stacks) / len(stacks) if stacks else float("nan")
        hstd_l = [round(c.mean_hidden_std(), 4) for c in stacks]
        sc = sum(c.mean_score() for c in stacks) / len(stacks) if stacks else float("nan")
        sc_l = [round(c.mean_score(), 4) for c in stacks]
        for c in stacks: c.reset_hidden_std()
        best = min(best, bpb)
        print(f"[VAL] step {step}: bpb={bpb:.4f} (best {best:.4f})")
        if stacks:
            print(f"[VAL] hidden std before LN (mean over tokens, heads; per layer {hstd_l}): {hstd:.4f}")
            if sc == sc:
                print(f"[VAL] mean score s (per layer {sc_l}): {sc:.4f}")
            wb.log({"step": step, "val_bpb": bpb, "hidden_std_pre_ln": hstd, **({"score_mean": sc} if sc == sc else {})}, step=step)
        else:
            wb.log({"step": step, "val_bpb": bpb}, step=step)
        csv_w.writerow([step, f"{acc:.6f}", f"{bpb:.6f}", f"{tv.item():.6e}", f"{step_ms:.1f}", f"{hstd:.5f}", f"{sc:.5f}"]); csv_f.flush()
    if step % CKPT_EVERY == 0 or step == N_STEPS:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT)

csv_f.close()
peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
summary = {"exp_name": cfg["exp_name"], "cartridge": cfg["cartridge"], "best_val_bpb": best, "final_val_bpb": bpb,
           "total_params": total_params, "table_params": n_table_params,
           "mean_step_ms": (sum(step_times[10:]) / max(1, len(step_times[10:]))) if len(step_times) > 10 else None, "peak_gb": peak_gb,
           "hours": round((time.time() - t_run) / 3600, 4), "n_steps": N_STEPS, "run_name": run_name}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
print("=== DONE ===", json.dumps(summary))
wb.summary.update(summary); wb.finish()
