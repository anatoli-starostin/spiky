"""Precise lutorch_ex reproduction of the NON-QUANT champion
(exp_n_abl_47_..._tph64_h8_seed1_noquant_headdrop20, best_val_bpb 1.10792), with FULL weight transfer
from the OLD champion so the initial full-model forward is bit-identical (max|Δ|=0).

NEW = ProjectionMHL(ConfidenceLUT)  (float scored LUT, lut_quant_mode=None — NO quantisation).
Init = transfer OLD LightMHL/CompressionMHL weights (tables, β/γ/τ, compress+bias, decompress+bias) into NEW;
the backbone is already identical (same build_model + seed). Anchors/scalars already match by construction.

VERIFY_ONLY=1 (default): build + transfer + one-batch forward compare, print max|Δ|/allclose, exit (no training).
VERIFY_ONLY=0: after the verify passes, train N_STEPS (full 48k LR schedule) with OLD optimizer grouping,
TV λ=10, table dropout 0.2, fixed eval; wandb per WANDB_MODE/exp_name. Intended for the worker VM."""
import os, sys, json, math, time, csv
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, os.environ.get("TOOLS_DIR", "/home/astarostin/projects/ffn_fix_wt/experiments/ffn_replacement/tools"))
sys.path.insert(0, os.environ.get("NANOCHAT_ROOT", "/home/astarostin/projects/nanochat"))
import torch, torch.nn as nn
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import get_base_dir
from model_build import build_model
from fixed_eval import evaluate_bpb_fixed, eval_config
from spiky.lutorch_ex import LUTSpec, ConfidenceLUT, ProjectionMHL, cell_tv_penalty

CFG = os.environ["NOQUANT_CONFIG"]; cfg = json.load(open(CFG)); DEV = "cuda"
assert cfg.get("lut_quant_mode") in (None, "none"), f"expected non-quant champion, got quant_mode={cfg.get('lut_quant_mode')}"
VOCAB = cfg["tokenizer_vocab_size"]; SEQ = cfg["seq_len"]; DBS = cfg["device_batch_size"]; TBS = cfg["total_batch_size"]
LR, WD, WARM = cfg["lr"], cfg["weight_decay"], cfg["lr_warmup_fraction"]
TV_LAMBDA = float(cfg["lut_cell_smoothness"]); DROP = float(cfg["lut_head_dropout_rate"])
N_STEPS = int(os.environ.get("N_STEPS", cfg["n_steps"])); LR_HORIZON = int(cfg["n_steps"]); EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 500))
VERIFY_ONLY = os.environ.get("VERIFY_ONLY", "1") == "1"
G, tph, nap = cfg["lut_n_heads"], cfg["lut_tables_per_head"], cfg["lut_n_anchor_pairs"]

tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), "tokenizer"))
token_bytes = get_token_bytes(device=DEV); EVAL = eval_config(cfg)

# OLD champion (float LightMHL) at seed 1 -> the transfer source (its init == the champion's init)
torch.manual_seed(cfg["random_seed"]); old = build_model(cfg, VOCAB, device=DEV)
# NEW skeleton (same build_model + seed -> identical backbone), then swap FFN for ProjectionMHL(ConfidenceLUT)
cb = dict(cfg); cb["lut_cell_smoothness"] = 0.0; cb["lut_head_dropout_rate"] = 0.0; cb["lut_quant_mode"] = None
torch.manual_seed(cfg["random_seed"]); new = build_model(cb, VOCAB, device=DEV)
class FFN(nn.Module):
    def __init__(self, n_embd, seed):
        super().__init__()
        spec = LUTSpec(h_in=G, h_out=G, tph=tph, nap=nap, d_in=cfg["lut_inner_in_dim"], d_out=cfg["lut_inner_out_dim"])
        self.cart = ConfidenceLUT(spec, seed=seed, weight_init_std=cfg["lut_init_weights_noise"],
            read_top_n=cfg["lut_read_top_n"], beta_init=cfg["lut_learned_margin_beta_init"],
            gamma_init=cfg["lut_learned_margin_gamma_init"], read_tau_init=cfg["lut_read_tau"],
            read_tau_learnable=bool(cfg["lut_read_tau_learnable"]), table_dropout_rate=DROP)
        self.proj = ProjectionMHL(self.cart, d_model=n_embd).float()
    def forward(self, x): return self.proj(x.float())
for i, b in enumerate(new.blocks):
    b.ffn = FFN(cfg["n_embd"], cfg["lut_base_seed"] + i).to(DEV)

# ---- FULL WEIGHT TRANSFER OLD -> NEW (bit-identical init incl. compress + biases) ----
with torch.no_grad():
    for ob, nb in zip(old.blocks, new.blocks):
        of, nf = ob.ffn, nb.ffn; ll = of.lut_light
        assert torch.equal(nf.cart.anchor_a, ll.anchor_a) and torch.equal(nf.cart.anchor_b, ll.anchor_b)
        G_, tph_, K_, dout_ = nf.cart.weights.shape
        nf.cart.weights.copy_(ll.tables.reshape(G_, tph_, K_, dout_))
        nf.cart.confidence_log_beta.copy_(ll.confidence_log_beta)
        nf.cart.confidence_log_gamma.copy_(ll.confidence_log_gamma)
        nf.cart.log_read_tau.copy_(ll.log_tau)
        nf.proj.compress.weight.copy_(of.compress.weight); nf.proj.compress.bias.copy_(of.compress.bias)
        nf.proj.decompress.weight.copy_(of.decompress.weight); nf.proj.decompress.bias.copy_(of.decompress.bias)

# ---- VERIFY: identical full-model forward on one val batch ----
old.eval(); new.eval()
vloader = tokenizing_distributed_data_loader_bos_bestfit(tok, 8, SEQ, split="val", device=DEV)
xb, yb = next(iter(vloader)); xb = xb.clone(); yb = yb.clone()
with torch.no_grad():
    lo = old(xb, yb, loss_reduction="none").float(); ln = new(xb, yb, loss_reduction="none").float()
mad = (lo - ln).abs().max().item(); rel = mad / lo.abs().max().clamp_min(1e-30).item()
print(f"[VERIFY] non-quant full-model per-token loss max|Δ|={mad:.3e} (rel {rel:.3e}); allclose(1e-5)={torch.allclose(lo, ln, atol=1e-5)}")
if VERIFY_ONLY:
    print("[VERIFY-ONLY] transfer verified; NOT training (awaiting worker VM). Set VERIFY_ONLY=0 to train."); sys.exit(0)
if mad > 2e-3:
    print(f"[VERIFY] FAILED max|Δ|={mad:.3e} -- not training."); sys.exit(2)

# ---- TRAIN (worker VM) ----
OUT = os.environ["OUT_DIR"]; os.makedirs(OUT, exist_ok=True)
import wandb_tracking; wandb_tracking.GROUP = os.environ.get("WANDB_GROUP", "lutorch_ex")
from wandb_tracking import Tracker
lut_ids = {id(b.ffn.cart.weights) for b in new.blocks}
decay, nod = [], []
for p in new.parameters():
    if p.requires_grad: (nod if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
opt = torch.optim.AdamW([dict(params=decay, weight_decay=WD), dict(params=nod, weight_decay=0.0)], lr=LR, betas=(0.9, 0.95), eps=1e-8)
for g in opt.param_groups: g["initial_lr"] = g["lr"]
grad_accum = max(1, TBS // (DBS * SEQ))
def lr_scale(s):
    w = int(WARM * LR_HORIZON)
    return s / max(w, 1) if s < w else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - w) / max(LR_HORIZON - w, 1)))
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DBS, SEQ, split="train", device=DEV)
cl = dict(cfg); cl["exp_name"] = os.environ.get("RUN_NAME", "lutorch_ex_noquant_champion_xfer"); cl["lut_impl"] = "lutorch_ex_noquant_xfer"
tr = Tracker.start(cl, OUT, grad_accum=grad_accum, total_params=sum(p.numel() for p in new.parameters()), extra_tags=["lutorch_ex", "noquant", "xfer"])
f = open(os.path.join(OUT, "metrics.csv"), "w", newline=""); w = csv.writer(f); w.writerow(["step", "train_loss", "val_bpb", "lut_tv"])
new.train(); ema = None; best = float("inf")
for step in range(1, N_STEPS + 1):
    s = lr_scale(step)
    for g in opt.param_groups: g["lr"] = g["initial_lr"] * s
    opt.zero_grad(set_to_none=True); acc = 0.0
    for _ in range(grad_accum):
        x, y = next(loader); loss = new(x, y); (loss / grad_accum).backward(); acc += loss.item() / grad_accum
    tv = cell_tv_penalty(new)
    if TV_LAMBDA > 0: (TV_LAMBDA * tv).backward()
    torch.nn.utils.clip_grad_norm_(new.parameters(), 1.0); opt.step()
    ema = acc if ema is None else 0.99 * ema + 0.01 * acc
    tr.train_step(step, acc, ema, s * LR)
    if step % 100 == 0 or step == 1: print(f"step {step} loss={acc:.4f} tv={tv.item():.3e}", flush=True)
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        new.eval(); bpb = evaluate_bpb_fixed(new, tok, token_bytes, SEQ, DEV, **EVAL); new.train()
        best = min(best, bpb); print(f"[VAL] step {step}: bpb={bpb:.4f}", flush=True)
        w.writerow([step, f"{acc:.6f}", f"{bpb:.6f}", f"{tv.item():.6e}"]); f.flush()
        tr.eval_step(step, bpb, ema, {"lut_tv": float(tv.item())}, new)
f.close(); tr.finish({"exp_name": cl["exp_name"], "best_val_bpb": best, "n_steps": N_STEPS})
print(f"=== NOQUANT REPRO DONE best_val_bpb={best:.5f} ===", flush=True)
