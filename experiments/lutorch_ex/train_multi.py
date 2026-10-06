"""Champion-topology lutorch_ex runs for the new-library report (SCRATCH). Cartridge via env CART:
  fss_smooth -> FusedSoftSignSmoothLUT   conf_n1 -> ConfidenceLUT(read_top_n=1)
  fmh -> FusedManifestoHardLUT           fms -> FusedManifestoSoftLUT   fss_hard -> FusedSoftSignHardLUT
Champion topology (h8/tph64/nap8/d48) WITH TV (lambda=10) + head dropout (0.2), 48k steps, seed1, champion
LR/WD schedule from the abl_47 config.json. Uses the LIBRARY-DEFAULT faithful small-std init (now on main):
NO manual init override. Checkpoints to OUT_DIR/ckpt.pt every CKPT_EVERY steps and resumes from it on restart
(stable wandb run name/id via a persisted RUN_TAG, resume='allow'). wandb group 'lutorch_ex', project Spiky."""
import os, sys, json, math, time, csv

TOOLS = os.environ.get("TOOLS_DIR", "/home/astarostin/projects/ffn_fix_wt/experiments/ffn_replacement/tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.environ.get("NANOCHAT_ROOT", "/home/astarostin/projects/nanochat"))
import torch, torch.nn as nn
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import get_base_dir
from model_build import build_model
from fixed_eval import evaluate_bpb_fixed, eval_config
import wandb_tracking
from spiky.lutorch_ex import (LUTSpec, ConfidenceLUT, QuantisedConfidenceLUT, ProjectionMHL, cell_tv_penalty,
                              FusedSoftSignSmoothLUT, FusedManifestoHardLUT, FusedManifestoSoftLUT,
                              FusedSoftSignHardLUT)

CART = os.environ["CART"]
_BUILDERS = {"fss_smooth": "FusedSoftSignSmoothLUT", "conf_n1": "ConfidenceLUT(read_top_n=1)",
             "conf_n2": "ConfidenceLUT(read_top_n=2)", "quant": "QuantisedConfidenceLUT(p2_int8, read_top_n=2)",
             "fmh": "FusedManifestoHardLUT", "fms": "FusedManifestoSoftLUT", "fss_hard": "FusedSoftSignHardLUT"}
assert CART in _BUILDERS, f"CART must be one of {list(_BUILDERS)}, got {CART}"
CFG = os.environ["ABL47_CONFIG"]
OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True)
CKPT = os.path.join(OUT_DIR, "ckpt.pt")
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "1000"))
cfg = json.load(open(CFG))
DEVICE = "cuda"
torch.manual_seed(cfg["random_seed"])
SEQ = cfg["seq_len"]; DEVICE_BS = cfg["device_batch_size"]; TOTAL_BS = cfg["total_batch_size"]
N_STEPS = int(os.environ.get("SMOKE_STEPS", cfg["n_steps"]))
EVAL_EVERY = int(os.environ.get("SMOKE_EVAL_EVERY", cfg["eval_every"])) if "SMOKE_STEPS" in os.environ else cfg["eval_every"]
LR, WD, WARM = cfg["lr"], cfg["weight_decay"], cfg["lr_warmup_fraction"]
TV_LAMBDA = float(cfg["lut_cell_smoothness"])          # 10.0
DROP = float(cfg["lut_head_dropout_rate"])             # 0.2

# Stable run tag: persist on first launch so a resume reuses the SAME wandb run name/id.
_tag_file = os.path.join(OUT_DIR, "run_tag.txt")
if os.path.exists(_tag_file):
    RUN_TAG = open(_tag_file).read().strip()
else:
    RUN_TAG = os.environ.get("RUN_TAG") or time.strftime("%m%d_%H%M")
    open(_tag_file, "w").write(RUN_TAG)

tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), "tokenizer"))
VOCAB = tok.get_vocab_size(); assert VOCAB == cfg["tokenizer_vocab_size"]
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DEVICE_BS, SEQ, split="train", device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)
EVAL = eval_config(cfg)

# skeleton with FLOAT LightMHL + no OLD TV/dropout (applied on lutorch_ex); avoids OLD pow2 JIT stall
cb = dict(cfg); cb["lut_cell_smoothness"] = 0.0; cb["lut_head_dropout_rate"] = 0.0; cb["lut_quant_mode"] = None
model = build_model(cb, VOCAB, device=DEVICE)


def make_cart(seed):
    spec = LUTSpec(h_in=cfg["lut_n_heads"], h_out=cfg["lut_n_heads"], tph=cfg["lut_tables_per_head"],
                   nap=cfg["lut_n_anchor_pairs"], d_in=cfg["lut_inner_in_dim"], d_out=cfg["lut_inner_out_dim"])
    base = dict(seed=seed, weight_init_std=cfg["lut_init_weights_noise"], table_dropout_rate=DROP)
    if CART in ("conf_n1", "conf_n2"):
        return ConfidenceLUT(spec, read_top_n=(1 if CART == "conf_n1" else 2),
                             beta_init=cfg["lut_learned_margin_beta_init"],
                             gamma_init=cfg["lut_learned_margin_gamma_init"], read_tau_init=cfg["lut_read_tau"],
                             read_tau_learnable=bool(cfg["lut_read_tau_learnable"]), **base)
    if CART == "quant":
        return QuantisedConfidenceLUT(spec, quant_mode="p2_int8", read_top_n=2,
                             beta_init=cfg["lut_learned_margin_beta_init"],
                             gamma_init=cfg["lut_learned_margin_gamma_init"], read_tau_init=cfg["lut_read_tau"],
                             read_tau_learnable=bool(cfg["lut_read_tau_learnable"]), **base)
    cls = {"fss_smooth": FusedSoftSignSmoothLUT, "fmh": FusedManifestoHardLUT,
           "fms": FusedManifestoSoftLUT, "fss_hard": FusedSoftSignHardLUT}[CART]
    return cls(spec, **base)


class LutorchExFFN(nn.Module):
    def __init__(self, n_embd, seed):
        super().__init__()
        self.cart = make_cart(seed)
        self.proj = ProjectionMHL(self.cart, d_model=n_embd).float()      # fp32 island
    def forward(self, x):
        return self.proj(x.float())


carts = []
for i, b in enumerate(model.blocks):
    f = LutorchExFFN(cfg["n_embd"], cfg["lut_base_seed"] + i).to(DEVICE)
    b.ffn = f; carts.append(f.cart)
# NO manual init override: the library default (faithful OLD small-std) is now the main default.
print(f"[init] CART={CART} ({_BUILDERS[CART]}) tables std(b0)={carts[0].weights.std().item():.6f} "
      f"compress std(b0)={model.blocks[0].ffn.proj.compress.weight.std().item():.6f} "
      f"decompress all_zero(b0)={bool((model.blocks[0].ffn.proj.decompress.weight==0).all().item())}")
total_params = sum(p.numel() for p in model.parameters())
print(f"CART={CART} blocks={len(model.blocks)} carts={len(carts)} params={total_params:,} "
      f"TV_lambda={TV_LAMBDA} dropout={DROP} N_STEPS={N_STEPS} RUN_TAG={RUN_TAG}")

lut_ids = {id(c.weights) for c in carts}
decay, nodecay = [], []
for p in model.parameters():
    if p.requires_grad:
        (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
opt = torch.optim.AdamW([dict(params=decay, weight_decay=WD), dict(params=nodecay, weight_decay=0.0)],
                        lr=LR, betas=(0.9, 0.95), eps=1e-8)
for g in opt.param_groups: g["initial_lr"] = g["lr"]
grad_accum = max(1, TOTAL_BS // (DEVICE_BS * SEQ))


def lr_scale(step):
    w = int(WARM * N_STEPS)
    if step < w: return step / max(w, 1)
    p = (step - w) / max(N_STEPS - w, 1); return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


# ---- resume from checkpoint if present ----
start_step, ema, best = 0, None, float("inf")
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    start_step = ck["step"]; ema = ck["ema"]; best = ck["best"]
    print(f"[resume] from step {start_step} (best={best:.4f}) ckpt={CKPT}")


def save_ckpt(step):
    tmp = CKPT + ".tmp"
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                "ema": ema, "best": best, "run_tag": RUN_TAG}, tmp)
    os.replace(tmp, CKPT)   # atomic


wandb_tracking.GROUP = "lutorch_ex"
from wandb_tracking import Tracker
cfg_log = dict(cfg); cfg_log["exp_name"] = f"lutorch_ex_abl47_{CART}_{RUN_TAG}"; cfg_log["lut_impl"] = f"lutorch_ex_{CART}"
# Human-readable, accurate wandb notes per run. The champion config.json carries stale "_*_note" blobs
# (a quantised-arm description that even leaks internal task ids); strip them and set a clean description
# of what THIS run actually trains, so the notes are correct and task-id-free from creation.
_DESC = {"fss_smooth": "FusedSoftSignSmooth cartridge (Gen-2 soft-sign, smooth blended read).",
         "conf_n1": "ConfidenceLUT with read_top_n=1 (learned-margin confidence score, single-cell read).",
         "conf_n2": "ConfidenceLUT with read_top_n=2 (learned-margin confidence score, two-cell tau-blended read).",
         "quant": "QuantisedConfidenceLUT read_top_n=2, p2_int8 (power-of-two int8 quant-aware read, straight-through STE).",
         "fmh": "FusedManifestoHard cartridge (hard argmax LUT read).",
         "fms": "FusedManifestoSoft cartridge (soft temperature-blended LUT read).",
         "fss_hard": "FusedSoftSignHard cartridge (Gen-2 soft-sign, hard read)."}
for _k in [k for k in list(cfg_log) if k.endswith("_note")]:
    cfg_log.pop(_k, None)
cfg_log["description"] = (f"{_DESC[CART]} Champion topology h8/tph64/nap8/d48 (6 blocks, ~68.2M params), "
                          f"cell-TV lambda=10 and head-dropout 0.2, 48k-step cosine schedule, seed 1, "
                          f"faithful small-std default init.")
tracker = Tracker.start(cfg_log, OUT_DIR, grad_accum=grad_accum, total_params=total_params,
                        extra_tags=["lutorch_ex", CART, f"tv:{TV_LAMBDA:g}", f"drop:{DROP:g}"])
csv_path = os.path.join(OUT_DIR, "metrics.csv")
csv_f = open(csv_path, "a" if start_step else "w", newline=""); csv_w = csv.writer(csv_f)
if not start_step:
    csv_w.writerow(["step", "train_loss", "val_bpb", "lut_tv", "step_ms"])

model.train(); t_run = time.time()
for step in range(start_step + 1, N_STEPS + 1):
    s = lr_scale(step)
    for g in opt.param_groups: g["lr"] = g["initial_lr"] * s
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    acc = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y); (loss / grad_accum).backward(); acc += loss.item() / grad_accum
    tv = cell_tv_penalty(model)
    if TV_LAMBDA > 0: (TV_LAMBDA * tv).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize(); step_ms = (time.perf_counter() - t0) * 1e3
    ema = acc if ema is None else 0.99 * ema + 0.01 * acc
    tracker.train_step(step, acc, ema, s * LR)
    if step % 20 == 0 or step == 1 or step <= 5:
        print(f"step {step:5d} loss={acc:.4f} ema={ema:.4f} tv={tv.item():.3e} lr={s*LR:.2e} step_ms={step_ms:.0f}")
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval(); bpb = evaluate_bpb_fixed(model, tok, token_bytes, SEQ, DEVICE, **EVAL); model.train()
        best = min(best, bpb)
        print(f"[VAL] step {step}: bpb={bpb:.4f}")
        csv_w.writerow([step, f"{acc:.6f}", f"{bpb:.6f}", f"{tv.item():.6e}", f"{step_ms:.1f}"]); csv_f.flush()
        tracker.eval_step(step, bpb, ema, {"lut_tv": float(tv.item())}, model)
    if step % CKPT_EVERY == 0 or step == N_STEPS:
        save_ckpt(step)
csv_f.close()
peak = torch.cuda.max_memory_allocated() / 1e9
summary = {"exp_name": cfg_log["exp_name"], "cart": CART, "best_val_bpb": best,
           "total_params": total_params, "peak_gb": round(peak, 3),
           "hours": round((time.time() - t_run) / 3600, 4), "n_steps": N_STEPS}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
print("=== DONE ===", json.dumps(summary)); tracker.finish(summary)
