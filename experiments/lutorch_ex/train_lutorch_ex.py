"""abl_47 reproduction with lutorch_ex (SCRATCH, uncommitted). Variant via argv[1] = quant|conf.
Builds the research MinimalGPT skeleton (FLOAT LightMHL, quant_mode=None -> no OLD pow2 JIT stall),
REPLACES every block FFN with ProjectionMHL(QuantisedConfidenceLUT|ConfidenceLUT) + fp32 island, wires
cell_tv (lambda=10, separate backward) + table_dropout_rate=0.2, a no-weight-decay group for the cartridge
tables, and trains with the research data loader + fixed eval + wandb group 'lutorch_ex'. Set SMOKE_STEPS
to run a short sanity (overrides n_steps + eval_every). Reads the abl_47 config.json; writes to OUT_DIR."""
import os, sys, json, math, time, csv

TOOLS = "/home/astarostin/projects/ffn_fix_wt/experiments/ffn_replacement/tools"
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.environ.get("NANOCHAT_ROOT", "/home/astarostin/projects/nanochat"))
import torch, torch.nn as nn
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import get_base_dir
from model_build import build_model
from fixed_eval import evaluate_bpb_fixed, eval_config
import wandb_tracking
from spiky.lutorch_ex import LUTSpec, ConfidenceLUT, QuantisedConfidenceLUT, ProjectionMHL, cell_tv_penalty

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "quant"
assert VARIANT in ("quant", "conf")
CFG = os.environ["ABL47_CONFIG"]
OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True)
cfg = json.load(open(CFG))
DEVICE = "cuda"
torch.manual_seed(cfg["random_seed"])
SEQ = cfg["seq_len"]; DEVICE_BS = cfg["device_batch_size"]; TOTAL_BS = cfg["total_batch_size"]
N_STEPS = int(os.environ.get("SMOKE_STEPS", cfg["n_steps"]))
EVAL_EVERY = int(os.environ.get("SMOKE_EVAL_EVERY", cfg["eval_every"])) if "SMOKE_STEPS" in os.environ else cfg["eval_every"]
LR, WD, WARM = cfg["lr"], cfg["weight_decay"], cfg["lr_warmup_fraction"]
TV_LAMBDA = float(cfg["lut_cell_smoothness"])          # 10.0
DROP = float(cfg["lut_head_dropout_rate"])             # 0.2

tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), "tokenizer"))
VOCAB = tok.get_vocab_size(); assert VOCAB == cfg["tokenizer_vocab_size"]
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DEVICE_BS, SEQ, split="train", device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)
EVAL = eval_config(cfg)

# skeleton with FLOAT LightMHL + no OLD TV/dropout (we apply both on lutorch_ex); avoids OLD pow2 JIT stall
cb = dict(cfg); cb["lut_cell_smoothness"] = 0.0; cb["lut_head_dropout_rate"] = 0.0; cb["lut_quant_mode"] = None
model = build_model(cb, VOCAB, device=DEVICE)

class LutorchExFFN(nn.Module):
    def __init__(self, n_embd, seed):
        super().__init__()
        spec = LUTSpec(h_in=cfg["lut_n_heads"], h_out=cfg["lut_n_heads"], tph=cfg["lut_tables_per_head"],
                       nap=cfg["lut_n_anchor_pairs"], d_in=cfg["lut_inner_in_dim"], d_out=cfg["lut_inner_out_dim"])
        common = dict(seed=seed, weight_init_std=cfg["lut_init_weights_noise"], read_top_n=cfg["lut_read_top_n"],
                      beta_init=cfg["lut_learned_margin_beta_init"], gamma_init=cfg["lut_learned_margin_gamma_init"],
                      read_tau_init=cfg["lut_read_tau"], read_tau_learnable=bool(cfg["lut_read_tau_learnable"]),
                      table_dropout_rate=DROP)
        self.cart = (QuantisedConfidenceLUT(spec, quant_mode=cfg["lut_quant_mode"], **common)
                     if VARIANT == "quant" else ConfidenceLUT(spec, **common))
        self.proj = ProjectionMHL(self.cart, d_model=n_embd).float()      # fp32 island
    def forward(self, x):
        return self.proj(x.float())

carts = []
for i, b in enumerate(model.blocks):
    f = LutorchExFFN(cfg["n_embd"], cfg["lut_base_seed"] + i).to(DEVICE)
    b.ffn = f; carts.append(f.cart)

# FAITHFUL OLD-matching init (task 6286aa75): the lutorch_ex defaults silently diverged from OLD and
# were the sole cause of the earlier bpb gap. Re-init to OLD LightMHL/CompressionMHL scheme exactly:
#   cartridge LUT tables  -> Uniform[-noise, +noise]  (std ~= noise/sqrt(3) ~= 5.77e-4), NOT Normal(0,noise);
#   compress/decompress   -> Normal(0, 0.02)          (OLD custom scale), NOT PyTorch-default kaiming (~0.0295).
_NOISE = float(cfg["lut_init_weights_noise"]); _PROJ_STD = 0.02
with torch.no_grad():
    for i, b in enumerate(model.blocks):
        f = b.ffn; base = cfg["lut_base_seed"] + i
        # tables: OLD's EXACT per-head draw -> BIT-EXACT. CPU generator seeded base+h+1 (NOT cuda:
        # OLD drew on CPU), (torch.rand(tph,K,d_out)-0.5)*2*noise, head-major into weights[h].
        for h in range(cfg["lut_n_heads"]):
            g = torch.Generator().manual_seed(base + h + 1)
            blk = (torch.rand(cfg["lut_tables_per_head"], f.cart.weights.shape[2], f.cart.weights.shape[3],
                              generator=g) - 0.5) * (2.0 * _NOISE)
            f.cart.weights[h] = blk.to(f.cart.weights.device, f.cart.weights.dtype)
        # decompress weight ZEROED (OLD zeroes it so the FFN starts ~0); bias left at default.
        f.proj.decompress.weight.zero_()
        # compress weight Normal(0,0.02) — OLD's scale (exact values are from build_model's global RNG,
        # not bit-reproducible by a standalone re-init; scale/distribution matched).
        gC = torch.Generator().manual_seed(base + 101)
        f.proj.compress.weight.copy_(torch.empty(f.proj.compress.weight.shape).normal_(0.0, _PROJ_STD, generator=gC)
                                     .to(f.proj.compress.weight.device, f.proj.compress.weight.dtype))
print(f"[faithinit v2] tables std(b0)={carts[0].weights.std().item():.6f} (Uniform[±{_NOISE:.0e}], bit-exact to OLD per-head CPU draw); "
      f"compress std(b0)={model.blocks[0].ffn.proj.compress.weight.std().item():.6f} (~{_PROJ_STD}); "
      f"decompress all_zero(b0)={bool((model.blocks[0].ffn.proj.decompress.weight==0).all().item())}")
total_params = sum(p.numel() for p in model.parameters())
print(f"VARIANT={VARIANT} blocks={len(model.blocks)} carts={len(carts)} params={total_params:,} "
      f"TV_lambda={TV_LAMBDA} dropout={DROP} N_STEPS={N_STEPS}")

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

wandb_tracking.GROUP = "lutorch_ex"
from wandb_tracking import Tracker
# UNIQUE run name/id per launch: the research tracker does wandb.init(name=id=exp_name, resume='allow'),
# so reusing a bare "lutorch_ex_abl47_quant" RESUMES a prior (e.g. killed) run and silently DISCARDS new
# steps as non-monotonic. A launch-time suffix (overridable via RUN_TAG) guarantees a fresh wandb run.
_RUN_TAG = os.environ.get("RUN_TAG") or time.strftime("%m%d_%H%M")
cfg_log = dict(cfg); cfg_log["exp_name"] = f"lutorch_ex_abl47_{VARIANT}_{_RUN_TAG}"; cfg_log["lut_impl"] = f"lutorch_ex_{VARIANT}"
tracker = Tracker.start(cfg_log, OUT_DIR, grad_accum=grad_accum, total_params=total_params,
                        extra_tags=[f"lutorch_ex", VARIANT, f"tv:{TV_LAMBDA:g}", f"drop:{DROP:g}"])
csv_f = open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline=""); csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "train_loss", "val_bpb", "lut_tv", "step_ms"])

model.train(); ema = None; best = float("inf"); t_run = time.time()
for step in range(1, N_STEPS + 1):
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
csv_f.close()
peak = torch.cuda.max_memory_allocated() / 1e9
summary = {"exp_name": cfg_log["exp_name"], "variant": VARIANT, "best_val_bpb": best,
           "total_params": total_params, "peak_gb": round(peak, 3),
           "hours": round((time.time() - t_run) / 3600, 4), "n_steps": N_STEPS}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
print("=== DONE ===", json.dumps(summary)); tracker.finish(summary)
