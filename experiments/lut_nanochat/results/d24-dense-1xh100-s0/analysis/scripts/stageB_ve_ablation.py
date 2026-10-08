"""
[lut_nanochat] Stage B(c): the DECISIVE value-embedding (VE) ablation on the trained d24 teacher.

Zero each of the 12 value-embedding tables (one Embedding per odd layer 1,3,...,23) one at a time, and all 12
together, and measure the change in val bits-per-byte vs a self-measured unablated baseline (same harness, same
val shard_06542, full base_eval budget). No training. Answers whether the 604M VE params are load-bearing.

SAFETY: this runs CONCURRENTLY with the live distillation trainer on the same H100. We hard-cap THIS process's
GPU memory (torch.cuda.set_per_process_memory_fraction ~0.22 = ~18 GiB) BEFORE loading the model, so if this
process ever exceeds its cap it OOMs itself and never starves the trainer. Measured Stage B eval peak is ~13.1 GiB.

Model stays resident; VE weights are swapped in place (clone -> zero -> eval -> copy_ back -> allclose check),
so the 2.8 GB checkpoint is loaded once. Writes data/stageB_ve_ablation.json and data/stageB_ve_ablation.md.

Run (pinned H100 venv, FA3), e.g.:
    HF_HUB_OFFLINE=1 NANOCHAT_FA3_REVISION=<pin> PYTHONPATH=<nanochat> \
      .venv/bin/python results/d24-dense-1xh100-s0/analysis/scripts/stageB_ve_ablation.py
"""
import os, json, time, torch

MEM_FRACTION = float(os.environ.get("STAGEB_MEM_FRACTION", "0.22"))   # ~18 GiB of 80
torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)           # cap FIRST, before any big alloc

from nanochat.checkpoint_manager import build_model
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.common import get_base_dir

DEV = torch.device("cuda")
DBS, SEQ = int(os.environ.get("STAGEB_DBS", "8")), 2048
SPLIT_TOKENS = int(os.environ.get("STAGEB_SPLIT_TOKENS", str(40 * 524288)))   # base_eval default
EVAL_STEPS = SPLIT_TOKENS // (DBS * SEQ)
ANA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))             # .../analysis
OUT_JSON = os.path.join(ANA, "data", "stageB_ve_ablation.json")
OUT_MD = os.path.join(ANA, "data", "stageB_ve_ablation.md")
CKPT_DIR = os.path.join(get_base_dir(), "base_checkpoints", "d24_1xh100")

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

model, _tok, meta = build_model(CKPT_DIR, 5568, DEV, "eval")
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=DEV)
params = dict(model.named_parameters())
ve_keys = sorted([k for k in params if k.startswith("value_embeds.") and k.endswith(".weight")],
                 key=lambda k: int(k.split(".")[1]))
log(f"model loaded; {len(ve_keys)} VE tables; DBS={DBS} seq={SEQ} split_tokens={SPLIT_TOKENS} eval_steps={EVAL_STEPS}")

def run_eval():
    # fresh val loader each time -> deterministic, identical batches across ablations (apples-to-apples)
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DBS, SEQ, split="val", device=DEV)
    with torch.inference_mode():
        return float(evaluate_bpb(model, loader, EVAL_STEPS, token_bytes))

res = {"checkpoint": "d24_1xh100@5568", "config": {"DBS": DBS, "seq": SEQ, "split_tokens": SPLIT_TOKENS,
       "eval_steps": EVAL_STEPS, "val_shard": "shard_06542", "mem_fraction": MEM_FRACTION},
       "baseline_bpb": None, "per_table": {}, "all12": None}

res["baseline_bpb"] = run_eval()
log(f"baseline val bpb = {res['baseline_bpb']:.6f}")

for k in ve_keys:
    layer = int(k.split(".")[1]); p = params[k]
    saved = p.detach().clone()
    with torch.no_grad(): p.zero_()
    b = run_eval()
    with torch.no_grad(): p.copy_(saved)
    assert torch.allclose(p, saved), f"restore mismatch for {k}"
    res["per_table"][str(layer)] = {"bpb": b, "delta": b - res["baseline_bpb"]}
    log(f"ablate layer {layer:2d}: bpb={b:.6f}  delta={b-res['baseline_bpb']:+.6f}  (restored OK)")

saved_all = {k: params[k].detach().clone() for k in ve_keys}
with torch.no_grad():
    for k in ve_keys: params[k].zero_()
b_all = run_eval()
with torch.no_grad():
    for k in ve_keys: params[k].copy_(saved_all[k])
assert all(torch.allclose(params[k], saved_all[k]) for k in ve_keys), "restore mismatch (all-12)"
res["all12"] = {"bpb": b_all, "delta": b_all - res["baseline_bpb"]}
log(f"ablate ALL 12: bpb={b_all:.6f}  delta={b_all-res['baseline_bpb']:+.6f}")

res["config"]["peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(res, open(OUT_JSON, "w"), indent=2)

# markdown summary, sorted by impact
rows = sorted(((int(l), d["bpb"], d["delta"]) for l, d in res["per_table"].items()), key=lambda r: -r[2])
with open(OUT_MD, "w") as f:
    f.write(f"# Stage B(c): value-embedding ablation (d24_1xh100 @ 5568)\n\n")
    f.write(f"Self-measured unablated baseline val bpb = **{res['baseline_bpb']:.6f}** "
            f"(val shard_06542, split_tokens={SPLIT_TOKENS}, eval_steps={EVAL_STEPS}, DBS={DBS}, FA3/SSSL, bf16).\n\n")
    f.write("| VE layer | val bpb | Δ vs baseline |\n|---|---|---|\n")
    for l, b, d in rows:
        f.write(f"| {l} | {b:.6f} | {d:+.6f} |\n")
    f.write(f"| **all 12** | {res['all12']['bpb']:.6f} | {res['all12']['delta']:+.6f} |\n\n")
    f.write(f"Peak reserved GPU memory (this process, capped at fraction {MEM_FRACTION}): "
            f"{res['config']['peak_reserved_GiB']} GiB.\n")
print("DONE " + json.dumps(res), flush=True)
