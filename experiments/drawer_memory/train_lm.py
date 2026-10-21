"""One character-level language-model run on Tiny Shakespeare -> runs/<name>/{config.json, metrics.csv, summary.json, sample.txt}.

Same model/mixers as the MQAR runs. First 90% of the corpus is training text, last 10% validation.
Metric: validation loss in bits per character (bpc), on a fixed set of windows.
No positional encoding for any mixer; local order comes from the token shift over shift_k previous chars.
"""
import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from models import Model, set_hardness
from thermal import Thermal

HERE = os.path.dirname(os.path.abspath(__file__))


def load_corpus(name="shakespeare"):
    if name == "enwik8":  # standard byte-level split: first 90M train, next 5M validation (last 5M test, unused)
        raw = open(os.path.join(HERE, "data", "enwik8", "enwik8"), "rb").read()
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        chars = [bytes([i]).decode("latin-1") for i in range(256)]
        return data[:90_000_000], data[90_000_000:95_000_000], chars
    text = open(os.path.join(HERE, "data", "tinyshakespeare.txt"), encoding="utf-8").read()
    chars = sorted(set(text))
    data = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], chars


def windows(data, B, T, g):
    ix = torch.randint(len(data) - T - 1, (B,), generator=g)
    x = torch.stack([data[i:i + T] for i in ix]).long()
    y = torch.stack([data[i + 1:i + T + 1] for i in ix]).long()
    return x, y


@torch.no_grad()
def val_bpc(model, val_batches):
    model.eval()
    tot = n = 0
    for x, y in val_batches:
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum")
        tot += loss.item()
        n += y.numel()
    model.train()
    return tot / n / math.log(2)


@torch.no_grad()
def sample(model, chars, dev, prompt="ROMEO:\n", n=400, T=256):
    if len(chars) == 256:
        prompt = "The history of "
    model.eval()
    idx = torch.tensor([[chars.index(c) for c in prompt]], device=dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(n):
        logits = model(idx[:, -T:])[0, -1] / 0.8
        nxt = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=g).to(dev)
        idx = torch.cat([idx, nxt[None]], 1)
    model.train()
    return "".join(chars[i] for i in idx[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixer", required=True)
    p.add_argument("--feat", type=int, default=256)
    p.add_argument("--nap", type=int, default=4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shift_k", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--hot", type=float, default=85.0)
    p.add_argument("--resume_at", type=float, default=70.0)
    p.add_argument("--write", default="add", choices=["add", "decay", "ema"])
    p.add_argument("--read_power", type=int, default=1)
    p.add_argument("--learned_addr", type=int, default=0, help="control: learned cell keys, top-k cells")
    p.add_argument("--init_mem", action="store_true", help="drawers: learned initial cell contents")
    p.add_argument("--out_gate", action="store_true", help="drawers: RMS-norm + gate on the read")
    p.add_argument("--signed", action="store_true", help="drawers: Count-Sketch +-1 signed writes/reads")
    p.add_argument("--keycheck", action="store_true", help="drawers: weight tables by query . cell mean address")
    p.add_argument("--bitflip", type=int, default=0, help="drawers: soft read over cells from flipping b least-certain bits")
    p.add_argument("--topk", type=int, default=0, help="drawers: bounded soft read of the k most likely cells")
    p.add_argument("--anneal_frac", type=float, default=0.0, help="ramp drawer hardness 0->1 over this share of steps")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--corpus", default="shakespeare", choices=["shakespeare", "enwik8"])
    p.add_argument("--name", required=True)
    a = p.parse_args()

    out = os.path.join(HERE, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=2)
    thermal = Thermal(log_path=os.path.join(out, "thermal.jsonl"))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train, val, chars = load_corpus(a.corpus)
    torch.manual_seed(a.seed)
    model = Model(len(chars), d=a.d, layers=a.layers, mixer=a.mixer, feat=a.feat, nap=a.nap,
                  shift_k=a.shift_k, anchor_seed=a.seed + 1, write=a.write, read_power=a.read_power, topk=a.topk, bitflip=a.bitflip, init_mem=a.init_mem, out_gate=a.out_gate, learned_addr=a.learned_addr, signed=a.signed, keycheck=a.keycheck,
                  window=a.window).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    ge = torch.Generator().manual_seed(1234)
    val_batches = [tuple(t.to(dev) for t in windows(val, 32, a.T, ge)) for _ in range(16)]  # 512 fixed windows
    gt = torch.Generator().manual_seed(a.seed)

    f = open(os.path.join(out, "metrics.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["step", "train_bpc", "val_bpc", "cpu_temp", "sys_power", "elapsed_s"])
    t0, paused, best, loss_avg = time.time(), 0.0, float("inf"), None
    for step in range(1, a.steps + 1):
        x, y = (t.to(dev) for t in windows(train, a.bs, a.T, gt))
        if a.anneal_frac > 0:
            set_hardness(model, min(1.0, step / (a.anneal_frac * a.steps)))
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        bpc = loss.item() / math.log(2)
        loss_avg = bpc if loss_avg is None else 0.98 * loss_avg + 0.02 * bpc
        if step % 25 == 0 and thermal.cpu_temp() and thermal.cpu_temp() > a.hot:
            paused += thermal.wait_below(a.resume_at, why=f"{a.name} step {step}")
        if step % a.eval_every == 0 or step == a.steps:
            vb = val_bpc(model, val_batches)
            best = min(best, vb)
            th = thermal.latest
            w.writerow([step, f"{loss_avg:.4f}", f"{vb:.4f}", th["cpu_temp"], th["sys_power"],
                        f"{time.time() - t0:.1f}"])
            f.flush()
            print(f"{a.name} step {step} train {loss_avg:.3f} val {vb:.3f} bpc "
                  f"cpu {th['cpu_temp'] or float('nan'):.1f}C", flush=True)
    f.close()
    open(os.path.join(out, "sample.txt"), "w").write(sample(model, chars, dev, T=a.T))
    summary = {"name": a.name, "final_val_bpc": round(vb, 4), "best_val_bpc": round(best, 4), "params": n_params,
               "train_s": round(time.time() - t0 - paused, 1), "thermal_pause_s": round(paused, 1),
               "max_cpu_temp": round(thermal.max_cpu_temp, 1), "device": str(dev),
               # run_grid's report reads final_acc/best_acc; for LM runs these hold val bpc (lower is better)
               "final_acc": round(vb, 4), "best_acc": round(best, 4)}
    torch.save({"config": vars(a), "model": model.state_dict()}, os.path.join(out, "model.pt"))  # for diagnostics
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary), flush=True)
    thermal.close()


if __name__ == "__main__":
    main()
