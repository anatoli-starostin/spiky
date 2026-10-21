"""One MQAR run -> runs/<name>/{config.json, metrics.csv, summary.json}.

MQAR sequence (n pairs): k1 v1 k2 v2 ... kn vn | q1 a1 q2 a2 ... qn an
keys in [0, NK), values in [NK, NK+NV); loss/accuracy only on predicting a_i at q_i.
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F

from models import Model, set_hardness
from thermal import Thermal

NK, NV, NF = 256, 256, 16  # keys, values, filler tokens (used only when gap > 0)
HERE = os.path.dirname(os.path.abspath(__file__))


def mqar_batch(B, n, device, g=None, gap=0):
    """gap > 0 inserts `gap` random filler tokens between each key and its value in the context part."""
    keys = torch.rand(B, NK, generator=g).argsort(-1)[:, :n]
    vals = torch.randint(NK, NK + NV, (B, n), generator=g)
    perm = torch.rand(B, n, generator=g).argsort(-1)
    qk, qv = keys.gather(1, perm), vals.gather(1, perm)
    fill = torch.randint(NK + NV, NK + NV + NF, (B, n, gap), generator=g)
    ctx = torch.cat([keys[..., None], fill, vals[..., None]], -1).view(B, n * (2 + gap))
    qry = torch.stack([qk, qv], -1).view(B, 2 * n)
    x = torch.cat([ctx, qry], 1)
    y = torch.full_like(x, -100)
    y[:, n * (2 + gap)::2] = qv  # at query-key position predict its value
    return x.to(device), y.to(device)


def update_batch(B, n, u, device, g=None):
    """n key-value pairs, then u of those keys re-assigned to new values, then all n keys queried.
    The correct answer is always the latest value. Returns x, y, upd (True at queries of updated keys)."""
    keys = torch.rand(B, NK, generator=g).argsort(-1)[:, :n]
    vals = torch.randint(NK, NK + NV, (B, n), generator=g)
    sel = torch.rand(B, n, generator=g).argsort(-1)[:, :u]                 # which pairs get updated
    new = NK + (vals.gather(1, sel) - NK + torch.randint(1, NV, (B, u), generator=g)) % NV  # differs from old
    latest = vals.scatter(1, sel, new)
    is_upd = torch.zeros(B, n, dtype=torch.bool).scatter(1, sel, True)
    perm = torch.rand(B, n, generator=g).argsort(-1)
    ctx = torch.stack([keys, vals], -1).view(B, 2 * n)
    upd = torch.stack([keys.gather(1, sel), new], -1).view(B, 2 * u)
    qry = torch.stack([keys.gather(1, perm), latest.gather(1, perm)], -1).view(B, 2 * n)
    x = torch.cat([ctx, upd, qry], 1)
    y = torch.full_like(x, -100)
    y[:, 2 * (n + u)::2] = latest.gather(1, perm)
    m = torch.zeros_like(x, dtype=torch.bool)
    m[:, 2 * (n + u)::2] = is_upd.gather(1, perm)
    return x.to(device), y.to(device), m.to(device)


@torch.no_grad()
def evaluate(model, data):
    """Returns (overall recall, recall on updated keys or None, recall on static keys or None)."""
    model.eval()
    c = {"all": [0, 0], "upd": [0, 0], "static": [0, 0]}
    for batch in data:
        x, y = batch[0], batch[1]
        pred = model(x).argmax(-1)
        m = y != -100
        groups = {"all": m}
        if len(batch) == 3:
            groups["upd"], groups["static"] = m & batch[2], m & ~batch[2]
        for k, gm in groups.items():
            c[k][0] += (pred[gm] == y[gm]).sum().item()
            c[k][1] += gm.sum().item()
    model.train()
    r = {k: (a / b if b else None) for k, (a, b) in c.items()}
    return r["all"], r["upd"], r["static"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixer", required=True)
    p.add_argument("--n_pairs", type=int, default=16)
    p.add_argument("--feat", type=int, default=256)
    p.add_argument("--nap", type=int, default=6)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--hot", type=float, default=85.0, help="pause training above this cpu temp")
    p.add_argument("--resume_at", type=float, default=70.0)
    p.add_argument("--gap", type=int, default=0, help="filler tokens between key and value")
    p.add_argument("--shift_k", type=int, default=1, help="token shift sees this many previous tokens")
    p.add_argument("--anchor_per_seed", action="store_true", help="draw drawer anchors per run seed")
    p.add_argument("--updates", type=int, default=0, help="keys re-assigned to new values before the queries")
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
    p.add_argument("--window", type=int, default=16, help="local softmax window (mixers local, hybrid)")
    p.add_argument("--ckpt_every", type=int, default=0, help="also save model_<step>.pt every N steps")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    name = a.name or f"{a.mixer}_n{a.n_pairs}_F{a.feat}_s{a.seed}"
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=2)

    thermal = Thermal(log_path=os.path.join(out, "thermal.jsonl"))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed)
    vocab = NK + NV + (NF if a.gap else 0)
    model = Model(vocab, d=a.d, mixer=a.mixer, feat=a.feat, nap=a.nap, shift_k=a.shift_k,
                  anchor_seed=a.seed + 1 if a.anchor_per_seed else 0, write=a.write,
                  read_power=a.read_power, topk=a.topk, bitflip=a.bitflip, init_mem=a.init_mem, out_gate=a.out_gate, learned_addr=a.learned_addr, signed=a.signed, keycheck=a.keycheck, window=a.window).to(dev)
    batch = ((lambda B, g: update_batch(B, a.n_pairs, a.updates, dev, g)) if a.updates else
             (lambda B, g: mqar_batch(B, a.n_pairs, dev, g, a.gap)))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    ge = torch.Generator().manual_seed(1234)
    eval_data = [batch(256, ge) for _ in range(4)]  # fixed 1024 sequences
    gt = torch.Generator().manual_seed(a.seed)

    # resume after a power-off / kill: model, optimiser, schedule, data RNG and counters are saved at every eval
    rpath, start = os.path.join(out, "resume.pt"), 1
    t0, paused, best, loss_avg = time.time(), 0.0, 0.0, None
    if os.path.exists(rpath):
        r = torch.load(rpath, map_location=dev, weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); sched.load_state_dict(r["sched"])
        gt.set_state(r["gt"].cpu()); start = r["step"] + 1
        t0, paused, best, loss_avg = time.time() - r["elapsed"], r["paused"], r["best"], r["loss_avg"]
        lines = open(os.path.join(out, "metrics.csv")).read().splitlines()
        keep = [l for l in lines[1:] if l and int(l.split(",")[0]) <= r["step"]]
        open(os.path.join(out, "metrics.csv"), "w").write("\n".join([lines[0]] + keep) + "\n")
        f = open(os.path.join(out, "metrics.csv"), "a", newline="")
        print(f"{name} resumed at step {start}", flush=True)
    else:
        f = open(os.path.join(out, "metrics.csv"), "w", newline="")
    w = csv.writer(f)
    if start == 1:
        w.writerow(["step", "loss", "eval_acc", "cpu_temp", "sys_power", "elapsed_s", "acc_updated", "acc_static"])
    for step in range(start, a.steps + 1):
        x, y = batch(a.bs, gt)[:2]
        if a.anneal_frac > 0:
            set_hardness(model, min(1.0, step / (a.anneal_frac * a.steps)))
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_avg = loss.item() if loss_avg is None else 0.98 * loss_avg + 0.02 * loss.item()
        if step % 25 == 0 and thermal.cpu_temp() and thermal.cpu_temp() > a.hot:  # checked every 25 steps
            paused += thermal.wait_below(a.resume_at, why=f"{name} step {step}")
        if step % 25 == 0:
            paused += thermal.wait_power(why=f"{name} step {step}")
        if step % a.eval_every == 0 or step == a.steps:
            acc, acc_u, acc_s = evaluate(model, eval_data)
            best = max(best, acc)
            th = thermal.latest
            w.writerow([step, f"{loss_avg:.4f}", f"{acc:.4f}", th["cpu_temp"], th["sys_power"],
                        f"{time.time() - t0:.1f}", "" if acc_u is None else f"{acc_u:.4f}",
                        "" if acc_s is None else f"{acc_s:.4f}"])
            if a.ckpt_every and step % a.ckpt_every == 0:
                torch.save({"config": vars(a), "model": model.state_dict()}, os.path.join(out, f"model_{step}.pt"))
            f.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "gt": gt.get_state(), "step": step, "elapsed": time.time() - t0, "paused": paused,
                        "best": best, "loss_avg": loss_avg}, rpath + ".tmp")
            os.replace(rpath + ".tmp", rpath)
            print(f"{name} step {step} loss {loss_avg:.3f} acc {acc:.3f} "
                  f"cpu {th['cpu_temp'] or float('nan'):.1f}C", flush=True)

    f.close()
    summary = {"name": name, "final_acc": acc, "best_acc": best, "params": n_params,
               "final_acc_updated": acc_u, "final_acc_static": acc_s,
               "train_s": round(time.time() - t0 - paused, 1), "thermal_pause_s": round(paused, 1),
               "max_cpu_temp": round(thermal.max_cpu_temp, 1), "device": str(dev)}
    torch.save({"config": vars(a), "model": model.state_dict()}, os.path.join(out, "model.pt"))  # for diagnostics
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    if os.path.exists(rpath):
        os.remove(rpath)
    print(json.dumps(summary), flush=True)
    thermal.close()


if __name__ == "__main__":
    main()
