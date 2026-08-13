"""Overnight option-matrix sweep for the LUT -> spiking distillation.

Matrix:  neuron {exact, lif} x variant {D, W} x scope {races, tau, weights}
                             x target {action, prelog}
plus a RECOVERY/TRAINABILITY probe on the exact neuron (perturb the race init away from the
analytically-exact solution and see whether training walks back to it).

Everything is seeded and every config is written into the result JSON.

  python sweep.py --steps 3000 --batch 256          # full matrix
  python sweep.py --only lif --steps 6000           # the cells that actually learn
  python sweep.py --probe-only
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from student import SpikingStudent

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "distill_exp19_100k.npz")
OUT = os.path.join(HERE, "results")

# pairs-seen milestones for the sample-efficiency curve
MILESTONES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
              200_000, 500_000, 1_000_000, 2_000_000]


def load(device, dtype=torch.float32, val_frac=0.2, seed=0):
    Z = np.load(NPZ)
    n = Z["x_norm"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    X = torch.tensor(Z["x_norm"], dtype=dtype, device=device)
    A = torch.tensor(Z["y_action_mean"], dtype=dtype, device=device)
    S = torch.tensor(Z["y_prelog"], dtype=dtype, device=device)
    meta = dict(weights=torch.tensor(Z["weights"], device=device),
                anchor_a=torch.tensor(Z["anchor_a"], device=device).long(),
                anchor_b=torch.tensor(Z["anchor_b"], device=device).long(),
                tau=float(Z["tau"]))
    return (X[tr.to(device)], A[tr.to(device)], S[tr.to(device)],
            X[va.to(device)], A[va.to(device)], S[va.to(device)], meta,
            float(A.std()))


@torch.no_grad()
def evaluate(st, X, A, chunk=4096):
    mx, tot, cnt, ns = 0.0, 0.0, 0, 0.0
    for i in range(0, X.shape[0], chunk):
        a, t, nospike, _ = st(X[i:i + chunk], hard=True)
        e = (a - A[i:i + chunk]).abs()
        mx = max(mx, float(e.max()))
        tot += float(e.sum())
        cnt += e.numel()
        if nospike is not None:
            ns += float(nospike.float().sum())
    return dict(max=mx, mean=tot / cnt, nospike=ns / cnt)


def run_one(cfg, data, args):
    Xtr, Atr, Str, Xva, Ava, Sva, meta, astd = data
    torch.manual_seed(cfg["seed"])
    st = SpikingStudent(meta["weights"], meta["anchor_a"], meta["anchor_b"], meta["tau"],
                        neuron=cfg["neuron"], variant=cfg["variant"], scope=cfg["scope"],
                        decode_mode=cfg.get("decode", "affine"),
                        bit_eps=args.bit_eps).to(Xtr.device)
    st.calibrate(Xtr[:4096], Atr[:4096])
    if cfg.get("perturb", 0.0) > 0:
        with torch.no_grad():
            st.race.A.add_(torch.randn_like(st.race.A) * cfg["perturb"])
        st.calibrate(Xtr[:4096], Atr[:4096])      # re-fit decode at the perturbed point

    base_tr, base_va = evaluate(st, Xtr, Atr), evaluate(st, Xva, Ava)
    params = [p for _, p in st.trainable()]
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    n = Xtr.shape[0]
    curve, seen, t0, diverged = [], 0, time.time(), False
    milestone_i = 0
    logS_tr = torch.log(Str)
    # The surrogate (straight-through) gradient does NOT vanish at the analytically exact
    # solution, so an optimiser will happily walk away from a perfect init. Best-val
    # early stopping is therefore not a nicety here — without it every exact-neuron cell
    # reports the drift instead of the result. Both numbers are kept.
    # step 0 counts: the untrained point is a legitimate candidate, and for several cells
    # it is in fact the best one. Omitting it would manufacture a fake "improvement".
    best = dict(val=base_va["mean"], max=base_va["max"], pairs=0,
                state={k: v.detach().clone() for k, v in st.state_dict().items()})
    step = -1
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), device=Xtr.device)
        g = st.gates(Xtr[idx], hard=True)
        if cfg["target"] == "prelog":
            loss = ((torch.log(st.sum_S(g).clamp_min(1e-20)) - logS_tr[idx]) ** 2).mean()
        else:
            t_f, _ = st.spike_time(g)
            loss = ((st.decode(t_f) - Atr[idx]) ** 2).mean()
        if not math.isfinite(float(loss.detach())):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        seen += args.batch
        if (step + 1) % args.eval_every == 0 or milestone_i < len(MILESTONES) and \
                seen >= MILESTONES[milestone_i]:
            m = evaluate(st, Xva, Ava)
            if m["mean"] < best["val"]:
                best = dict(val=m["mean"], max=m["max"], pairs=seen,
                            state={k: v.detach().clone() for k, v in st.state_dict().items()})
            while milestone_i < len(MILESTONES) and seen >= MILESTONES[milestone_i]:
                curve.append(dict(pairs=MILESTONES[milestone_i],
                                  val_mean=m["mean"], val_max=m["max"],
                                  best_so_far=best["val"]))
                milestone_i += 1

    fin_tr, fin_va = evaluate(st, Xtr, Atr), evaluate(st, Xva, Ava)
    # restore the best-val checkpoint and measure it on train and val
    if best["state"] is not None:
        st.load_state_dict(best["state"])
    best_tr, best_va = evaluate(st, Xtr, Atr), evaluate(st, Xva, Ava)
    return dict(cfg=cfg, n_trainable=n_par, action_std=astd, wall_s=round(time.time() - t0, 1),
                diverged=diverged, steps_run=step + 1, pairs_seen=seen,
                delay_clamped=st.n_delay_clamped, theta=float(st.theta),
                tau_final=float(st.tau), tau_init=meta["tau"],
                base_train=base_tr, base_val=base_va,
                final_train=fin_tr, final_val=fin_va,
                best_train=best_tr, best_val=best_va, best_pairs=best["pairs"],
                base_val_norm=base_va["mean"] / astd,
                final_val_norm=fin_va["mean"] / astd,
                best_val_norm=best_va["mean"] / astd,
                curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--bit-eps", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--only", default="all", choices=["all", "exact", "lif", "probe"])
    ap.add_argument("--exact-steps", type=int, default=2000,
                    help="exact-neuron cells are VERIFICATION; they need no long run")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(dev)
    os.makedirs(OUT, exist_ok=True)
    print(f"device {dev}  train {tuple(data[0].shape)}  val {tuple(data[3].shape)}  "
          f"action std {data[7]:.4f}")

    jobs = []
    if a.only in ("all", "exact", "lif"):
        neurons = ["exact", "lif"] if a.only == "all" else [a.only]
        for neuron in neurons:
            decodes = ("affine", "corrected") if neuron == "lif" else ("affine",)
            for variant in ("D", "W"):
                for scope in ("races", "tau", "weights"):
                    for target in ("action", "prelog"):
                        for decode in decodes:
                            for lr in a.lrs:
                                for seed in a.seeds:
                                    jobs.append(dict(neuron=neuron, variant=variant,
                                                     scope=scope, target=target,
                                                     decode=decode, lr=lr,
                                                     seed=seed, perturb=0.0))
    if a.only in ("all", "probe"):
        for sigma in (0.05, 0.2, 0.5):
            for variant in ("D", "W"):
                for lr in a.lrs:
                    for seed in a.seeds:
                        jobs.append(dict(neuron="exact", variant=variant, scope="races",
                                         target="action", lr=lr, seed=seed, perturb=sigma))

    results = []
    for i, cfg in enumerate(jobs, 1):
        steps_saved = a.steps
        if cfg["neuron"] == "exact" and cfg["perturb"] == 0.0:
            a.steps = a.exact_steps                 # verification cell, short by design
        tag = (f"{cfg['neuron']}/{cfg['variant']}/{cfg['scope']}/{cfg['target']}"
               f"/{cfg.get('decode', 'affine')}"
               + (f"/perturb{cfg['perturb']}" if cfg["perturb"] else "")
               + f"  lr={cfg['lr']:g} seed={cfg['seed']}")
        print(f"[{i}/{len(jobs)}] {tag}", flush=True)
        r = run_one(cfg, data, a)
        a.steps = steps_saved
        results.append(r)
        print(f"    p={r['n_trainable']:,}  base {r['base_val']['mean']:.3e} "
              f"-> best {r['best_val']['mean']:.3e} (norm {r['best_val_norm']:.4f}, "
              f"@{r['best_pairs']:,} pairs)  final {r['final_val']['mean']:.3e}  "
              f"{r['wall_s']}s" + ("  DIVERGED" if r["diverged"] else ""), flush=True)
        json.dump(results, open(os.path.join(OUT, f"sweep_{a.tag}.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, f'sweep_{a.tag}.json')}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
