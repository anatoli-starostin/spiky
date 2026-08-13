"""exp_c03 1b — distil the PPO teacher into a LUT policy, and evaluate it (#75).

Supervised MSE on the teacher's deterministic actions, then the ONLY number that
counts: a deterministic 100-episode eval in the CPU reference env.

One config per invocation so the sweep can run them independently and a failure in
one cell cannot take the sweep down.

Usage:
  python distill.py --nap 8 --tph 64 --epochs 6
"""
import argparse, json, os, time

import numpy as np
import torch
import torch.nn.functional as F

from lut_policy import LUTPolicy, save
from eval_lut import eval_policy

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nap", type=int, default=8)
    ap.add_argument("--tph", type=int, default=64)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--module", default="hyperplane", choices=["hyperplane", "fast"])
    ap.add_argument("--forward-mode", default="hybrid_smooth")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="use only N pairs (0 = all)")
    ap.add_argument("--tag", default="")
    # exp_c08 reuses this trainer verbatim against the SAC-teacher dataset;
    # only the data/output location changes.
    ap.add_argument("--data-dir", default=HERE)
    ap.add_argument("--out-dir", default=HERE)
    # retention is measured against whichever teacher produced the dataset
    ap.add_argument("--teacher-score", type=float, default=5555.5,
                    help="PPO teacher 5555.5 (default); SAC teacher 5273.4")
    a = ap.parse_args()

    dev = "cuda"
    obs = np.load(os.path.join(a.data_dir, "obs.npy"), mmap_mode="r")
    act = np.load(os.path.join(a.data_dir, "act.npy"), mmap_mode="r")
    if a.limit:
        obs, act = obs[:a.limit], act[:a.limit]
    N = len(obs)
    stats = json.load(open(os.path.join(a.data_dir, "dataset_stats.json")))

    X = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=dev)
    Y = torch.as_tensor(np.asarray(act), dtype=torch.float32, device=dev)

    model = LUTPolicy(n_anchor_pairs=a.nap, tables_per_head=a.tph, n_heads=a.heads,
                      module=a.module, forward_mode=a.forward_mode,
                      obs_mean=stats["obs_mean"], obs_std=stats["obs_std"],
                      device=dev).to(dev)
    name = f"{a.module}_nap{a.nap}_tph{a.tph}_h{a.heads}{a.tag}"
    print(f"[{name}] {model.describe()} | dataset {N:,} pairs", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps_per_epoch = N // a.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(a.epochs * steps_per_epoch, 1))

    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        perm = torch.randperm(N, device=dev)
        tot, nb = 0.0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            pred = model(X[idx])
            loss = F.mse_loss(pred, Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        mse = tot / max(nb, 1)
        hist.append(dict(epoch=ep, action_mse=mse))
        print(f"  epoch {ep+1}/{a.epochs}  action MSE {mse:.5f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # held-out action MSE on the last 200k pairs (never shuffled into training order
    # differently, but a fresh read is still a useful sanity number)
    with torch.no_grad():
        ho = slice(max(N - 200_000, 0), N)
        ho_mse = F.mse_loss(model(X[ho]), Y[ho]).item()

    ckpt = os.path.join(a.out_dir, f"lut_{name}.pt")
    save(model, ckpt)

    mean, std, rets, _ = eval_policy(model, episodes=a.episodes)
    teacher = a.teacher_score
    print(f"[{name}] CPU-reference {a.episodes}-ep eval: {mean:.1f} +/- {std:.1f} "
          f"| retention {100*mean/teacher:.1f}% of teacher | "
          f"{'SOLVED' if mean >= 3000 else 'below 3000'}", flush=True)

    out = dict(name=name, module=a.module, nap=a.nap, tph=a.tph, heads=a.heads,
               rows=2 ** a.nap, forward_mode=a.forward_mode,
               table_params=model.table_params(), index_params=model.index_params(),
               total_params=model.n_params(), dataset_pairs=N, epochs=a.epochs,
               final_action_mse=hist[-1]["action_mse"], heldout_action_mse=ho_mse,
               eval_mean=mean, eval_std=std, teacher_retention_pct=100 * mean / teacher,
               solved=bool(mean >= 3000), train_s=round(time.time() - t0, 1),
               history=hist, checkpoint=os.path.basename(ckpt))
    with open(os.path.join(a.out_dir, f"result_{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", f"result_{name}.json", flush=True)


if __name__ == "__main__":
    main()
