"""Does forward_group_size=2 cause the build hang? Test on the DETERMINISTIC reproducer.

The round-6 K=128 checkpoint hangs build_pool every time (3/3 supervisor resumes), so we can
compare group sizes on identical genomes without fighting a 10%-of-the-time flake.

Both while(true) loops in connections_manager_kernels_logic.proto walk the block chain with
the sole exit `if(header.shift_to_next_group == 0) break;` and no cycle guard. A smaller
forward group size packs fewer targets per block, so a source needs proportionally MORE
chained blocks (~48 per source at gs=2 vs ~12 at gs=8 for our ~95-synapse fanout) — more
links, more chances for a malformed one. That is the hypothesis under test.

    python probe_gs_hang.py --gs 2 --tries 3
"""
import argparse
import subprocess
import sys

import numpy as np

FIELDS = ("src_pool", "src_idx", "tgt_pool", "tgt_idx", "delay", "weight")
REPRO = "results/hang_repro_round6_k128.npz"


def load_genomes(path):
    z = np.load(path, allow_pickle=False)
    K = int(z["n_genomes"][0])
    return [{f: z[f"g{i}_{f}"] for f in FIELDS} for i in range(K)]


def child(gs, path, stdp_lr, w_max=30.0):
    import torch
    import steady_state as S
    S.GROUP_SIZE = gs                      # engine synapse_group_size in build_pool
    orig = S.stage2_metas
    # stage2_metas' group_size default was bound at def time, so override explicitly.
    S.stage2_metas = lambda lr, wm, group_size=gs, backward_group_size=32: orig(
        lr, wm, group_size=group_size, backward_group_size=backward_group_size)
    genomes = load_genomes(path)
    h = S.build_pool(genomes, "cuda", seed=1, stdp_lr=stdp_lr, w_max=w_max)
    torch.cuda.synchronize()
    print(f"BUILD-OK {h['n_syn']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs", type=int, default=None)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--path", default=REPRO)
    ap.add_argument("--stdp-lr", type=float, default=0.01)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.gs, a.path, a.stdp_lr)
        sys.exit(0)

    for gs in ([a.gs] if a.gs else [2, 4, 8, 16, 32]):
        tal = {"ok": 0, "err": 0, "hang": 0}
        detail = ""
        for t in range(a.tries):
            try:
                r = subprocess.run(
                    [sys.executable, __file__, "--child", "--gs", str(gs),
                     "--path", a.path, "--stdp-lr", str(a.stdp_lr)],
                    capture_output=True, text=True, timeout=a.timeout)
                txt = r.stdout + r.stderr
                if "BUILD-OK" in txt:
                    tal["ok"] += 1
                else:
                    tal["err"] += 1
                    e = [l.strip() for l in txt.splitlines() if "Error" in l]
                    if e and not detail:
                        detail = e[-1].split("error")[-1].strip()[:58]
            except subprocess.TimeoutExpired:
                tal["hang"] += 1
        print(f"  forward_group_size {gs:2d}: ok {tal['ok']}/{a.tries}  "
              f"error {tal['err']}/{a.tries}  HANG {tal['hang']}/{a.tries}"
              f"{'   ' + detail if detail else ''}", flush=True)
