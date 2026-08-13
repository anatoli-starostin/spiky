"""Output-ORDERING evaluation of the trained 128/32 RSNN vs the int4 LUT ground truth (held-out 512 obs).
No retrain of a NEW net: the training pipeline is fully deterministic (weight generator manual_seed(0),
obs seeds fixed, no RNG in Adam/calibration), so re-running std=1.0 / 300 steps reproduces the SAME net
reported for task 738d15da. We cache its checkpoint (walker_rsnn_ckpt.pt) so later ordering-type evals
just load it. Then: rank the 6 output dims by the net's hard first-spike time (earliest = highest action,
consistent with the decode a=(C-t)/alpha) and by the LUT action values, and compare.

TIE CONVENTION (stated explicitly):
  * Net first-spike ticks are integers in [0,T]; non-firing dims are assigned t=T (latest = lowest action).
    Ranking is by decoded action value (monotone-decreasing in tick), so equal ticks -> equal action = a tie.
  * For EXACT-ordering and top-1 we argsort by value with a STABLE sort, so tied dims fall in ascending dim
    index. For PAIRWISE / Kendall / Spearman we treat a pair as tied when the two values are exactly equal;
    tied pairs are counted as NON-matching in the plain pairwise-agreement fraction (conservative), while
    Kendall tau-b and Spearman rho use the standard tie corrections. LUT action values are continuous floats
    so LUT ties essentially never occur; the ties that matter are equal net spike ticks.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import json
import numpy as np
import torch
import walker_rsnn_distill as R

EVO = os.path.dirname(os.path.abspath(__file__))       # self-contained: data + checkpoint beside the script
CKPT = f"{EVO}/walker_rsnn_ckpt_T{R.T}_r{R.READOUT_START}.pt"   # per-config so T=32 / T=64 don't collide


def get_model():
    m = R.RSNN(1.0)
    if os.path.exists(CKPT):
        d = torch.load(CKPT)
        m.load_state_dict(d["state"]); m.thr_h = d["thr_h"]; m.thr_o = d["thr_o"]
        print("loaded cached checkpoint (no retrain)")
        return m
    print("no checkpoint -> reproducing the identical trained net (std=1.0, 300 steps, deterministic)...")
    Xtr = R.sample_obs(512, seed=0); Ytr = R.oracle_actions(Xtr)
    m, l0, hist = R.train_run(1.0, 300, Xtr, Ytr, lambda *a: None)
    torch.save({"state": m.state_dict(), "thr_h": m.thr_h, "thr_o": m.thr_o}, CKPT)
    print(f"trained + cached; final train mse {hist[-1][1]:.4f}")
    return m


def avg_ranks(v):                     # average-rank of values (descending = rank 1 highest), tie-averaged
    order = np.argsort(-v, kind="stable")
    ranks = np.empty(len(v)); i = 0
    sv = -np.sort(-v)
    # assign tie-averaged ranks on descending-sorted values
    tmp = np.empty(len(v))
    j = 0
    while j < len(v):
        k = j
        while k + 1 < len(v) and sv[k + 1] == sv[j]:
            k += 1
        r = (j + k) / 2.0 + 1
        for idx in range(j, k + 1):
            tmp[idx] = r
        j = k + 1
    for pos, idx in enumerate(order):
        ranks[idx] = tmp[pos]
    return ranks


def kendall_spearman(x, y):
    n = len(x); C = D = P = Q = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            if dx == 0: P += 1
            if dy == 0: Q += 1
            if dx != 0 and dy != 0:
                if (dx > 0) == (dy > 0): C += 1
                else: D += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - P) * (n0 - Q))
    tau_b = (C - D) / denom if denom > 0 else 0.0
    tau_a = (C - D) / n0
    rx = avg_ranks(x); ry = avg_ranks(y)
    rho = np.corrcoef(rx, ry)[0, 1] if np.std(rx) > 0 and np.std(ry) > 0 else 0.0
    return C, D, P, Q, tau_a, tau_b, rho, n0


def main():
    m = get_model()
    Xval = R.sample_obs(512, seed=1); Yval = R.oracle_actions(Xval)
    ev = R.evaluate(m, Xval, Yval, hard=True)          # hard integer-delay first-spike readout
    netdec = ev["dec"]                                 # (512,6) decoded actions (monotone in first-spike tick)
    lut = Yval                                         # (512,6) LUT ground-truth action means
    M = netdec.shape[0]

    exact = 0; top1_max = 0; top1_min = 0
    agree_pairs = []; taua = []; taub = []; rho = []; tied_pairs = []
    for k in range(M):
        no = np.argsort(-netdec[k], kind="stable")     # net order (highest action first)
        lo = np.argsort(-lut[k], kind="stable")        # LUT order
        exact += int(np.array_equal(no, lo))
        top1_max += int(no[0] == lo[0])                # earliest-spiking / highest-action dim
        top1_min += int(no[-1] == lo[-1])              # latest-spiking / lowest-action dim
        C, Dd, P, Q, ta, tb, rr, n0 = kendall_spearman(netdec[k], lut[k])
        agree_pairs.append(C / n0)                     # fraction of all 15 pairs that are concordant
        taua.append(ta); taub.append(tb); rho.append(rr); tied_pairs.append(P + Q)

    out = []
    def L(s): out.append(s); print(s)
    L("=== RSNN output-ORDERING vs int4 LUT (held-out 512 obs, no retrain — deterministic reproduction) ===")
    L(f"net = 128 exc / 32 inh, init std 1.0; ranking by hard first-spike tick (earliest = highest action)")
    L("")
    L(f"EXACT full-ordering match (all 6 dims argsort identical): {exact}/{M} = {100*exact/M:.1f}%")
    L(f"PAIRWISE ordering agreement (concordant of 15 pairs, mean over obs): {100*np.mean(agree_pairs):.1f}%")
    L(f"   = (Kendall tau + 1)/2 with tau_a: {(np.mean(taua)+1)/2:.3f}")
    L(f"mean Kendall tau_a (all-15 denom): {np.mean(taua):+.3f}   tau_b (tie-corrected): {np.mean(taub):+.3f}")
    L(f"mean Spearman rho: {np.mean(rho):+.3f}")
    L(f"top-1 argMAX match (net earliest-spike dim == LUT argmax): {top1_max}/{M} = {100*top1_max/M:.1f}%")
    L(f"top-1 argMIN match (net latest-spike dim  == LUT argmin): {top1_min}/{M} = {100*top1_min/M:.1f}%")
    L(f"avg tied pairs per obs (equal net ticks or equal LUT vals, of 15): {np.mean(tied_pairs):.2f} "
      f"(LUT ties ~0; these are equal net first-spike ticks)")
    L(f"random-chance baselines: exact=1/720=0.14%, pairwise=50%, top-1=1/6=16.7%")
    open(f"{EVO}/walker_rsnn_ordering.txt", "w").write("\n".join(out))
    json.dump(dict(exact=exact, M=M, top1_max=top1_max, top1_min=top1_min,
                   pairwise=float(np.mean(agree_pairs)), taua=float(np.mean(taua)),
                   taub=float(np.mean(taub)), rho=float(np.mean(rho)),
                   tied=float(np.mean(tied_pairs)), taub_hist=[float(x) for x in taub]),
              open(f"{EVO}/walker_rsnn_ordering_data.json", "w"))
    L("wrote walker_rsnn_ordering.txt + walker_rsnn_ordering_data.json")


if __name__ == "__main__":
    main()
