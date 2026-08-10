"""exp009 diagnostic: MSE-trained nets land at the constant-predictor baseline. Why?

Decompose the held-out MSE of the best evolved net into
  * how much rank information it actually carries (Pearson r, Kendall tau)
  * how much of the error is SCALE MISCALIBRATION rather than genuine error, by asking what
    the MSE would be if the predictions were optimally rescaled (a*pred + b, least squares)
If MSE_optimally_rescaled << MSE_raw, the net knows the answer and is expressing it on the
wrong scale; if they are close, it genuinely does not know more than the mean.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/astarostin/projects/spiky/experiments/neurodarwinism/src")
import steady_state as S                                   # noqa: E402
from data import load                                      # noqa: E402

ND = "/home/astarostin/projects/spiky/experiments/neurodarwinism"
S.D_MAX, S.N_DELAY_METAS = 20, 20
S.OUT_GATE, S.OUT_D_MIN, S.OUT_D_MAX, S.N_OUT_DELAY_METAS = True, 64, 80, 17

ARMS = [("exp009 mse-trained", f"{ND}/exp009_mse-objective/mse_seed%d/ck.npz"),
        ("exp008 tau-trained", f"{ND}/exp008_output-delay-gate/gated_seed%d/ck.npz")]

for name, tpl in ARMS:
    print(f"\n=== {name} ===")
    print(f"{'seed':>4s} {'MSE':>8s} {'baseline':>9s} {'MSE rescaled':>13s} {'r':>7s} {'|r|':>7s} "
          f"{'tau':>8s} {'pred sd':>8s} {'tgt sd':>7s} {'slope':>7s}")
    agg = []
    for s in (0, 1, 2):
        ck = tpl % s
        if not os.path.exists(ck):
            continue
        _X, _Y, Xpool, Ypool, Xval, Yval = load(64, s, 2000)
        enc = S.LatencyEncoder(Xpool)
        S.fit_target_stats(Ypool, 2.5, 32)
        genomes, ewma, *_ = S.load_ckpt(ck)
        b = int(np.nanargmax(ewma))
        h = S.build_eval_pool(genomes[b], "cuda", 0.01, 30.0)
        ft, first, _, _ = S.score(h, Xval, Yval, enc, 200.0, 0.0, readout_window=32,
                                  coverage_penalty=0.0, objective="tau")
        pred = first[:, 0, :]                       # [B, 6] offsets
        tgt = S.target_offsets(Yval)                # [B, 6]
        mse = float(np.mean((pred - tgt) ** 2))
        const = float(np.mean((tgt - S.target_offsets(Ypool).mean()) ** 2))
        # least-squares rescale per dimension: a*pred + b
        res = []
        for d in range(pred.shape[1]):
            A = np.stack([pred[:, d], np.ones(pred.shape[0])], 1)
            coef, *_ = np.linalg.lstsq(A, tgt[:, d], rcond=None)
            res.append(np.mean((A @ coef - tgt[:, d]) ** 2))
        mse_rs = float(np.mean(res))
        rs = np.array([np.corrcoef(pred[:, d], tgt[:, d])[0, 1]
                       for d in range(pred.shape[1])])
        r, r_abs = float(rs.mean()), float(np.abs(rs).mean())
        slope = float(np.mean([np.polyfit(pred[:, d], tgt[:, d], 1)[0]
                               for d in range(pred.shape[1])]))
        print(f"{s:4d} {mse:8.3f} {const:9.3f} {mse_rs:13.3f} {r:+7.3f} {r_abs:7.3f} "
              f"{float(ft[0]):+8.4f} {pred.std():8.3f} {tgt.std():7.3f} {slope:+7.3f}")
        print(f"       per-dim r: {np.array2string(rs, precision=3, sign='+')}")
        agg.append((mse, const, mse_rs, r, float(ft[0]), pred.std(), r_abs))
        del h
        torch.cuda.empty_cache()
    if agg:
        a = np.array(agg)
        print(f"mean {a[:, 0].mean():8.3f} {a[:, 1].mean():9.3f} {a[:, 2].mean():13.3f} "
              f"{a[:, 3].mean():+7.3f} {a[:, 4].mean():+8.4f} {a[:, 5].mean():8.3f}")
