"""Ablation: T=64 rollout with a dedicated SETTLING phase [0,32) + SECOND-HALF readout [32,64].
Everything else identical to the committed T=32 net (128 exc / 32 inh, Dale's law, learnable delays,
inhibitory fixed delay 1, single-tau LIF soft reset, arctan surrogate, same loss, same optimizer/steps,
init std 1.0, same seeds). Output first-spike is gated to [32,64], decode maps the action range into that
window at the SAME per-tick resolution (alpha_out=3). Head-to-head vs the stored T=32 baseline.
No env / mujoco. Run:  RSNN_T=64 RSNN_READOUT_START=32 python walker_rsnn_ablation_t64.py
"""
import os
os.environ["RSNN_T"] = "64"; os.environ["RSNN_READOUT_START"] = "32"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import json
import numpy as np
import torch
import walker_rsnn_distill as R
from walker_rsnn_ordering import kendall_spearman

EVO = os.path.dirname(os.path.abspath(__file__))
BASE_MED_PCT = 23.04          # T=32 held-out median |err| (% of range) from the committed baseline report


def ordering_metrics(netdec, lut):
    M = netdec.shape[0]; exact = t1max = t1min = 0
    agree = []; taua = []; taub = []; rho = []; tied = []
    for k in range(M):
        no = np.argsort(-netdec[k], kind="stable"); lo = np.argsort(-lut[k], kind="stable")
        exact += int(np.array_equal(no, lo)); t1max += int(no[0] == lo[0]); t1min += int(no[-1] == lo[-1])
        C, Dd, P, Q, ta, tb, rr, n0 = kendall_spearman(netdec[k], lut[k])
        agree.append(C / n0); taua.append(ta); taub.append(tb); rho.append(rr); tied.append(P + Q)
    return dict(exact=exact, M=M, top1max=t1max, top1min=t1min, pairwise=float(np.mean(agree)),
                taua=float(np.mean(taua)), taub=float(np.mean(taub)), rho=float(np.mean(rho)),
                tied=float(np.mean(tied)))


def main():
    assert R.T == 64 and R.READOUT_START == 32, (R.T, R.READOUT_START)
    print(f"CONFIG: T={R.T} readout=[{R.READOUT_START},{R.T}] c_out={R.C_OUT} alpha_out={R.ALPHA_OUT} "
          f"one-tick={1/R.ALPHA_OUT:.3f}")
    Xtr = R.sample_obs(512, 0); Ytr = R.oracle_actions(Xtr)
    Xval = R.sample_obs(512, 1); Yval = R.oracle_actions(Xval)
    model, l0, hist = R.train_run(1.0, 300, Xtr, Ytr, print)   # deterministic, same as baseline recipe
    torch.save({"state": model.state_dict(), "thr_h": model.thr_h, "thr_o": model.thr_o},
               f"{EVO}/walker_rsnn_ckpt_T64_r32.pt")
    ev = R.evaluate(model, Xval, Yval, hard=True)              # hard integer-delay first-spike readout
    a = ev["a"]; dec = ev["dec"]
    nonfire_readout = int((np.abs(dec - (R.C_OUT - R.T) / R.ALPHA_OUT) < 1e-6).sum())
    om = ordering_metrics(dec, Yval)

    # stored T=32 baseline (from the committed run)
    b_val = json.load(open(f"{EVO}/walker_rsnn_result.json"))
    b_ord = json.load(open(f"{EVO}/walker_rsnn_ordering_data.json"))

    def pct(x): return 100 * x
    out = []
    def L(s): out.append(s); print(s)
    L("=== ABLATION: T=64 settling[0,32)+readout[32,64]  vs  BASELINE T=32 ===")
    L("metric                          |     T=32 (baseline) |   T=64 second-half")
    L("loss (task MSE) start->end      | %8.3f->%6.3f     | %8.3f->%6.3f" %
      (b_val["l0"], b_val["lend"], l0, hist[-1][1]))
    L("held-out |err| mean (%% range)   | %13.2f%%      | %13.2f%%" %
      (b_val["val_pct"], pct(ev["mean"] / ev["arange"])))
    L("held-out |err| median (%% range) | %13.2f%%      | %13.2f%%" %
      (BASE_MED_PCT, pct(ev["med"] / ev["arange"])))
    L("within one output tick          | %13.2f%%      | %13.2f%%" % (pct(b_val["w1"]), pct(ev["w1"])))
    L("within two output ticks         | %13.2f%%      | %13.2f%%" % (pct(b_val["w2"]), pct(ev["w2"])))
    L("O non-firing (readout window)   | %13d       | %13d" % (b_val["o_nonfire"], nonfire_readout))
    L("rates H_ex / H_inh / O          | %.3f/%.3f/%.3f     | %.3f/%.3f/%.3f" %
      (b_val["rates"]["hex"], b_val["rates"]["hinh"], b_val["rates"]["o"],
       a["hex_rate"], a["hinh_rate"], a["o_rate"]))
    L("--- ORDERING vs LUT ---")
    L("exact full-argsort match        | %11.1f%%       | %11.1f%%" %
      (pct(b_ord["exact"] / b_ord["M"]), pct(om["exact"] / om["M"])))
    L("pairwise agreement (C/15)       | %11.1f%%       | %11.1f%%" %
      (pct(b_ord["pairwise"]), pct(om["pairwise"])))
    L("Kendall tau_b                   | %11.3f        | %11.3f" % (b_ord["taub"], om["taub"]))
    L("Spearman rho                    | %11.3f        | %11.3f" % (b_ord["rho"], om["rho"]))
    L("top-1 argmax match              | %11.1f%%       | %11.1f%%" %
      (pct(b_ord["top1_max"] / b_ord["M"]), pct(om["top1max"] / om["M"])))
    L("top-1 argmin match              | %11.1f%%       | %11.1f%%" %
      (pct(b_ord["top1_min"] / b_ord["M"]), pct(om["top1min"] / om["M"])))

    d_err = pct(ev["mean"] / ev["arange"]) - b_val["val_pct"]
    d_rho = om["rho"] - b_ord["rho"]; d_t1 = pct(om["top1max"] / om["M"]) - pct(b_ord["top1_max"] / b_ord["M"])
    improved = (d_err < -1.0) or (d_rho > 0.03) or (d_t1 > 3.0)
    verdict = ("IMPROVED" if improved else "roughly FLAT") + \
        f" — err {d_err:+.1f}pp, Spearman {d_rho:+.3f}, top-1 {d_t1:+.1f}pp vs T=32."
    if not improved:
        verdict += " A dedicated settling phase + second-half readout did NOT help → the limiter is " \
                   "capacity/optimization (surrogate credit assignment, hidden size, steps), not settling time."
    else:
        verdict += " Settling time was a real limiter."
    L("VERDICT: " + verdict)
    open(f"{EVO}/walker_rsnn_ablation_t64.txt", "w").write("\n".join(out))
    json.dump(dict(t64_val_pct=pct(ev["mean"] / ev["arange"]), t64_med_pct=pct(ev["med"] / ev["arange"]),
                   t64_w1=ev["w1"], t64_w2=ev["w2"], t64_l0=l0, t64_lend=hist[-1][1],
                   t64_nonfire=nonfire_readout, t64_rates=[a["hex_rate"], a["hinh_rate"], a["o_rate"]],
                   t64_ord=om, base_val=b_val, base_ord=b_ord, improved=bool(improved)),
              open(f"{EVO}/walker_rsnn_ablation_t64.json", "w"), indent=2)
    L("wrote walker_rsnn_ablation_t64.txt + .json")


if __name__ == "__main__":
    main()
