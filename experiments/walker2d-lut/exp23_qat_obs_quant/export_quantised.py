"""Export a QUANTISED (QAT) exp19-family checkpoint into the walker2d-viz artifact format.

A sibling of `../exp19_lut-lse-expmlpcrit-t32/export_for_viz.py`, same contract: torch-free
numpy artifact + a parity gate that refuses to export if numpy and torch disagree. Two
things are added, because this policy was TRAINED with them in the loop and is only correct
when they are applied at inference too:

  INPUT   the normalised observation is quantised to 128 Gaussian-companded buckets
          (sigma = 1), through ONE shared monotone map over all 17 coordinates.
  OUTPUT  the action mean is clipped to [-1,1] and snapped to a 22-level uniform grid.

THE SERVER HAS NO SCIPY, so neither erf nor erfinv is available there. Both maps are
therefore baked into the npz as plain arrays:
  `in_quant_edges`    (127,) the bucket boundaries -> tick = searchsorted(edges, x)
  `in_quant_dequant`  (128,) the value each tick decodes to
which is exactly how the shipped spiking actor handles its own encoder table.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
DEPS = os.path.join(HERE, "..", "_qat_deps")


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def phi_inv(p):
    from scipy.special import erfinv          # export-time only; never at inference
    return math.sqrt(2.0) * erfinv(2.0 * np.asarray(p, np.float64) - 1.0)


def build_in_quant(n_ticks=128, sigma=1.0):
    """(edges, dequant) reproducing obs_quant.GaussianCompandingQuantizer exactly.

    tick = round((n-1) * Phi(x/sigma)) changes value at (n-1)*Phi(x/sigma) = k + 0.5, i.e.
    at x = sigma * Phi^-1((k+0.5)/(n-1)) for k = 0..n-2 -- those are the 127 edges.
    Bucket centres are Phi^-1(k/(n-1)), with the two unbounded end buckets taking the
    midpoint of their Phi-interval (Phi^-1(0) / Phi^-1(1) would be infinite).
    """
    span = n_ticks - 1
    edges = sigma * phi_inv((np.arange(span) + 0.5) / span)
    p = np.arange(n_ticks, dtype=np.float64) / span
    p[0] = 0.25 / span
    p[-1] = 1.0 - 0.25 / span
    return edges, sigma * phi_inv(p)


def numpy_actor_forward(obs, P):
    """Pure-numpy reference — the exact logic shipped in the quantised server actor."""
    x = np.atleast_2d(np.asarray(obs, np.float64))
    x = (x - P["obs_mean"]) / np.sqrt(P["obs_var"] + 1e-8)
    # --- INPUT quantiser: shared 128-bucket Gaussian companding -------------------------
    t = np.searchsorted(P["in_quant_edges"], x.ravel(), side="left").reshape(x.shape)
    x = P["in_quant_dequant"][np.clip(t, 0, P["in_quant_dequant"].shape[0] - 1)]
    # --- the LUT itself (unchanged from the un-quantised actor) --------------------------
    a_idx, b_idx = P["anchor_a"], P["anchor_b"]
    d = x[:, a_idx] - x[:, b_idx]
    nap = a_idx.shape[1]
    pow2 = (1 << np.arange(nap - 1, -1, -1))
    idx = ((d > 0).astype(np.int64) * pow2).sum(-1)
    W = P["weights"]
    T = W.shape[0]
    sel = W[np.arange(T)[None, :], idx]
    tau = float(P["tau_actor"])
    z = sel / tau
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0, :] + np.log(np.exp(z - m).sum(axis=1))
    out = T * tau * (lse - np.log(T))
    # --- OUTPUT quantiser: clip then snap to the uniform grid ---------------------------
    clip = float(P["out_quant_clip"])
    n = int(P["out_quant_levels"])
    step = 2.0 * clip / (n - 1)
    c = np.clip(out, -clip, clip)
    return np.clip(np.round((c + clip) / step) * step - clip, -clip, clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default="walker2d_fastlut_lse_exp19_quantised")
    ap.add_argument("--in-quant-ticks", type=int, default=128)
    ap.add_argument("--in-quant-sigma", type=float, default=1.0)
    ap.add_argument("--out-quant-levels", type=int, default=22)
    ap.add_argument("--out-quant-clip", type=float, default=1.0)
    ap.add_argument("--oob-penalty", type=float, default=0.3)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    print(f"loaded {a.ckpt}")
    print(f"  arch {ck['arch']}  seed {ck['seed']}  final_ep_ret {ck['final_ep_ret']:.1f}")

    tau = max(float(softplus(np.array(float(sd["actor_lut.exp_outputs_tau_raw"])))), 1e-3)
    edges, dequant = build_in_quant(a.in_quant_ticks, a.in_quant_sigma)
    P = dict(
        weights=sd["actor_lut.weights"].numpy().astype(np.float32),
        anchor_a=sd["actor_lut.soft_anchor_a_long"].numpy().astype(np.int64),
        anchor_b=sd["actor_lut.soft_anchor_b_long"].numpy().astype(np.int64),
        tau_actor=np.float64(tau),
        obs_mean=ck["obs_mean"].numpy().astype(np.float64),
        obs_var=ck["obs_var"].numpy().astype(np.float64),
        log_std=sd["log_std"].numpy().astype(np.float32),
        in_quant_edges=edges.astype(np.float64),
        in_quant_dequant=dequant.astype(np.float64),
        in_quant_ticks=np.int64(a.in_quant_ticks),
        in_quant_sigma=np.float64(a.in_quant_sigma),
        out_quant_levels=np.int64(a.out_quant_levels),
        out_quant_clip=np.float64(a.out_quant_clip),
    )
    print(f"  weights {P['weights'].shape}  anchors {P['anchor_a'].shape}  tau {tau:.6f}")
    print(f"  input  quantiser: {a.in_quant_ticks} buckets, sigma {a.in_quant_sigma}, "
          f"range [{dequant[0]:.4f}, {dequant[-1]:.4f}]")
    print(f"  output quantiser: {a.out_quant_levels} levels on "
          f"[-{a.out_quant_clip:g}, {a.out_quant_clip:g}], "
          f"step {2*a.out_quant_clip/(a.out_quant_levels-1):.6f}")

    # ---- parity gate: numpy vs the real torch module + the training-time quantisers ----
    sys.path.insert(0, DEPS)
    sys.path.insert(0, SRC)
    from models import REGISTRY                       # noqa: E402
    from obs_quant import GaussianCompandingQuantizer  # noqa: E402
    from act_quant import UniformActionQuantizer      # noqa: E402
    torch.manual_seed(0)
    mod = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"],
                               tables_per_head=ck["tables_per_head"])
    mod.load_state_dict(sd)
    mod.eval()
    # straight_through=False on purpose. The TRAINING forward is `x + (xq - x).detach()`,
    # which is NOT value-exact in float32: for a large |x| the round-trip returns xq plus a
    # relative-epsilon perturbation, so two coordinates sharing a bucket come out UNEQUAL and
    # the LUT's `d > 0` tie is broken by float noise instead of deterministically. A shipped
    # artifact must be reproducible, so it dequantises exactly (equal ticks -> equal values ->
    # tie -> bit 0), which is also what the spiking encoder does by construction.
    # Measured cost of that choice on 100k REAL observations: 0.0018% of (sample, table)
    # address rows differ, 0.0400% of samples have any differing table. The discrepancy is
    # concentrated entirely in the saturated end buckets (1.32% of scalars).
    iq = GaussianCompandingQuantizer(a.in_quant_ticks, a.in_quant_sigma,
                                     straight_through=False)
    oq = UniformActionQuantizer(a.out_quant_levels, a.out_quant_clip,
                                straight_through=False)

    rng = np.random.default_rng(0)
    tests = {
        "random N(0,1) obs": rng.standard_normal((4096, ck["obs_dim"])),
        "random wide obs": rng.standard_normal((2048, ck["obs_dim"])) * 5.0,
        "zeros": np.zeros((1, ck["obs_dim"])),
    }
    worst, worst_tick = 0.0, 0
    for name, raw in tests.items():
        with torch.no_grad():
            xn = (torch.tensor(raw, dtype=torch.float32) - ck["obs_mean"]) / torch.sqrt(
                ck["obs_var"] + 1e-8)
            mean_t, _ = mod(iq(xn))
            act_t = oq(mean_t).numpy()
            tick_t = iq.ticks(xn).numpy()
        act_n = numpy_actor_forward(raw, P)
        xnn = (raw - P["obs_mean"]) / np.sqrt(P["obs_var"] + 1e-8)
        tick_n = np.searchsorted(P["in_quant_edges"], xnn.ravel(),
                                 side="left").reshape(xnn.shape)
        err = float(np.abs(act_t - act_n).max())
        dtick = int(np.abs(tick_t - tick_n).max())
        worst = max(worst, err); worst_tick = max(worst_tick, dtick)
        print(f"  parity [{name:<18}] max|torch - numpy| = {err:.3e}  "
              f"max tick disagreement = {dtick}")
    if worst > 1e-4:
        raise SystemExit(f"PARITY FAILED: worst {worst:.3e} > 1e-4 — refusing to export")
    if worst_tick != 0:
        raise SystemExit(f"INPUT QUANTISER MISMATCH: ticks differ by {worst_tick} — "
                         "the searchsorted edges do not reproduce the training map")
    print(f"  PARITY OK (worst {worst:.3e}, input ticks bit-identical)")

    npz_path = os.path.join(a.out, f"{a.stem}.npz")
    np.savez_compressed(npz_path, **P)
    meta = dict(
        source_experiment="exp23_qat_obs_quant / qat_n22_l2 (QAT fine-tune of "
                          "exp19_lut-lse-expmlpcrit-t32)",
        parent_checkpoint="exp19_lut-lse-expmlpcrit-t32/deploy_matched/actor_s2.pt "
                          "(final_ep_ret 5966.3)",
        arch=ck["arch"], seed=int(ck["seed"]),
        final_ep_ret=float(ck["final_ep_ret"]),
        obs_dim=int(ck["obs_dim"]), act_dim=int(ck["act_dim"]),
        tables_per_head=int(ck["tables_per_head"]),
        n_anchor_pairs=int(P["anchor_a"].shape[1]),
        tau_actor=float(tau),
        readout="T * tau * log((1/T) * sum_t exp(w_t / tau))  over T=32 tables",
        obs_normalisation="x = (obs - obs_mean) / sqrt(obs_var + 1e-8)  (stats in the npz)",
        input_quantisation=dict(
            kind="Gaussian companding, ONE shared monotone map over all 17 coords",
            ticks=int(a.in_quant_ticks), sigma=float(a.in_quant_sigma),
            forward="tick = round(127 * Phi(x / sigma)); x_hat = dequant[tick]",
            deployed_as="searchsorted(in_quant_edges, x) -> in_quant_dequant[tick] "
                        "(no scipy needed; verified bit-identical to the training map)",
            why_shared="the LUT addresses by comparisons BETWEEN coordinates "
                       "(bit = 1[x[a] > x[b]]), so a per-coordinate map would change the "
                       "meaning of every address bit spanning two maps",
        ),
        output_quantisation=dict(
            kind="uniform, models the spiking Stage-3 first-spike readout",
            levels=int(a.out_quant_levels), clip=float(a.out_quant_clip),
            step=float(2 * a.out_quant_clip / (a.out_quant_levels - 1)),
            forward="clip(mean, -1, 1) then snap to linspace(-1, 1, 22)",
            note="the EMITTED action is strictly within [-1, 1] and always exactly on one "
                 "of the 22 grid points; both rails are exactly representable",
        ),
        oob_penalty=float(a.oob_penalty),
        oob_penalty_form="loss += w * mean_batch( sum_o relu(|mu_raw[o]| - 1)^2 )",
        oob_penalty_why="the clip's gradient is exactly zero outside [-1,1] and an "
                        "out-of-band action is free in both physics and reward, so nothing "
                        "else pulls the RAW readout in-band; without this term the LUT "
                        "weights widen and the spiking delay span grows",
        action_postprocess="clip(mean, -1, 1) then 22-level uniform quantisation",
        numpy_torch_parity_max_abs=float(worst),
        note="Actor only. The critic is not needed at inference. This artifact is only "
             "correct WITH both quantisers applied — the policy was trained with them in "
             "the loop.",
    )
    json.dump(meta, open(os.path.join(a.out, f"{a.stem}_meta.json"), "w"), indent=2)
    print(f"\nwrote {npz_path} ({os.path.getsize(npz_path):,} bytes)")
    print(f"wrote {os.path.join(a.out, a.stem + '_meta.json')}")


if __name__ == "__main__":
    main()
