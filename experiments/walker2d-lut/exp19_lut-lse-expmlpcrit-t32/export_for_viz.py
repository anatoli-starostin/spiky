"""Export an exp19 checkpoint into the walker2d-viz server's artifact format.

The server (landing/walker2d-viz/server/) is deliberately TORCH-FREE — every actor is pure
numpy (see ADDING_MODELS.md §2). So this script:

  1. loads the torch checkpoint written by ppo.py --save-model,
  2. writes `server/models/<stem>.npz`: the actor LUT weights, its fixed anchor pairs, the
     learned tau, and the observation-normalisation statistics the policy needs,
  3. RE-IMPLEMENTS the forward in numpy and checks it against the real torch module on
     random and on real observations — the export is refused if they disagree.

Only the ACTOR is exported. The critic (including exp19's exponential value head) plays no
part at inference: the demo needs actions, not value estimates.

Usage:
    python export_for_viz.py --ckpt rerun_ckpt/actor_s1.pt --out deploy/
"""
import argparse
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def numpy_actor_forward(obs, P):
    """Pure-numpy reference implementation — the exact logic shipped in the server actor.

    obs : (17,) or (B,17) raw Walker2d-v5 observation
    P   : dict of arrays as stored in the npz
    """
    x = np.atleast_2d(np.asarray(obs, np.float64))
    x = (x - P["obs_mean"]) / np.sqrt(P["obs_var"] + 1e-8)      # training-time normalisation
    a_idx, b_idx = P["anchor_a"], P["anchor_b"]                 # (T, NAP) int
    d = x[:, a_idx] - x[:, b_idx]                               # (B, T, NAP)
    nap = a_idx.shape[1]
    pow2 = (1 << np.arange(nap - 1, -1, -1))                    # MSB-first, as in the module
    idx = ((d > 0).astype(np.int64) * pow2).sum(-1)             # (B, T) row per table
    W = P["weights"]                                            # (T, K, 6)
    T = W.shape[0]
    sel = W[np.arange(T)[None, :], idx]                         # (B, T, 6) selected rows
    tau = float(P["tau_actor"])
    # Sum-scaled log-sum-exp over tables:  T * tau * log( (1/T) sum_t exp(w_t / tau) )
    # computed stably by subtracting the max (equivalent, avoids overflow).
    z = sel / tau
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0, :] + np.log(np.exp(z - m).sum(axis=1))
    out = T * tau * (lse - np.log(T))                           # (B, 6) action means
    # Training clipped actions with env `action.clamp(-1, 1)` — NOT tanh. Match it.
    return np.clip(out, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a .pt from ppo.py --save-model")
    ap.add_argument("--out", default="deploy", help="output directory")
    ap.add_argument("--stem", default="walker2d_fastlut_lse_exp19",
                    help="basename for the .npz / .json artifacts")
    a = ap.parse_args()

    ckpt_path = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(HERE, a.ckpt)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(out_dir, exist_ok=True)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    print(f"loaded {ckpt_path}")
    print(f"  arch {ck['arch']}  seed {ck['seed']}  final_ep_ret {ck['final_ep_ret']:.1f}")

    tau_raw = float(sd["actor_lut.exp_outputs_tau_raw"])
    tau = max(float(softplus(np.array(tau_raw))), 1e-3)          # module's softplus + floor
    P = dict(
        weights=sd["actor_lut.weights"].numpy().astype(np.float32),
        anchor_a=sd["actor_lut.soft_anchor_a_long"].numpy().astype(np.int64),
        anchor_b=sd["actor_lut.soft_anchor_b_long"].numpy().astype(np.int64),
        tau_actor=np.float64(tau),
        obs_mean=ck["obs_mean"].numpy().astype(np.float64),
        obs_var=ck["obs_var"].numpy().astype(np.float64),
        log_std=sd["log_std"].numpy().astype(np.float32),        # not used at inference
    )
    print(f"  weights {P['weights'].shape}  anchors {P['anchor_a'].shape}  tau {tau:.6f}")

    # ---- parity gate: numpy vs the real torch module -----------------------
    import sys
    sys.path.insert(0, SRC)
    from models import REGISTRY                                   # noqa: E402
    torch.manual_seed(0)
    mod = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"],
                               tables_per_head=ck["tables_per_head"])
    mod.load_state_dict(sd)
    mod.eval()

    rng = np.random.default_rng(0)
    tests = {
        "random N(0,1) obs": rng.standard_normal((4096, ck["obs_dim"])),
        "random wide obs": rng.standard_normal((2048, ck["obs_dim"])) * 5.0,
        "zeros": np.zeros((1, ck["obs_dim"])),
    }
    worst = 0.0
    for name, raw in tests.items():
        with torch.no_grad():
            xn = (torch.tensor(raw, dtype=torch.float32) - ck["obs_mean"]) / torch.sqrt(
                ck["obs_var"] + 1e-8)
            mean_t, _ = mod(xn)
            act_t = mean_t.clamp(-1, 1).numpy()
        act_n = numpy_actor_forward(raw, P)
        err = float(np.abs(act_t - act_n).max())
        worst = max(worst, err)
        print(f"  parity [{name:<18}] max|torch - numpy| = {err:.3e}")
    if worst > 1e-4:
        raise SystemExit(f"PARITY FAILED: worst {worst:.3e} > 1e-4 — refusing to export")
    print(f"  PARITY OK (worst {worst:.3e})")

    npz_path = os.path.join(out_dir, f"{a.stem}.npz")
    np.savez_compressed(npz_path, **P)
    meta = dict(
        source_experiment="exp19_lut-lse-expmlpcrit-t32",
        arch=ck["arch"], seed=int(ck["seed"]),
        final_ep_ret=float(ck["final_ep_ret"]),
        obs_dim=int(ck["obs_dim"]), act_dim=int(ck["act_dim"]),
        tables_per_head=int(ck["tables_per_head"]), n_anchor_pairs=int(P["anchor_a"].shape[1]),
        tau_actor=float(tau),
        readout="T * tau * log((1/T) * sum_t exp(w_t / tau))  over T=32 tables",
        obs_normalisation="x = (obs - obs_mean) / sqrt(obs_var + 1e-8)  (stats in the npz)",
        action_postprocess="clip(mean, -1, 1)  — training used env action.clamp(-1,1), NOT tanh",
        numpy_torch_parity_max_abs=float(worst),
        note="Actor only. The critic (exp19's exponential value head) is not needed at inference.",
    )
    json.dump(meta, open(os.path.join(out_dir, f"{a.stem}_meta.json"), "w"), indent=2)
    print(f"\nwrote {npz_path} ({os.path.getsize(npz_path):,} bytes)")
    print(f"wrote {os.path.join(out_dir, a.stem + '_meta.json')}")


if __name__ == "__main__":
    main()
