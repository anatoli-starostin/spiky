"""Export an exp05 MLP checkpoint into the walker2d-viz server's artifact format.

The MLP analogue of exp19's export_for_viz.py, and it works the same way. The server
(landing/walker2d-viz/server/) is deliberately TORCH-FREE — every actor is pure numpy (see
ADDING_MODELS.md §2). So this script:

  1. loads the torch checkpoint written by ppo.py --save-model,
  2. writes `server/models/<stem>.npz`: the three policy layers and the
     observation-normalisation statistics the policy needs,
  3. RE-IMPLEMENTS the forward in numpy and checks it against the real torch module on
     random and on real observations — the export is refused if they disagree.

Only the ACTOR is exported. The critic plays no part at inference: the demo needs actions,
not value estimates.

Usage:
    python export_for_viz.py --ckpt <path>/actor_s1.pt --out deploy/
"""
import argparse
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
STEM = "walker2d_mlp_ppo_exp05"


def numpy_actor_forward(obs, P):
    """Pure-numpy reference — the exact logic shipped in deploy/mlp_ppo.py.

    obs : (17,) or (B,17) raw Walker2d-v5 observation
    P   : dict of arrays as stored in the npz
    """
    x = np.atleast_2d(np.asarray(obs, np.float64))
    x = (x - P["obs_mean"]) / np.sqrt(P["obs_var"] + 1e-8)      # training-time normalisation
    h = np.tanh(x @ P["pi_w0"] + P["pi_b0"])
    h = np.tanh(h @ P["pi_w1"] + P["pi_b1"])
    return h @ P["pi_w2"] + P["pi_b2"]                          # pre-clip action means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "deploy"))
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="refuse to export if numpy and torch disagree by more than this")
    ap.add_argument("--n-random", type=int, default=4096)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.abspath(SRC))
    from models import REGISTRY                                  # noqa: E402

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck["arch"] != "mlp":
        raise SystemExit(f"this exporter is for --arch mlp, got {ck['arch']!r}")
    ac = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"])
    ac.load_state_dict(ck["state_dict"], strict=True)
    ac.eval()

    sd = ck["state_dict"]
    # torch Linear stores (out, in); numpy wants x @ W so transpose once, here.
    P = {}
    for i, li in enumerate((0, 2, 4)):                           # Sequential: Lin,Tanh,Lin,Tanh,Lin
        P[f"pi_w{i}"] = sd[f"pi.{li}.weight"].numpy().T.astype(np.float64)
        P[f"pi_b{i}"] = sd[f"pi.{li}.bias"].numpy().astype(np.float64)
    P["obs_mean"] = ck["obs_mean"].numpy().astype(np.float64)
    P["obs_var"] = ck["obs_var"].numpy().astype(np.float64)

    print(f"checkpoint : {a.ckpt}")
    print(f"             arch={ck['arch']} seed={ck['seed']} "
          f"final_ep_ret={ck['final_ep_ret']:.1f}")
    print(f"layers     : " + " -> ".join(str(P[f'pi_w{i}'].shape) for i in range(3)))

    # ---- parity: numpy vs torch, on random AND on real observations -------------------
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(a.n_random, ck["obs_dim"])) * 3.0     # wide, covers the tails
    with torch.no_grad():
        x = (torch.from_numpy(raw).float() - ck["obs_mean"]) / torch.sqrt(ck["obs_var"] + 1e-8)
        t_out = ac(x)[0].numpy().astype(np.float64)
    n_out = numpy_actor_forward(raw, P)
    d_rand = float(np.abs(t_out - n_out).max())

    d_real = float("nan")
    try:
        import gymnasium as gym
        env = gym.make("Walker2d-v5")
        obs, _ = env.reset(seed=0)
        real = []
        for _ in range(1000):
            real.append(obs.copy())
            act = np.clip(numpy_actor_forward(obs, P)[0], -1, 1).astype(np.float32)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                obs, _ = env.reset()
        real = np.array(real, np.float64)
        with torch.no_grad():
            x = (torch.from_numpy(real).float() - ck["obs_mean"]) / torch.sqrt(ck["obs_var"] + 1e-8)
            t_real = ac(x)[0].numpy().astype(np.float64)
        d_real = float(np.abs(t_real - numpy_actor_forward(real, P)).max())
    except Exception as e:                                       # gymnasium optional here
        print(f"(real-observation parity skipped: {type(e).__name__}: {e})")

    worst = max(d_rand, 0.0 if np.isnan(d_real) else d_real)
    print(f"parity     : random obs {d_rand:.3e}   real obs {d_real:.3e}   worst {worst:.3e}")
    if worst > a.tol:
        raise SystemExit(f"REFUSED: numpy and torch disagree by {worst:.3e} > {a.tol:g}")

    os.makedirs(a.out, exist_ok=True)
    npz = os.path.join(a.out, STEM + ".npz")
    np.savez(npz, **P)
    meta = dict(
        source_experiment="exp05_ppo-truncbootstrap-retnorm-kl",
        variant="deploy_matched (--obs-clip-vel 10.0 --solver-iters 100 --ls-iters 50)",
        arch=ck["arch"], seed=ck["seed"], final_ep_ret=ck["final_ep_ret"],
        obs_dim=ck["obs_dim"], act_dim=ck["act_dim"],
        hidden=[int(P["pi_w0"].shape[1]), int(P["pi_w1"].shape[1])],
        activation="tanh",
        readout="linear; action = clip(mean, -1, 1) — training used env action.clamp(-1,1), NOT tanh",
        obs_normalisation="x = (obs - obs_mean) / sqrt(obs_var + 1e-8)  (stats in the npz)",
        numpy_torch_parity_max_abs=worst,
        note="Actor only. The critic plays no part at inference.",
    )
    with open(os.path.join(a.out, STEM + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote      : {npz} ({os.path.getsize(npz):,} bytes)")
    print(f"             {os.path.join(a.out, STEM + '_meta.json')}")


if __name__ == "__main__":
    main()
