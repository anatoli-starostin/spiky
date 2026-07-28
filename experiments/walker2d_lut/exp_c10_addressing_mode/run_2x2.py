"""exp_c10 — the 2x2: addressing (learned hyperplane vs fixed anchors) x forward mode
(hard single-read vs hybrid_smooth top-2 blend), all at nap4/tph32 (#75).

Every cell is trained in the mode it is evaluated in (the primary number), on the same
PPO-teacher dataset, with the same deterministic 100-episode CPU-reference protocol.

Also reports the CROSS number — train hybrid_smooth, evaluate hard — because that is
the honest cost of taking a smooth-trained table to a single-read deployment, which is
what the #74 spiking track would actually compile.
"""
import json, os, subprocess, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C03 = os.path.join(HERE, "..", "exp_c03_distillation")
PY = os.path.expanduser("~/projects/spiky/.venv/bin/python")
sys.path.insert(0, C03)

NAP, TPH = 4, 32
CELLS = [("hyperplane", "hybrid_smooth"), ("hyperplane", "hard"),
         ("fast", "hybrid_smooth"), ("fast", "hard")]


def train(module, mode):
    tag = f"_c10_{mode}"
    name = f"{module}_nap{NAP}_tph{TPH}_h1{tag}"
    res = os.path.join(HERE, f"result_{name}.json")
    if os.path.exists(res):
        print(f"[cached] {name}", flush=True)
        return json.load(open(res))
    print(f"=== {module} + {mode} ===", flush=True)
    r = subprocess.run(
        [PY, "-u", os.path.join(C03, "distill.py"),
         "--module", module, "--nap", str(NAP), "--tph", str(TPH),
         "--forward-mode", mode, "--epochs", "6", "--episodes", "100",
         "--data-dir", C03, "--out-dir", HERE, "--tag", tag],
        cwd=C03, env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED\n{r.stdout[-900:]}\n{r.stderr[-900:]}", flush=True)
        return None
    print("  " + r.stdout.strip().splitlines()[-2], flush=True)
    return json.load(open(res))


def cross_eval_hard(module):
    """Load the hybrid_smooth-trained table and evaluate it in HARD mode."""
    from lut_policy import LUTPolicy
    from eval_lut import eval_policy
    ck = os.path.join(HERE, f"lut_{module}_nap{NAP}_tph{TPH}_h1_c10_hybrid_smooth.pt")
    if not os.path.exists(ck):
        return None
    d = torch.load(ck, map_location="cuda", weights_only=False)
    cfg = dict(d["cfg"]); cfg["forward_mode"] = "hard"      # the only change
    m = LUTPolicy(obs_mean=d["obs_mean"], obs_std=d["obs_std"], device="cuda", **cfg)
    m.load_state_dict(d["state_dict"])
    m = m.to("cuda").eval()
    mean, std, _, _ = eval_policy(m, episodes=100)
    return dict(mean=mean, std=std)


def main():
    out = {}
    for module, mode in CELLS:
        r = train(module, mode)
        if r:
            out[f"{module}|{mode}"] = dict(params=r["total_params"],
                                           mean=r["eval_mean"], std=r["eval_std"],
                                           mse=r["heldout_action_mse"])
    print("\n=== cross: trained hybrid_smooth, evaluated HARD ===", flush=True)
    for module in ("hyperplane", "fast"):
        c = cross_eval_hard(module)
        if c:
            out[f"{module}|cross_smooth2hard"] = c
            print(f"  {module:<12} {c['mean']:8.1f} +/- {c['std']:6.1f}", flush=True)

    print("\n=== 2x2: addressing x forward mode (nap4/tph32, CPU-ref 100 ep) ===")
    print(f"{'variant':<34}{'params':>9}{'mean':>10}{'sigma':>9}{'MSE':>9}")
    for k in ("hyperplane|hybrid_smooth", "hyperplane|hard",
              "fast|hybrid_smooth", "fast|hard"):
        if k in out:
            v = out[k]
            print(f"{k:<34}{v['params']:>9,}{v['mean']:>10.1f}{v['std']:>9.1f}"
                  f"{v.get('mse', float('nan')):>9.4f}")
    json.dump(out, open(os.path.join(HERE, "results_2x2.json"), "w"), indent=1)
    print("\nwrote results_2x2.json")


if __name__ == "__main__":
    main()
