"""exp_c03 Phase 1c + Phase 2 — the capacity sweep / representability curve (#75).

Runs each config as a SEPARATE subprocess so one bad cell cannot take the sweep down,
and writes a partial progress file the Slack bar reads.

Curve axis is total trainable params (table + index); the y-axis is the deterministic
100-episode return in the CPU reference env — never a training proxy.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/projects/spiky/.venv/bin/python")
TEACHER = 5555.5

# (module, NAP, tables_per_head) — spans ~800 to ~6.3M table params
CONFIGS = [
    ("hyperplane", 4, 8), ("hyperplane", 4, 32),
    ("hyperplane", 6, 16), ("hyperplane", 6, 64),
    ("hyperplane", 8, 16), ("hyperplane", 8, 64), ("hyperplane", 8, 256),
    ("hyperplane", 10, 64), ("hyperplane", 10, 256),
    ("hyperplane", 12, 64), ("hyperplane", 12, 256),
    # FastMHL control: fixed anchor-pair addressing, the same table body
    ("fast", 8, 64), ("fast", 10, 256),
]


def main():
    epochs = int(os.environ.get("SWEEP_EPOCHS", "6"))
    episodes = int(os.environ.get("SWEEP_EPISODES", "100"))
    done = []
    t0 = time.time()
    for i, (mod, nap, tph) in enumerate(CONFIGS):
        name = f"{mod}_nap{nap}_tph{tph}_h1"
        print(f"\n=== [{i+1}/{len(CONFIGS)}] {name} ===", flush=True)
        cmd = [PY, "-u", os.path.join(HERE, "distill.py"),
               "--module", mod, "--nap", str(nap), "--tph", str(tph),
               "--epochs", str(epochs), "--episodes", str(episodes)]
        r = subprocess.run(cmd, cwd=HERE, env=dict(os.environ, OMP_NUM_THREADS="1"),
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}",
                  flush=True)
            done.append(dict(name=name, failed=True,
                             error=(r.stderr or "")[-400:]))
        else:
            print(r.stdout.strip().splitlines()[-2] if r.stdout.strip() else "", flush=True)
            f = os.path.join(HERE, f"result_{name}.json")
            done.append(json.load(open(f)) if os.path.exists(f)
                        else dict(name=name, failed=True, error="no result file"))
        json.dump(dict(completed=len(done), total=len(CONFIGS),
                       elapsed_s=round(time.time() - t0, 1), results=done),
                  open(os.path.join(HERE, "sweep_partial.json"), "w"), indent=1)

    ok = [d for d in done if not d.get("failed")]
    ok.sort(key=lambda d: d["total_params"])
    print("\n=== representability curve (CPU-reference 100-ep deterministic) ===",
          flush=True)
    print(f"{'config':<26}{'params':>10}{'rows':>7}{'act MSE':>10}"
          f"{'return':>12}{'retention':>11}  solved")
    for d in ok:
        print(f"{d['name']:<26}{d['total_params']:>10,}{d['rows']:>7}"
              f"{d['heldout_action_mse']:>10.4f}"
              f"{d['eval_mean']:>8.0f}±{d['eval_std']:<4.0f}"
              f"{d['teacher_retention_pct']:>10.1f}%  "
              f"{'YES' if d['solved'] else '-'}", flush=True)
    solved = [d for d in ok if d["solved"]]
    if solved:
        smallest = min(solved, key=lambda d: d["total_params"])
        print(f"\nSMALLEST LUT clearing 3000: {smallest['name']} at "
              f"{smallest['total_params']:,} params -> {smallest['eval_mean']:.0f}",
              flush=True)
    json.dump(dict(teacher=TEACHER, completed=len(done), results=done),
              open(os.path.join(HERE, "sweep_results.json"), "w"), indent=1)
    print("wrote sweep_results.json", flush=True)


if __name__ == "__main__":
    main()
