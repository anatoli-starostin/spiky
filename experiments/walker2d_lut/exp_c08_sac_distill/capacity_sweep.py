"""exp_c08b — capacity sweep for the SAC-taught LUT (#75).

Phase 6 left the SAC-taught LUT at 85.6% of its teacher with a large nominal sigma
(1603), because SAC's smooth policy (only 5.5% of actions saturated) cannot be captured
by 16 rows per table. This sweeps capacity up toward the Phase-1 plateau (~31k params)
looking for the smallest config that MATCHES the SAC teacher's nominal (~5277) with a
tight sigma.

Each config is a separate subprocess so one failure cannot take the sweep down.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DISTILL = os.path.join(HERE, "..", "exp_c03_distillation", "distill.py")
PY = os.path.expanduser("~/projects/spiky/.venv/bin/python")
SAC_TEACHER = 5273.4

# (NAP, tph) spanning 5.4k -> 31.5k params, i.e. up to the Phase-1 plateau
CONFIGS = [(4, 32), (5, 32), (6, 32), (5, 64), (7, 32), (6, 64)]


def main():
    rows = []
    for nap, tph in CONFIGS:
        name = f"hyperplane_nap{nap}_tph{tph}_h1_sac"
        res = os.path.join(HERE, f"result_{name}.json")
        if os.path.exists(res):
            rows.append(json.load(open(res)))
            print(f"[cached] {name}", flush=True)
        else:
            print(f"=== nap{nap} tph{tph} ===", flush=True)
            r = subprocess.run(
                [PY, "-u", DISTILL, "--nap", str(nap), "--tph", str(tph),
                 "--epochs", "6", "--episodes", "100",
                 "--teacher-score", str(SAC_TEACHER),
                 "--data-dir", HERE, "--out-dir", HERE, "--tag", "_sac"],
                cwd=os.path.dirname(DISTILL),
                env=dict(os.environ, OMP_NUM_THREADS="1"),
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAILED\n{r.stdout[-800:]}\n{r.stderr[-800:]}", flush=True)
                continue
            print(r.stdout.strip().splitlines()[-2], flush=True)
            rows.append(json.load(open(res)))
        json.dump(rows, open(os.path.join(HERE, "capacity_sweep.json"), "w"), indent=1)

    rows.sort(key=lambda d: d["total_params"])
    print(f"\n=== SAC-taught LUT capacity sweep (teacher {SAC_TEACHER}) ===")
    print(f"{'config':<26}{'params':>9}{'rows':>6}{'MSE':>9}{'mean':>9}{'sigma':>8}"
          f"{'retention':>11}")
    for d in rows:
        print(f"{d['name']:<26}{d['total_params']:>9,}{d['rows']:>6}"
              f"{d['heldout_action_mse']:>9.4f}{d['eval_mean']:>9.0f}"
              f"{d['eval_std']:>8.0f}{d['teacher_retention_pct']:>10.1f}%")
    matched = [d for d in rows if d["eval_mean"] >= SAC_TEACHER]
    if matched:
        w = min(matched, key=lambda d: d["total_params"])
        print(f"\nSMALLEST config matching the SAC teacher: {w['name']} "
              f"({w['total_params']:,} params) -> {w['eval_mean']:.0f} "
              f"+/- {w['eval_std']:.0f}")
    else:
        best = max(rows, key=lambda d: d["eval_mean"])
        print(f"\nNo config reached {SAC_TEACHER}; best is {best['name']} "
              f"({best['total_params']:,}) -> {best['eval_mean']:.0f} "
              f"+/- {best['eval_std']:.0f} ({best['teacher_retention_pct']:.1f}%)")


if __name__ == "__main__":
    main()
