"""exp_c23 — collect and judge the Phase 1 EGGROLL vs full-rank ES comparison (#75).

THRESHOLDS ARE WRITTEN HERE BEFORE THE RUNS FINISH, deliberately. This chapter has
already had to withdraw three claims that were framed after seeing the data (RESULTS.md,
exp_c22), so the verdicts below are fixed in advance and the script simply reads them off.

WHAT THIS COMPARISON CAN AND CANNOT SHOW
----------------------------------------
n = 1 seed per arm. That is enough to answer the EFFICIENCY question, which is
deterministic bookkeeping (how much noise is sampled, how long a generation takes), and
NOT enough to answer the QUALITY question. exp_c18 measured a seed-to-seed sd of ~500 on
this task for a single configuration; a one-seed difference smaller than that is noise.
So:

  * efficiency   -> reported as a measured fact.
  * quality      -> reported as "indistinguishable at n=1" unless the gap is enormous,
                    with the multi-seed rerun named as the thing that would settle it.

The pre-registered quality bands, in CPU-reference return:
  |gap| < 500        -> consistent with the paper's parity finding; nothing to chase.
  500 <= |gap| <1000 -> suggestive but inside one sd of exp_c18's seed spread; needs seeds.
  |gap| >= 1000      -> larger than seed noise plausibly explains; worth 6 seeds per arm.

Usage:
  python collect.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = [("EGGROLL r=4", "eggroll_mlp256x3_r4_s0"),
        ("full-rank ES", "eggroll_mlp256x3_r0_s0")]

# Pre-registered bands (see docstring). Do not edit after the runs finish.
BAND_NOISE = 500.0
BAND_SEEDS = 1000.0
SOLVED_BAR = 3000.0          # the issue's definition, used everywhere in this track


def load(stem):
    tr = json.load(open(os.path.join(HERE, f"{stem}.json")))
    ev_path = os.path.join(HERE, f"{stem}_cpueval.json")
    ev = json.load(open(ev_path)) if os.path.exists(ev_path) else None
    return tr, ev


def main():
    rows = []
    for label, stem in ARMS:
        try:
            tr, ev = load(stem)
        except FileNotFoundError:
            print(f"!! {label}: {stem}.json missing — arm did not finish")
            continue
        rows.append((label, tr, ev))

    if len(rows) < 2:
        print("\nIncomplete: need both arms before any verdict.")
        return

    print("=== 1. EFFICIENCY (deterministic bookkeeping, n=1 is sufficient) ===\n")
    print(f"{'arm':<14}{'noise floats/gen':>18}{'wall s':>10}{'s/gen':>9}"
          f"{'env-steps':>14}")
    for label, tr, _ in rows:
        print(f"{label:<14}{tr['noise_floats_per_gen']:>18,}{tr['wall_s']:>10,.0f}"
              f"{tr['wall_s']/tr['gens']:>9.1f}{tr['total_env_steps']:>14,}")
    egg, full = rows[0][1], rows[1][1]
    noise_ratio = full["noise_floats_per_gen"] / egg["noise_floats_per_gen"]
    time_ratio = full["wall_s"] / egg["wall_s"]
    print(f"\n  EGGROLL samples {noise_ratio:.1f}x less noise per generation.")
    print(f"  Wall-clock ratio (full-rank / EGGROLL) = {time_ratio:.2f}x")
    if time_ratio < 1.05:
        print("  -> NO WALL-CLOCK WIN HERE, and that is expected: this run is bound by "
              "MJX\n     physics, not by noise generation or the controller matmul. The "
              "memory saving\n     is real and measured; the speed claim needs a regime "
              "where the model, not the\n     simulator, is the cost -- which is Phase 2 "
              "(LUT tables) or a much wider net.")
    else:
        print(f"  -> EGGROLL is {time_ratio:.2f}x faster in wall-clock at equal budget.")

    print("\n=== 2. QUALITY (CPU-reference, deterministic 100 episodes) ===\n")
    if any(ev is None for _, _, ev in rows):
        print("  CPU evals not present yet — rerun once run_phase1.sh completes.")
        return
    print(f"{'arm':<14}{'CPU-ref mean':>14}{'sd':>9}{'full-len eps':>14}"
          f"{'MJX proxy':>12}")
    for label, tr, ev in rows:
        print(f"{label:<14}{ev['cpu_reference_mean']:>14.1f}{ev['cpu_reference_std']:>9.1f}"
              f"{ev['full_length_episodes']:>11}/100"
              f"{tr['best_mean_policy_mjx']:>12.1f}")

    a, b = rows[0][2], rows[1][2]
    gap = a["cpu_reference_mean"] - b["cpu_reference_mean"]
    print(f"\n  gap (EGGROLL - full-rank) = {gap:+.1f}")
    if abs(gap) < BAND_NOISE:
        v = (f"INDISTINGUISHABLE at n=1. The {abs(gap):.0f}-point gap is well inside the "
             f"~500 seed-to-seed sd exp_c18 measured on this task, and is exactly the "
             f"parity the paper reports for its own RL suite. Nothing to chase.")
    elif abs(gap) < BAND_SEEDS:
        better = "EGGROLL" if gap > 0 else "full-rank ES"
        v = (f"SUGGESTIVE BUT UNRESOLVED. {better} leads by {abs(gap):.0f}, which is "
             f"inside one sd of exp_c18's seed spread. n=1 per arm cannot separate this "
             f"from seed luck; 6 seeds per arm would.")
    else:
        better = "EGGROLL" if gap > 0 else "full-rank ES"
        v = (f"LARGER THAN SEED NOISE PLAUSIBLY EXPLAINS. {better} leads by "
             f"{abs(gap):.0f}, beyond the ~500 sd exp_c18 measured. Worth 6 seeds per "
             f"arm to confirm before it is claimed.")
    print(f"\n  {v}")

    print("\n=== 3. AGAINST THE REST OF THE TRACK ===\n")
    print(f"  {'SAC baseline (reference)':<34}5273.4")
    print(f"  {'MJX/PPO baseline':<34}5555.5")
    print(f"  {'exp_c05 OpenAI-ES, MLP[32,32]':<34}2051.1")
    print(f"  {'exp_c05 sep-CMA-ES, MLP[32,32]':<34}2996.7")
    for label, _, ev in rows:
        print(f"  {'exp_c23 ' + label:<34}{ev['cpu_reference_mean']:.1f}")
    best = max(ev["cpu_reference_mean"] for _, _, ev in rows)
    print(f"\n  Best evolved policy here: {best:.1f} "
          f"({'SOLVED' if best >= SOLVED_BAR else 'below'} the {SOLVED_BAR:.0f} bar).")

    json.dump(dict(noise_ratio=noise_ratio, time_ratio=time_ratio, gap=gap, verdict=v,
                   arms=[dict(arm=l, cpu_reference_mean=ev["cpu_reference_mean"],
                              cpu_reference_std=ev["cpu_reference_std"],
                              full_length_episodes=ev["full_length_episodes"],
                              noise_floats_per_gen=tr["noise_floats_per_gen"],
                              wall_s=tr["wall_s"], env_steps=tr["total_env_steps"])
                         for l, tr, ev in rows]),
              open(os.path.join(HERE, "phase1_results.json"), "w"), indent=1)
    print("\nwrote phase1_results.json")


if __name__ == "__main__":
    main()
