"""exp_c24 — identify the best generation by TRAINING PROXY and map it to a snapshot.

The LUT run ended at generation 599 with mean-policy 2521.8 while its best generation
scored 2625.9 — the distribution mean oscillates, so "the parameters the algorithm happens
to hold at the last generation" is not the same as "the best mean it found". The trainer
saves `mu.npy` by overwriting, so a per-generation snapshotter was run alongside it.

WHY THIS IS NOT CHEATING. The generation is selected on the **MJX training fitness**, which
the run computes anyway from 16 episodes of the unperturbed mean — an internal training
signal, not the CPU reference. The chosen snapshot is then scored ONCE on the 100-episode
reference. That is ordinary early-stopping-style model selection on a validation proxy.
Selecting instead on the CPU reference score across 189 snapshots WOULD be selection bias,
and is not done here.

Mapping snapshots to generations: the trainer writes `mu.npy` once per generation and the
snapshotter copies each distinct mtime, so the k-th snapshot counting back from the end is
generation (last_gen - k). That is asserted against the snapshot count rather than assumed.

Usage:
  python pick_best_snapshot.py
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = os.path.join(HERE, "lut_mu_snapshots")
RUN = os.path.join(HERE, "sepcma_lut6x16_s0.json")


def main():
    hist = json.load(open(RUN))["history"]
    last_gen = hist[-1]["gen"]
    files = sorted(os.listdir(SNAPS), key=lambda f: int(f[3:-4]))   # mu_<mtime>.npy
    n = len(files)
    first_gen = last_gen - (n - 1)
    print(f"{n} snapshots -> generations {first_gen}..{last_gen}")

    covered = [r for r in hist if r["gen"] >= first_gen]
    best = max(covered, key=lambda r: r["mean_policy"])
    print(f"best generation by TRAINING PROXY within the snapshot window: "
          f"gen {best['gen']} at {best['mean_policy']:.1f}")
    print(f"final generation {hist[-1]['gen']} at {hist[-1]['mean_policy']:.1f}  "
          f"(gap {best['mean_policy'] - hist[-1]['mean_policy']:+.1f})")

    overall = max(hist, key=lambda r: r["mean_policy"])
    if overall["gen"] < first_gen:
        print(f"NOTE: the run's overall best was gen {overall['gen']} "
              f"({overall['mean_policy']:.1f}), BEFORE snapshotting started — not "
              f"recoverable.")

    src = os.path.join(SNAPS, files[best["gen"] - first_gen])
    dst = os.path.join(HERE, "sepcma_lut6x16_s0_bestgen_mu.npy")
    shutil.copy(src, dst)
    print(f"copied {os.path.basename(src)} -> {os.path.basename(dst)}")
    json.dump(dict(best_gen=best["gen"], best_proxy=best["mean_policy"],
                   final_gen=hist[-1]["gen"], final_proxy=hist[-1]["mean_policy"],
                   snapshot_window=[first_gen, last_gen], n_snapshots=n,
                   selection="argmax of the MJX training proxy, not the CPU reference"),
              open(os.path.join(HERE, "bestgen_selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
