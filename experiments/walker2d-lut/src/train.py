"""Single entry point: select the trainer with --algo {ppo,sac}. All other flags pass
through to the chosen trainer (see ppo.py / sac.py for their args).

    python train.py --algo ppo --arch mlp --envs 8192 --graph
    python train.py --algo sac --arch mlp --envs 8192 --graph --utd 4
"""
import sys


def main():
    argv = sys.argv[1:]
    algo = "ppo"
    if "--algo" in argv:
        i = argv.index("--algo"); algo = argv[i + 1]; del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv
    if algo == "sac":
        import sac; sac.main()
    elif algo == "ppo":
        import ppo; ppo.main()
    else:
        raise SystemExit(f"unknown --algo {algo!r} (choose ppo|sac)")


if __name__ == "__main__":
    main()
