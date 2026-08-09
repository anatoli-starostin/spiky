"""Supervisor that makes a long steady-state run survive the create_forward_groups race.

WHY A SUPERVISOR AND NOT AN IN-PROCESS RETRY: the race has two faces. When build_pool RAISES
it is recoverable in-process, and steady_state.build_pool_retry already handles that. When it
HANGS it is not: a spinning CUDA kernel cannot be interrupted or killed from the host, and
the context is wedged regardless. Killing the process is the only lever, so the recovery has
to live OUTSIDE it.

Mechanism: launch the run with --ckpt/--resume, watch the checkpoint's mtime, and if no round
completes within --stall seconds, SIGKILL the process and relaunch it from the checkpoint.
Normal rounds take ~1-20s depending on K, so a 120s stall threshold is far above noise.

    python supervise_run.py -- --pool 128 --rounds 300 --batch 64 ...
Everything after `--` is passed through to steady_state.py verbatim.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def alive(p):
    return p.poll() is None


def _rounds_done(ckpt):
    """next_rnd from the checkpoint, i.e. how many rounds are already finished."""
    try:
        import numpy as np
        return int(np.load(ckpt, allow_pickle=False)["next_rnd"][0])
    except Exception:
        return None


def _requested_rounds(passthrough):
    """--rounds as handed to the child."""
    try:
        return int(passthrough[passthrough.index("--rounds") + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--stall", type=int, default=120,
                    help="seconds without a completed round before declaring a stall; the "
                         "clock starts only after this launch's first checkpoint")
    ap.add_argument("--warmup-max", type=int, default=600,
                    help="seconds a launch may take to write its FIRST checkpoint before it "
                         "is killed anyway (bounds the warm-up exemption)")
    ap.add_argument("--max-restarts", type=int, default=40)
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args after `--` are passed to steady_state.py")
    a = ap.parse_args()
    passthrough = [x for x in a.rest if x != "--"]

    restarts = 0
    t_start = time.time()
    while restarts <= a.max_restarts:
        cmd = [sys.executable, "-u", os.path.join(HERE, "steady_state.py"),
               "--ckpt", a.ckpt, "--resume"] + passthrough
        with open(a.log, "a") as lf:
            lf.write(f"\n=== supervisor: launch #{restarts} at "
                     f"{time.time()-t_start:.0f}s ===\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, cwd=HERE,
                                 start_new_session=True)
        print(f"[supervisor] launch #{restarts}, pid {p.pid}", flush=True)

        last_mtime = os.path.getmtime(a.ckpt) if os.path.exists(a.ckpt) else 0.0
        last_change = time.time()
        # The first round of every launch is far slower than steady state -- the child has to
        # import torch, reload the dataset and rebuild the encoder before round one (~55-77s
        # measured, vs ~20s rounds). So the --stall clock does not start until this launch has
        # written its FIRST checkpoint. `warmed` lives inside the relaunch loop, so every
        # restart gets its own fresh exemption. --warmup-max still bounds the exemption, so a
        # child that wedges during warm-up is not immune forever.
        warmed = False
        launched_at = time.time()
        while alive(p):
            time.sleep(5)
            m = os.path.getmtime(a.ckpt) if os.path.exists(a.ckpt) else 0.0
            if m != last_mtime:
                last_mtime, last_change = m, time.time()
                if not warmed:
                    print(f"[supervisor] warm-up done in {time.time()-launched_at:.0f}s; "
                          f"stall detection now armed at {a.stall}s", flush=True)
                warmed = True
            elif not warmed and time.time() - launched_at > a.warmup_max:
                print(f"[supervisor] STALL during warm-up: no first checkpoint in "
                      f"{a.warmup_max}s -- killing pid {p.pid}", flush=True)
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
                p.wait(timeout=60)
                break
            elif warmed and time.time() - last_change > a.stall:
                print(f"[supervisor] STALL: no round completed in {a.stall}s "
                      f"-- killing pid {p.pid} and resuming from checkpoint", flush=True)
                with open(a.log, "a") as lf:
                    lf.write(f"=== supervisor: STALL detected, killing pid {p.pid} ===\n")
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
                p.wait(timeout=60)
                break
        else:
            rc = p.returncode
            print(f"[supervisor] process exited rc={rc}", flush=True)
            if rc == 0:
                print(f"[supervisor] RUN COMPLETE after {restarts} restart(s), "
                      f"{time.time()-t_start:.0f}s", flush=True)
                return 0
            # COMPLETION CHECK. Without this, a run that has already finished its rounds but
            # then dies in the final held-out evaluation gets "resumed" forever: the round
            # loop is empty, so each relaunch opens a progress bar, does nothing, dies the
            # same way, and relaunches. That burned 41 launches on delay148 and produced a
            # wall of empty bars. If the checkpoint says the rounds are done, we are done.
            done_rounds = _rounds_done(a.ckpt)
            want_rounds = _requested_rounds(passthrough)
            if done_rounds is not None and want_rounds is not None and done_rounds >= want_rounds:
                print(f"[supervisor] checkpoint shows {done_rounds}/{want_rounds} rounds "
                      f"complete -- the run FINISHED and only its post-run step failed "
                      f"(rc={rc}). Not resuming.", flush=True)
                return 0
            print("[supervisor] nonzero exit -- resuming from checkpoint", flush=True)
            time.sleep(min(30, 2 ** min(restarts, 4)))   # back off; don't spin on a hard fail
        restarts += 1
    print(f"[supervisor] gave up after {a.max_restarts} restarts", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
