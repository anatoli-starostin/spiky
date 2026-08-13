"""Recalibrate the STDP learning rate for batch 64.

STDP accumulates as a SUM over the batch with no averaging, so the effective step scales
linearly with batch size. lr=0.1 at batch 512 sat ~6-8x into the clip (max|dw| railed at
exactly 45 = 1.5*w_max every round, liveness pinned at 1.00). Dropping to batch 64 removes
8x of that on its own; this sweep finds what lr the remainder wants.

Each lr runs in its own subprocess so a crash in one cell cannot contaminate the rest.
"""
import subprocess
import sys

LRS = [0.005, 0.01, 0.02, 0.04]
HERE = __file__.replace("lr_sweep_b64.py", "verify_stage2.py")

rows = []
for lr in LRS:
    r = subprocess.run([sys.executable, HERE, "--pool", "8", "--rounds", "15",
                        "--batch", "64", "--mature-batches", "8",
                        "--stdp-lr", str(lr)],
                       capture_output=True, text=True)
    txt = r.stdout + r.stderr
    line = [ln for ln in txt.splitlines() if ln.startswith("SWEEP")]
    if not line:
        err = [ln.strip() for ln in txt.splitlines() if "Error" in ln]
        print(f"  lr={lr:<6} FAILED  {err[-1][:80] if err else 'rc=' + str(r.returncode)}",
              flush=True)
        continue
    kv = dict(p.split("=", 1) for p in line[0].split()[1:])
    rows.append((lr, kv))
    print(f"  lr={lr:<6} max|dw| med {float(kv['maxdw_med']):7.3f}/45 "
          f"({float(kv['maxdw_med'])/45:5.1%} of range)  "
          f"clipped med {float(kv['clip_med']):6.2%}  "
          f"live med {float(kv['live_med']):.3f}  "
          f"CORR batch {float(kv['corr_batch_last']):+.4f} "
          f"EWMA last {float(kv['corr_ewma_last']):+.4f} best {float(kv['corr_ewma_best']):+.4f}",
          flush=True)

print("\n  (want: max|dw| a modest fraction of 45, clipped ~0, liveness NOT pinned at 1.000,"
      " CORR as high as possible)")
