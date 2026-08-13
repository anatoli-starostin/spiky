"""exp_c38 — time the TORCH reference at the exact shipped config, for the head-to-head.

Runs in the SPIKY venv (torch, no jax). Its JAX counterpart is `bench_jax_actor.py`; both
print the same three numbers at the same shapes so the ratio is meaningful:

    eval forward       module.eval()  — the hard inference path, no softmax, no temps
    train forward      module.train() — the straight-through path
    train fwd+bwd      the thing training actually pays, twice per SAC update

Batch 512 (the training batch), CUDA, fp32, steady state with compile excluded.

TWO TORCH MODES, because the reference carries an `@torch.compile` on forward and the
comparison is only honest if that is stated:
  compiled   as shipped
  eager      TORCHDYNAMO_DISABLE=1

Usage (from run_headtohead.sh):
  PYTHONPATH=<stage> python bench_torch_ref.py [reps]
"""
import json
import os
import sys
import time

import torch

from spiky.lutorch.lif_multi_head_lut import LIFMultiHeadLUT

CFG = dict(input_dim=17, n_heads=1, n_outputs=12, tables_per_head=32,
           n_det=6, n_buckets=2)
BATCH = 512
WARMUP = 15


def timeit(fn, reps):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1e3


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    dev = torch.device("cuda")
    torch.manual_seed(0)
    m = LIFMultiHeadLUT(**CFG, freeze_temperature=True, delay_init_std=4.0,
                        device=dev).to(dev)
    x = torch.randn(BATCH, CFG["input_dim"], device=dev)
    gout = torch.randn(BATCH, CFG["n_heads"], CFG["n_outputs"], device=dev)
    mode = "eager" if os.environ.get("TORCHDYNAMO_DISABLE") == "1" else "compiled"
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  mode={mode}  "
          f"batch={BATCH}  {CFG['tables_per_head']} tables x {CFG['n_det']} det x "
          f"{CFG['n_buckets']} bkt = {m.cells} cells  params {m.param_count():,}",
          flush=True)

    res = {}

    def f_eval():
        m.eval()
        with torch.no_grad():
            return m(x)

    def f_train():
        m.train()
        with torch.no_grad():
            return m(x)

    def f_bwd():
        m.train()
        m.zero_grad(set_to_none=True)
        (m(x) * gout).sum().backward()

    res["eval_fwd"] = timeit(f_eval, reps)
    res["train_fwd"] = timeit(f_train, reps)
    res["train_fwd_bwd"] = timeit(f_bwd, reps)
    for k in ("eval_fwd", "train_fwd", "train_fwd_bwd"):
        print(f"  torch {mode:<9} {k:<16} {res[k]:9.3f} ms", flush=True)

    res["mode"] = mode
    res["batch"] = BATCH
    res["device"] = torch.cuda.get_device_name(0)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(res, open(os.path.join(here, f"bench_torch_{mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
