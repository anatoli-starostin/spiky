"""exp_c30 — dump the TORCH reference: params, inputs, outputs and gradients.

Half one of the parity test. This half runs in the SPIKY venv (torch, no jax); the other
half (`parity_check.py`) runs in the MJX venv (jax, no torch). The two venvs are disjoint,
so parity cannot be asserted inside one process -- it is asserted across an npz.

Everything the JAX side needs to reproduce the reference exactly is written out: the
initialised parameters (so the port is never compared against a DIFFERENT random draw),
the input batch, the upstream gradient, and the reference outputs/gradients.

Runs on CPU deliberately: `Tensor.prod(dim=)` fails to compile on this box's RTX 5090
under torch 2.9.1+cu130, and `_prow` uses it. That is a torch/Blackwell issue, not a model
bug, and it does not affect the reference values.

Usage (from run_parity.sh):
  PYTHONPATH=<dir with spiky/lutorch/lif_detectors_mhl.py> python torch_ref_dump.py OUT.npz
"""
import sys

import numpy as np
import torch

from spiky.lutorch.lif_detectors_mhl import LIFDetectorsMHL

# The shape the SAC actor will use: 17 obs in, 6 mu + 6 log-sigma out, nap6/tph32.
CFG = dict(input_dim=17, n_heads=1, n_outputs=12, n_anchor_pairs=6, tables_per_head=32)
BATCH = 24
EPS = 0.7          # mid-anneal: exercises the gate in its soft regime, not a hard limit


def main():
    out_path = sys.argv[1]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    m = LIFDetectorsMHL(**CFG)

    g = torch.Generator().manual_seed(11)
    x = torch.randn(BATCH, CFG["input_dim"], generator=g)
    gout = torch.randn(BATCH, CFG["n_heads"], CFG["n_outputs"], generator=g)

    with torch.no_grad():
        y_hard = m(x, eps=EPS, mode="hard")
        y_soft = m(x, eps=EPS, mode="soft")
    y_st = m(x, eps=EPS, mode="st")
    (y_st * gout).sum().backward()

    dump = {f"p_{k}": v.detach().numpy() for k, v in m.named_parameters()}
    dump.update({f"g_{k}": (v.grad.numpy() if v.grad is not None
                            else np.zeros(tuple(v.shape), np.float32))
                 for k, v in m.named_parameters()})
    dump.update(x=x.numpy(), gout=gout.numpy(),
                y_st=y_st.detach().numpy(), y_hard=y_hard.numpy(),
                y_soft=y_soft.numpy(),
                eps=np.float32(EPS),
                **{k: np.int32(v) for k, v in CFG.items()})
    np.savez(out_path, **dump)

    st_hard = float(np.abs(y_st.detach().numpy() - y_hard.numpy()).max())
    print(f"torch reference written to {out_path}")
    print(f"  torch {torch.__version__}  shape {tuple(y_st.shape)}  eps {EPS}")
    print(f"  |st - hard|_max = {st_hard:.3e}   (must be ~0: ST forward IS the hard value)")
    print(f"  params dumped: {len(list(m.named_parameters()))}")


if __name__ == "__main__":
    main()
