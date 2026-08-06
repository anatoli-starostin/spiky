"""exp_c32 — dump the TORCH reference for BucketLIFDetectorsMHL.

Half one of the parity test; runs in the SPIKY venv (torch, no jax). See exp_c31's
equivalent for why the two halves cannot share a process.

TWO CASES. `init` is not sufficient on its own and is actively misleading here: every
per-LUT parameter starts identical across tables (tau_raw=1, log_T_cross=0, log_T_bkt=0,
beta_base=0), `delay` is all zeros, and every row of `beta_raw` holds the SAME constant, so
the boundaries are evenly spaced and identical in all 32 tables. A port that transposed the
boundary axis, or that built the cumulative sum along the wrong dimension, would reproduce
`init` exactly and fail on anything real. `perturbed` gives every tensor a distinct value.

TIES. `torch.sort`'s default tie-break is not stable and this port cannot change it -- the
module is nucstar's and is staged read-only. Inputs are drawn so exact arrival collisions
are a probability-zero event (|x| < 5.33 keeps `clamp(16-3x, 0, 32)` off both rails, and
`delay` is perturbed by a continuous draw).

Runs on CPU deliberately: this is where the 5090 `Tensor.prod(dim=)` problem bit exp_c30.
Bucket LIF has no `prod` -- it addresses by comparison, not by a product of Bernoullis --
so the issue does not arise, but the reference stays on CPU for consistency and because
the values are identical either way.

Usage (from run_parity.sh):
  PYTHONPATH=<dir with spiky/lutorch/bucket_lif_detectors_mhl.py> python torch_ref_dump.py OUT.npz
"""
import sys

import numpy as np
import torch

from spiky.lutorch.bucket_lif_detectors_mhl import BucketLIFDetectorsMHL

# Anatoli's spec: 16 buckets, 32 tables. NOTE there is no n_anchor_pairs in this module --
# rows per table are n_buckets, not 2**nap.
CFG = dict(input_dim=17, n_heads=1, n_outputs=12, tables_per_head=32)
N_BUCKETS = 16
BATCH = 24
EPS = 0.7          # passed through; the module ignores it. Kept to prove it is ignored.


def perturb(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.delay.copy_(0.8 * torch.randn(m.delay.shape, generator=g))
        m.w.copy_(0.3 * torch.randn(m.w.shape, generator=g))
        m.tau_raw.copy_(0.5 + 0.6 * torch.randn(m.tau_raw.shape, generator=g))
        m.log_T_cross.copy_(0.4 * torch.randn(m.log_T_cross.shape, generator=g))
        m.log_T_bkt.copy_(0.4 * torch.randn(m.log_T_bkt.shape, generator=g))
        m.beta_base.copy_(2.0 * torch.randn(m.beta_base.shape, generator=g))
        # Keep the boundaries spread over the window -- a wild draw would collapse them
        # all below the first arrival and every table would sit in the last bucket, which
        # is a degenerate case that tests nothing.
        m.beta_raw.copy_(m.beta_raw + 0.3 * torch.randn(m.beta_raw.shape, generator=g))
        m.table.copy_(0.2 * torch.randn(m.table.shape, generator=g))
    return m


def one_case(name, m, seed, dump):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, CFG["input_dim"], generator=g)
    gout = torch.randn(BATCH, CFG["n_heads"], CFG["n_outputs"], generator=g)

    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        y_hard = m(x, eps=EPS, mode="hard")
        y_soft = m(x, eps=EPS, mode="soft")
        y_hard_e = m(x, eps=0.05, mode="hard")
        y_soft_e = m(x, eps=0.05, mode="soft")
        t_hard, t_soft = m._first_spike(x)
        g_soft = m._bucket_soft(t_soft)
        bnd = m.boundaries
    y_st = m(x, eps=EPS, mode="st")
    (y_st * gout).sum().backward()

    eps_free = max(float((y_hard - y_hard_e).abs().max()),
                   float((y_soft - y_soft_e).abs().max()))
    dump.update({f"p_{name}_{k}": v.detach().numpy() for k, v in m.named_parameters()})
    dump.update({f"g_{name}_{k}": (v.grad.numpy() if v.grad is not None
                                   else np.zeros(tuple(v.shape), np.float32))
                 for k, v in m.named_parameters()})
    dump.update({f"x_{name}": x.numpy(), f"gout_{name}": gout.numpy(),
                 f"y_st_{name}": y_st.detach().numpy(),
                 f"y_hard_{name}": y_hard.numpy(), f"y_soft_{name}": y_soft.numpy(),
                 f"t_hard_{name}": t_hard.numpy(), f"t_soft_{name}": t_soft.numpy(),
                 f"g_soft_{name}": g_soft.numpy(), f"bnd_{name}": bnd.detach().numpy()})

    addr = m.address(x)
    print(f"  [{name}] |st-hard|_max {float((y_st.detach()-y_hard).abs().max()):.3e}   "
          f"eps-insensitivity {eps_free:.3e} (must be 0.0)")
    print(f"          partition sums to 1: max|Σg - 1| = "
          f"{float((g_soft.sum(-1) - 1).abs().max()):.3e}   "
          f"buckets used {len(np.unique(addr.numpy()))}/{N_BUCKETS}   "
          f"boundaries strictly increasing: "
          f"{bool((bnd[:, 1:] > bnd[:, :-1]).all())}")


def main():
    out_path = sys.argv[1]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dump = {}
    print(f"torch {torch.__version__}  cfg {CFG}  n_buckets {N_BUCKETS}  batch {BATCH}")
    one_case("init", BucketLIFDetectorsMHL(**CFG, n_buckets=N_BUCKETS), 11, dump)
    one_case("perturbed",
             perturb(BucketLIFDetectorsMHL(**CFG, n_buckets=N_BUCKETS), 77), 13, dump)

    m = BucketLIFDetectorsMHL(**CFG, n_buckets=N_BUCKETS)
    per = {k: int(v.numel()) for k, v in m.named_parameters()}
    n_par = sum(per.values())
    dump.update(eps=np.float32(EPS), n_params=np.int32(n_par),
                n_buckets=np.int32(N_BUCKETS),
                **{k: np.int32(v) for k, v in CFG.items()})
    np.savez(out_path, **dump)
    print(f"  params: {n_par:,}  ({', '.join(f'{k} {v:,}' for k, v in per.items())})")
    print(f"torch reference written to {out_path}")


if __name__ == "__main__":
    main()
