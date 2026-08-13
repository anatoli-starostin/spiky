"""exp_c31 — dump the TORCH reference for PureLIFDetectorsMHL: params, inputs, outputs,
gradients.

Half one of the parity test. This half runs in the SPIKY venv (torch, no jax); the other
half (`parity_check.py`) runs in the MJX venv (jax, no torch). The two venvs are disjoint,
so parity cannot be asserted inside one process -- it is asserted across an npz.

TWO CASES, not one, and the second is the one that can actually fail:

  * `init`      -- the module exactly as constructed. Every per-table parameter is
                   IDENTICAL across tables at init (tau_raw=1, log_T_cross=0,
                   log_temp_bit=0) and `delay` is all zeros. A port that repeated the
                   per-table params in the WRONG order, or that mixed up the (n_tables,
                   nap) grouping, would pass this case by symmetry.
  * `perturbed` -- every parameter given a distinct random value, including `delay`.
                   Breaks the symmetry, so `repeat(tau, nap)` vs `tile(tau, nap)` and any
                   transposed detector index become visible. This is the real test.

TIES. The reference calls `torch.sort` with its default (non-stable) tie-break, and this
port cannot change that -- the module belongs to nucstar's branch and is staged read-only.
So the inputs are drawn to avoid ties entirely: `latency = clamp(16 - 3x, 0, 32)` saturates
only for |x| > 5.33, and x ~ N(0,1) with `delay` perturbed by a continuous draw makes exact
arrival collisions a probability-zero event. Ties can still occur during TRAINING on real
observations; that is a self-consistency question inside JAX, not a parity question, and
`jax_pure_lif` pins its own tie-break to `stable=True`.

Runs on CPU deliberately: `Tensor.prod(dim=)` fails to compile on this box's RTX 5090 under
torch 2.9.1+cu130, and `_prow` uses it. That is a torch/Blackwell issue, not a model bug,
and it does not affect the reference values.

Usage (from run_parity.sh):
  PYTHONPATH=<dir with spiky/lutorch/pure_lif_detectors_mhl.py> python torch_ref_dump.py OUT.npz
"""
import sys

import numpy as np
import torch

from spiky.lutorch.pure_lif_detectors_mhl import PureLIFDetectorsMHL

# The shape the SAC actor will use: 17 obs in, 6 mu + 6 log-sigma out, nap6/tph32 -- the
# same geometry as exp_c18's hyperplane anchor and as exp_c30/c30b.
CFG = dict(input_dim=17, n_heads=1, n_outputs=12, n_anchor_pairs=6, tables_per_head=32)
BATCH = 24
EPS = 0.7          # passed through; the module ignores it. Kept to prove it is ignored.


def perturb(m, seed):
    """Give every parameter a distinct value, so no symmetry can hide an indexing bug."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.delay.copy_(0.8 * torch.randn(m.delay.shape, generator=g))
        m.w.copy_(0.3 * torch.randn(m.w.shape, generator=g))
        # L near the middle of the window, spread wide enough that both bit values occur.
        m.L.copy_(0.5 * m.t_window + 4.0 * torch.randn(m.L.shape, generator=g))
        m.tau_raw.copy_(0.5 + 0.6 * torch.randn(m.tau_raw.shape, generator=g))
        m.log_T_cross.copy_(0.4 * torch.randn(m.log_T_cross.shape, generator=g))
        m.log_temp_bit.copy_(0.3 * torch.randn(m.log_temp_bit.shape, generator=g))
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
        # eps is documented as unused; assert it rather than trusting the docstring.
        y_hard_other_eps = m(x, eps=0.05, mode="hard")
        y_soft_other_eps = m(x, eps=0.05, mode="soft")
    y_st = m(x, eps=EPS, mode="st")
    (y_st * gout).sum().backward()

    eps_free = max(float((y_hard - y_hard_other_eps).abs().max()),
                   float((y_soft - y_soft_other_eps).abs().max()))
    dump.update({f"p_{name}_{k}": v.detach().numpy() for k, v in m.named_parameters()})
    dump.update({f"g_{name}_{k}": (v.grad.numpy() if v.grad is not None
                                   else np.zeros(tuple(v.shape), np.float32))
                 for k, v in m.named_parameters()})
    dump.update({f"x_{name}": x.numpy(), f"gout_{name}": gout.numpy(),
                 f"y_st_{name}": y_st.detach().numpy(),
                 f"y_hard_{name}": y_hard.numpy(), f"y_soft_{name}": y_soft.numpy()})

    st_hard = float((y_st.detach() - y_hard).abs().max())
    bits_set = float((m.address(x) > 0).float().mean())
    print(f"  [{name}] |st-hard|_max {st_hard:.3e}   "
          f"eps-insensitivity {eps_free:.3e} (must be 0.0)   "
          f"nonzero addresses {100*bits_set:.1f}%")
    return st_hard, eps_free


def main():
    out_path = sys.argv[1]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dump = {}
    print(f"torch {torch.__version__}  cfg {CFG}  batch {BATCH}")
    one_case("init", PureLIFDetectorsMHL(**CFG), 11, dump)
    one_case("perturbed", perturb(PureLIFDetectorsMHL(**CFG), 77), 13, dump)

    m = PureLIFDetectorsMHL(**CFG)
    n_par = sum(int(p.numel()) for p in m.parameters())
    per = {k: int(v.numel()) for k, v in m.named_parameters()}
    dump.update(eps=np.float32(EPS), n_params=np.int32(n_par),
                **{k: np.int32(v) for k, v in CFG.items()})
    np.savez(out_path, **dump)

    print(f"  params: {n_par:,}  ({', '.join(f'{k} {v:,}' for k, v in per.items())})")
    print(f"torch reference written to {out_path}")


if __name__ == "__main__":
    main()
