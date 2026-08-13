"""exp_c40 — dump the TORCH reference for LIFMultiHeadLUT with the STRUCTURED DELAY init.

Half one of the parity test; runs in the SPIKY venv (torch, no jax). The two halves cannot
share a process: spiky/.venv has torch and no jax, walker2d_mjx/.venv has jax and no torch.

THREE CASES, and each one exists to catch a specific class of port bug:

  run        The exact shipped configuration -- heads=1, tph=32, n_det=3, n_buckets=4,
             freeze_temperature=True, delay_init_std=4 -- at ITS OWN INIT. This is the
             thing we are about to spend GPU on, so it is checked as-is rather than only
             in some convenient surrogate shape. `delay` is left at its half-normal init.

             4**3 = 64 cells, the same row count as exp_c38's 2**6, so this experiment
             isolates digit WIDTH against digit COUNT at fixed table capacity. Unlike c38
             this shape has a NON-EMPTY soft-partition middle term (M=4 gives three
             boundaries per detector) and a radix of [16, 4, 1] rather than powers of two,
             so it exercises paths c38's shape could not.

  perturbed  The same shape with freeze_temperature=FALSE and every tensor given a
             distinct value. Two reasons it is not redundant. (a) At init every per-table
             and per-detector parameter is identical across tables (tau_raw=1,
             log_T_*=0, beta_base=0, one constant in beta_raw), so a port that transposed
             the (T, D) axes or built the boundary cumsum along the wrong dimension would
             reproduce `run` exactly and fail on anything real. (b) With the temperatures
             frozen, log_T_cross and log_T_bkt carry no gradient at all, so `run` cannot
             test their backward paths; this case can.

             `delay` here is deliberately drawn SIGNED and with two entries forced out of
             range, so the [0, t_window] clamp is exercised on both rails. In the shipped
             config the clamp is a no-op at init and would otherwise go untested until it
             first mattered in training.

  alt        A different shape entirely: heads=2, tph=3, n_det=6, n_buckets=2. Two things
             the shipped shape cannot test. n_heads=2 checks the head/tph reshape is not
             silently transposed. And at M=2 the soft partition's middle term
             S[..., :-1] - S[..., 1:] is EMPTY while the mixed-radix weights degenerate to
             powers of two -- the one arrangement where several plausible indexing
             mistakes coincide with the right answer, so it is checked deliberately rather
             than left uncovered. It is also exp_c38's detector/bucket layout, which
             cross-checks this port against a shape already known good.

TIES. `torch.sort`'s default tie-break is not stable and this port cannot change it -- the
module is nucstar's and is staged read-only. Arrival collisions are a probability-zero
event here: latency saturates only at |x| > 16/3 (~1e-7 of standard-normal draws) and the
delays are continuous draws.

Runs on CPU, and with TORCHDYNAMO_DISABLE=1 set by run_parity.sh so the @torch.compile on
forward falls back to eager -- a compiled reference would be testing inductor, not the
module.

Usage (from run_parity.sh):
  PYTHONPATH=<dir with spiky/lutorch/lif_multi_head_lut.py> python torch_ref_dump.py OUT.npz
"""
import sys

import numpy as np
import torch

from spiky.lutorch.lif_multi_head_lut import LIFMultiHeadLUT

BATCH = 24
DELAY_INIT_STD = 0.0          # exp_c48: ZERO delays, c36s setting (consumes no RNG)
TABLE_INIT_STD = 0.1          # exp_c48: STOCK, c36s value — deliberately NOT fan-in
SHARE_BETAS = False           # exp_c47: PER-TABLE ladders — the control for c46
BOUNDARY_OFFSET = 0.0        # stock c39 boundaries — this experiment isolates the table std
INPUT_DIM = 17
N_OUT = 12

# The shipped configuration. Anatoli's spec.
RUN = dict(input_dim=INPUT_DIM, n_heads=1, n_outputs=N_OUT, tables_per_head=128,
           n_det=1, n_buckets=16)
ALT = dict(input_dim=INPUT_DIM, n_heads=2, n_outputs=N_OUT, tables_per_head=3,
           n_det=6, n_buckets=2)

PKEYS = ("delay", "w_raw", "tau_raw", "beta_base", "beta_raw",
         "log_T_cross", "log_T_bkt", "table")


def perturb(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # SIGNED, so ~half get clamped to 0 by the causal floor in the forward.
        m.delay.copy_(3.0 * torch.randn(m.delay.shape, generator=g))
        m.delay[0, 0, 0] = -5.0                    # below the floor
        m.delay[0, 0, 1] = 10.0 * m.t_window       # above the ceiling
        # w_raw, not w -- `w` is a read-only property (w_max * sigmoid(w_raw)). Spread
        # around the hot init so effective weights land across (0, w_max) rather than
        # bunched at one end, which would leave the sigmoid saturation untested.
        m.w_raw.copy_(-2.2 + 1.2 * torch.randn(m.w_raw.shape, generator=g))
        m.tau_raw.copy_(0.5 + 0.6 * torch.randn(m.tau_raw.shape, generator=g))
        m.beta_base.copy_(2.0 * torch.randn(m.beta_base.shape, generator=g))
        # Keep the boundaries spread over the window -- a wild draw would collapse them
        # all below the first arrival and every detector would sit in the last bucket,
        # a degenerate case that tests nothing.
        m.beta_raw.copy_(m.beta_raw + 0.3 * torch.randn(m.beta_raw.shape, generator=g))
        m.log_T_cross.copy_(0.4 * torch.randn(m.log_T_cross.shape, generator=g))
        m.log_T_bkt.copy_(0.4 * torch.randn(m.log_T_bkt.shape, generator=g))
        m.table.copy_(0.2 * torch.randn(m.table.shape, generator=g))
    return m


def one_case(name, m, cfg, seed, dump):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, cfg["input_dim"], generator=g)
    gout = torch.randn(BATCH, cfg["n_heads"], cfg["n_outputs"], generator=g)

    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        m.eval()
        y_eval = m(x)                                  # the efficient hard path
        m.train()
        y_train_nograd = m(x)                          # the ST path, value only
        t_hard, t_soft = m._first_spike(x)
        b_hard, g_soft = m._bucket(t_hard, t_soft)
        y_hard_read = m._hard_read(b_hard)
        y_addr = m._soft_read(g_soft)
        bnd = m.boundaries
        addr = m.address(x)

    m.train()
    y_st = m(x)
    (y_st * gout).sum().backward()

    # The table gradient must be a HARD SCATTER: exactly the cells addressed by
    # (sample, table) receive gradient and every other cell is exactly 0.0. If the soft
    # readout's table were not detached, every cell would be touched and this collapses.
    tg = m.table.grad
    touched = int((tg.abs().sum(dim=-1) > 0).sum())
    exactly_zero = int((tg.abs().sum(dim=-1) == 0.0).sum())
    n_cells = tg.shape[0] * tg.shape[1]

    dump.update({f"p_{name}_{k}": v.detach().numpy() for k, v in m.named_parameters()})
    dump.update({f"g_{name}_{k}": (v.grad.numpy() if v.grad is not None
                                   else np.zeros(tuple(v.shape), np.float32))
                 for k, v in m.named_parameters()})
    dump.update({f"rg_{name}_{k}": np.bool_(v.requires_grad)
                 for k, v in m.named_parameters()})
    dump.update({
        f"x_{name}": x.numpy(), f"gout_{name}": gout.numpy(),
        f"y_st_{name}": y_st.detach().numpy(), f"y_eval_{name}": y_eval.numpy(),
        f"t_hard_{name}": t_hard.numpy(), f"t_soft_{name}": t_soft.numpy(),
        f"b_hard_{name}": b_hard.numpy().astype(np.int32),
        f"g_soft_{name}": g_soft.numpy(), f"bnd_{name}": bnd.detach().numpy(),
        f"y_hard_read_{name}": y_hard_read.numpy(), f"y_addr_{name}": y_addr.numpy(),
        f"addr_{name}": addr.numpy().astype(np.int32),
        f"cells_{name}": np.int32(m.cells),
        f"radix_{name}": m.radix.numpy().astype(np.int64),
        f"frozen_{name}": np.bool_(not m.log_T_cross.requires_grad),
        f"touched_{name}": np.int32(touched),
    })
    dump.update({f"cfg_{name}_{k}": np.int32(v) for k, v in cfg.items()})

    st_eq = float((y_st.detach() - y_eval).abs().max())
    st_eq2 = float((y_train_nograd - y_eval).abs().max())
    nsp = float((t_hard >= m.t_window).float().mean())
    print(f"  [{name}] train(ST) vs eval(hard) forward: max|diff| {st_eq:.3e} "
          f"(no-grad {st_eq2:.3e})")
    print(f"          partition sums to 1: max|Sum g - 1| = "
          f"{float((g_soft.sum(-1) - 1).abs().max()):.3e}   "
          f"cells used {len(np.unique(addr.numpy()))}/{m.cells}   "
          f"boundaries increasing: "
          f"{bool((bnd[..., 1:] > bnd[..., :-1]).all()) if bnd.shape[-1] > 1 else 'n/a (M=2)'}")
    print(f"          table grad scatter: {touched} of {n_cells} cells touched, "
          f"{exactly_zero} EXACTLY 0.0 (cap {BATCH * m.n_tables})   "
          f"NO-SPIKE mass {nsp:.3f}   "
          f"eff w in [{float(m.w.min()):.3f}, {float(m.w.max()):.3f}]   "
          f"tau mean {float(m.tau.mean()):.3f}")
    dead = [k for k, v in m.named_parameters()
            if v.requires_grad and (v.grad is None or float(v.grad.abs().max()) == 0.0)]
    print(f"          trainable params with gradient: "
          f"{sum(1 for _, v in m.named_parameters() if v.requires_grad) - len(dead)}"
          f"/{sum(1 for _, v in m.named_parameters() if v.requires_grad)}"
          f"{'' if not dead else f'   DEAD: {dead}'}")


def main():
    out_path = sys.argv[1]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dump = {}
    print(f"torch {torch.__version__}  batch {BATCH}")
    print(f"run   cfg {RUN}  freeze_temperature=False  delay_init_std={DELAY_INIT_STD}  boundary_offset={BOUNDARY_OFFSET} table_init_std={TABLE_INIT_STD} share_betas={SHARE_BETAS}")
    one_case("run", LIFMultiHeadLUT(**RUN, freeze_temperature=False,
                                    delay_init_std=DELAY_INIT_STD, table_init_std=TABLE_INIT_STD, share_betas=SHARE_BETAS),
             RUN, 11, dump)
    print(f"perturbed cfg {RUN}  freeze_temperature=False")
    one_case("perturbed", perturb(LIFMultiHeadLUT(**RUN), 77), RUN, 13, dump)
    print(f"alt   cfg {ALT}  freeze_temperature=False")
    one_case("alt", perturb(LIFMultiHeadLUT(**ALT), 99), ALT, 17, dump)

    # The parameter budget of the shipped config, reported the way the chapter reports it.
    m = LIFMultiHeadLUT(**RUN, freeze_temperature=False, delay_init_std=DELAY_INIT_STD,
                        table_init_std=TABLE_INIT_STD, share_betas=SHARE_BETAS)
    per = {k: int(v.numel()) for k, v in m.named_parameters()}
    n_par = m.param_count()
    n_train = sum(int(v.numel()) for v in m.parameters() if v.requires_grad)
    dump.update(n_params=np.int32(n_par), n_trainable=np.int32(n_train),
                batch=np.int32(BATCH))
    np.savez(out_path, **dump)
    print(f"\n  shipped config params: {n_par:,} total, {n_train:,} trainable")
    print(f"    {', '.join(f'{k} {v:,}' for k, v in per.items())}")
    print(f"    vs the 28,032 hyperplane baseline: {100 * n_par / 28032:.1f}% "
          f"(trainable {100 * n_train / 28032:.1f}%)")
    print(f"torch reference written to {out_path}")


if __name__ == "__main__":
    main()
