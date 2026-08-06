"""exp_c39 — assert the JAX LIFMultiHeadLUT port matches the torch reference.

Half two of the parity test; runs in the MJX venv (jax, no torch).

WHAT IS CHECKED, and why each item is here rather than trusted:

  A. BOTH FORWARDS. The reference has NO mode kwarg -- `forward(x)` branches on
     `self.training` -- so the torch side calls `m.train(); m(x)` and `m.eval(); m(x)`,
     and this side calls mode="train" and mode="eval" against them respectively. There is
     no "soft" forward to check because the module no longer has one; the soft readout is
     compared separately as `y_addr`, the internal address-gradient term. Also checked:
     train == eval on the JAX side. The reference's whole straight-through claim is that
     those two are the same value; a port that got the cancellation term wrong would still
     train, just not on the discrete model.

  B. THE INTERMEDIATES, not just the outputs. `boundaries`, `t_hard`, `t_soft`, the
     per-detector soft partition `g`, the per-detector HARD DIGITS `b_hard`, the joint
     mixed-radix cell index `address`, and the two halves of the ST decode (`y_hard_read`,
     `y_addr`) are each compared. Addressing can be wrong in ways the output hides: if the
     radix were LSB-first, or the (T, D) axes transposed, most samples would still land in
     *some* cell and the output would merely be wrong rather than malformed. Comparing the
     digits and the index directly catches it. The digits and the index are compared for
     EXACT integer equality, not within a tolerance.

  C. GRADIENT PARITY ON EVERY PARAMETER, including beta_raw/beta_base (which reach the
     loss only through the soft partition) and log_T_cross/log_T_bkt (only through the
     soft crossing and the soft partition respectively).

  D. THE PARTITION SUMS TO ONE, per detector. sum_m g_m must be 1 for every
     (sample, table, detector). A partition summing to 0.97 would train perfectly happily
     and silently scale every addressed row; nothing else here would notice.

  E. THE BOUNDARIES ARE STRICTLY INCREASING (where M > 2 makes that meaningful) -- the
     invariant the softplus-cumsum parameterisation exists to guarantee.

  F. THE TABLE GRADIENT IS A HARD SCATTER: at most B*n_tables cells receive gradient and
     every other cell is EXACTLY 0.0, not merely small. This is the observable consequence
     of detaching the table in the soft address readout; if that detach were dropped the
     count would jump to every cell.

  G. THE FREEZE IS REAL AND IS NOT MASKING A DEAD PATH. On the frozen case the reference
     reports requires_grad=False and no gradient for log_T_cross/log_T_bkt. JAX has no
     such flag, so the trainer zeroes those two gradients; this asserts that the masked
     JAX gradient matches torch's (both zero) AND that the UNMASKED one is nonzero -- i.e.
     the freeze is suppressing a live gradient rather than sitting on a path that never
     carried one.

Tolerances are relative-to-scale, as in exp_c30/c31/c32b.

Usage:
  python parity_check.py REF.npz
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import jax_mhl_lut as M

TOL = 2e-5
CASES = ("run", "perturbed", "alt")
FREEZE_KEYS = ("log_T_cross", "log_T_bkt")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def main():
    z = np.load(sys.argv[1])
    fails = []

    def check(name, got, ref):
        r = rel(got, ref)
        ok = r < TOL
        fails.append(None if ok else name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} rel {r:.3e}")

    def check_bool(name, ok, detail):
        fails.append(None if ok else name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {detail}")

    print(f"torch reports {int(z['n_params']):,} params for the shipped config "
          f"({int(z['n_trainable']):,} trainable)")

    for case in CASES:
        cfg = {k[len(f'cfg_{case}_'):]: int(z[k])
               for k in z.files if k.startswith(f"cfg_{case}_")}
        heads, tph = cfg["n_heads"], cfg["tables_per_head"]
        nd, nb = cfg["n_det"], cfg["n_buckets"]
        n_tables, cells = heads * tph, int(z[f"cells_{case}"])
        frozen = bool(z[f"frozen_{case}"])
        print(f"\n--- case: {case}  (heads={heads} tph={tph} n_det={nd} n_buckets={nb} "
              f"-> {n_tables} tables x {cells} cells, "
              f"{'FROZEN' if frozen else 'trainable'} temps) ---")

        p = {k[len(f'p_{case}_'):]: jnp.asarray(z[k])
             for k in z.files if k.startswith(f"p_{case}_")}
        x = jnp.asarray(z[f"x_{case}"])
        gout = jnp.asarray(z[f"gout_{case}"])

        # --- exp_c40: the structured delay init is actually present ----------
        # Parity loads torch's parameters, so it exercises the FORWARD under structured
        # delays but not the JAX init that builds them. These two checks close that gap:
        # the reference's dumped delays must carry the per-detector bias, and our own init
        # must impose the same structure. Without them a delay_offset silently dropped on
        # either side would still pass every other assertion in this file.
        if case == "run" and nd == 1:
            # exp_c43 is the n_det=1 special case, where the "mixed-radix" combination is
            # a no-op: radix is [1] and the joint cell index IS the single bucket digit.
            # The general code path must reduce to that exactly, with no off-by-one and no
            # spurious axis, or the model would be addressing a permuted table while every
            # aggregate still looked healthy.
            check_bool(f"{case}: radix is trivial at n_det=1",
                       list(np.asarray(M.radix(nd, nb))) == [1],
                       f"radix {list(np.asarray(M.radix(nd, nb)))}")
            th_, ts_ = M.first_spike(p, x)
            dg_, _ = M.bucket(p, th_, ts_)
            idx_ = M.cell_index(dg_, nd, nb)
            same = bool(np.array_equal(np.asarray(idx_), np.asarray(dg_)[:, :, 0]))
            check_bool(f"{case}: cell index == the single bucket digit", same,
                       f"{int((np.asarray(idx_) != np.asarray(dg_)[:, :, 0]).sum())} "
                       f"of {idx_.size} differ")
            # 63 boundaries per detector, all of which must be strictly increasing --
            # the softplus-cumsum invariant has far more to get right here than at M=4.
            bb = np.asarray(M.boundaries(p))                  # (T, 1, 63)
            inc = bool(np.all(np.diff(bb, axis=-1) > 0))
            check_bool(f"{case}: all {nb-1} boundaries strictly increasing", inc,
                       f"min gap {float(np.diff(bb, axis=-1).min()):.5f} over "
                       f"{bb.shape[0]}x{bb.shape[-1]} = {bb.shape[0]*bb.shape[-1]} gaps")

        if case == "run":
            # exp_c42: the reduced table std must actually be in the reference's dumped
            # table, and our own init must use the same constant. Parity loads torch's
            # parameters, so without these a table_init_std silently dropped on either
            # side would pass every other assertion in this file.
            TSTD = 0.0176777
            tab = np.asarray(p["table"])
            # The trainer's log-sigma bias touches dims ACT: only, so the mu half is the
            # clean read of the draw's std.
            mu_std = float(tab[:, :, :6].std())
            check_bool(f"{case}: torch table drawn at the reduced std",
                       abs(mu_std - TSTD) / TSTD < 0.05,
                       f"mu-half std {mu_std:.5f} vs requested {TSTD:.5f} "
                       f"(stock would be 0.1)")
            kt = jax.random.PRNGKey(0)
            p_lo = M.init(kt, nb, nd, tph, heads, 17, 12, delay_init_std=4.0,
                          table_init_std=TSTD)
            p_hi = M.init(kt, nb, nd, tph, heads, 17, 12, delay_init_std=4.0,
                          table_init_std=0.1)
            lo, hi = float(np.asarray(p_lo["table"]).std()), \
                float(np.asarray(p_hi["table"]).std())
            check_bool(f"{case}: jax table_init_std scales the draw",
                       abs(lo / hi - TSTD / 0.1) < 0.02,
                       f"std ratio {lo/hi:.4f} vs requested {TSTD/0.1:.4f}")
            # The point of the fan-in correction: the SUMMED head output, not the
            # per-element std, is what the policy sees.
            summed = float(np.asarray(p_lo["table"])[:, :, :6].sum(0).std())
            check_bool(f"{case}: summed mu-head output std ~0.1 (fan-in corrected)",
                       0.08 < summed < 0.13,
                       f"sum over {tph} tables gives std {summed:.4f} "
                       f"(stock 0.1 would give {0.1 * np.sqrt(tph):.3f})")

        # --- the mixed-radix weights themselves ------------------------------
        check_bool(f"{case}: radix (MSB-first)",
                   bool(np.array_equal(np.asarray(M.radix(nd, nb)),
                                       z[f"radix_{case}"])),
                   f"{list(np.asarray(M.radix(nd, nb)))}")

        # --- B: the intermediates -------------------------------------------
        bnd = M.boundaries(p)
        check(f"{case}: boundaries", bnd, z[f"bnd_{case}"])
        t_hard, t_soft = M.first_spike(p, x)
        check(f"{case}: t_hard", t_hard, z[f"t_hard_{case}"])
        check(f"{case}: t_soft", t_soft, z[f"t_soft_{case}"])
        b_hard, g_soft = M.bucket(p, t_hard, t_soft)
        check(f"{case}: soft partition g", g_soft, z[f"g_soft_{case}"])
        eqd = bool(np.array_equal(np.asarray(b_hard), z[f"b_hard_{case}"]))
        nbad = int((np.asarray(b_hard) != z[f"b_hard_{case}"]).sum())
        check_bool(f"{case}: bucket digits EXACT", eqd,
                   f"{nbad} of {b_hard.size} differ")
        addr = M.address(p, x, nd, nb)
        eqa = bool(np.array_equal(np.asarray(addr), z[f"addr_{case}"]))
        check_bool(f"{case}: mixed-radix cell index EXACT", eqa,
                   f"{int((np.asarray(addr) != z[f'addr_{case}']).sum())} of "
                   f"{addr.size} differ; {len(np.unique(np.asarray(addr)))}/{cells} "
                   f"cells used")
        check(f"{case}: y_hard_read (ST value half)",
              M.hard_read(p, b_hard, nd, nb), z[f"y_hard_read_{case}"])
        check(f"{case}: y_addr (ST gradient half)",
              M.soft_read(p, g_soft, nd, nb), z[f"y_addr_{case}"])

        # --- A: both forwards ------------------------------------------------
        y_st = M.apply(p, x, heads, tph, nb, nd, mode="train")
        y_hard = M.apply(p, x, heads, tph, nb, nd, mode="eval")
        check(f"{case}: forward train (torch m.train())", y_st, z[f"y_st_{case}"])
        check(f"{case}: forward eval  (torch m.eval())", y_hard, z[f"y_eval_{case}"])
        d = float(np.abs(np.asarray(y_st) - np.asarray(y_hard)).max())
        check_bool(f"{case}: train == eval (jax side)", d < 1e-6, f"max|diff| {d:.3e}")
        try:
            M.apply(p, x, heads, tph, nb, nd, mode="soft")
            ok_no_soft = False
        except ValueError:
            ok_no_soft = True
        check_bool(f"{case}: no 'soft' forward mode exists", ok_no_soft,
                   "mode='soft' rejected — the reference has no soft forward")

        # --- D/E: the structural invariants ----------------------------------
        s = float(np.abs(np.asarray(g_soft).sum(-1) - 1.0).max())
        check_bool(f"{case}: partition sums to 1 (per detector)", s < 1e-5,
                   f"max|Sum g - 1| {s:.3e}")
        bnp = np.asarray(bnd)
        if bnp.shape[-1] > 1:
            inc = bool(np.all(bnp[..., 1:] > bnp[..., :-1]))
            check_bool(f"{case}: boundaries strictly increasing", inc,
                       f"min gap {float(np.diff(bnp, axis=-1).min()):.4f}")
        else:
            print(f"  ----  {case}: boundaries strictly increasing".ljust(52)
                  + "n/a — one boundary per detector at M=2")

        # --- C: gradient parity on every parameter ---------------------------
        def loss(pp):
            return (M.apply(pp, x, heads, tph, nb, nd, mode="train") * gout).sum()

        g = jax.grad(loss)(p)
        for k in sorted(g):
            got = g[k]
            if frozen and k in FREEZE_KEYS:
                # G: the trainer's freeze mask, applied here so the comparison is against
                # what training will actually do.
                raw = float(np.abs(np.asarray(got)).max())
                got = jnp.zeros_like(got)
                check_bool(f"{case}: grad {k} FROZEN -> 0", raw > 0.0,
                           f"unmasked |grad|max {raw:.3e} (nonzero => freeze is real, "
                           f"not a dead path)")
            check(f"{case}: grad {k}", got, z[f"g_{case}_{k}"])

        # --- F: the table gradient is a hard scatter -------------------------
        tg = np.asarray(g["table"])
        touched = int((np.abs(tg).sum(-1) > 0).sum())
        zeros = int((np.abs(tg).sum(-1) == 0.0).sum())
        cap = x.shape[0] * n_tables
        check_bool(f"{case}: table grad is a hard scatter",
                   touched <= cap and touched + zeros == n_tables * cells,
                   f"{touched} of {n_tables*cells} cells touched (cap {cap}), "
                   f"{zeros} EXACTLY 0.0")
        check_bool(f"{case}: table touched-count matches torch",
                   touched == int(z[f"touched_{case}"]),
                   f"jax {touched} vs torch {int(z[f'touched_{case}'])}")

        # --- nothing dead ----------------------------------------------------
        live = [k for k in sorted(g)
                if not (frozen and k in FREEZE_KEYS)]
        dead = [k for k in live if float(np.abs(np.asarray(g[k])).max()) == 0.0]
        check_bool(f"{case}: no dead trainable parameter", not dead,
                   f"all {len(live)} receive gradient" if not dead
                   else f"DEAD: {dead}")

        # --- the frozen values are what torch says they are ------------------
        if frozen:
            ok = all(float(np.abs(np.asarray(p[k])).max()) == 0.0 for k in FREEZE_KEYS)
            check_bool(f"{case}: frozen temps pinned at T=1.0", ok,
                       "log_T_cross = log_T_bkt = 0.0 exactly")
            check_bool(f"{case}: torch reports requires_grad=False",
                       all(not bool(z[f"rg_{case}_{k}"]) for k in FREEZE_KEYS),
                       "both temperatures non-trainable in the reference")

    bad = [f for f in fails if f]
    if bad:
        print(f"\nPARITY FAILED on {len(bad)}: {', '.join(bad)}")
        sys.exit(1)
    print(f"\nPARITY OK — {len(fails)} checks over {len(CASES)} cases, "
          f"all within {TOL:.0e} relative")


if __name__ == "__main__":
    main()
