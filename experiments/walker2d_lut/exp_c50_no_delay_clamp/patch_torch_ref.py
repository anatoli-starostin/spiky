"""exp_c50 / exp_c51 — scratch torch patch adding the two delay knobs that isolate the
clamp trap found in exp_c49, plus the table/beta knobs carried from c42-c48.

nucstar's branch stays read-only: `run_parity.sh` extracts `lif_multi_head_lut.py` out of
git into /tmp and this patches that copy. Nothing is checked out or committed.

THE TRAP, restated so the patch is legible. Upstream computes

    a = latency(x) + torch.clamp(self.delay, 0.0, self.t_window)

The lower bound enforces causality (no synapse arrives before its latency code); the upper
keeps arrivals in [0, 2*t_window] so `exp(a/tau)` stays float32-safe in the reference's
cumsum membrane. But below 0 the clamp returns 0 in the forward AND zeroes the gradient, so
a delay pushed negative can never return. Starting from delay_init_std=0 -- every delay
exactly on the floor -- exp_c49 measured 95-97% of 2,176 delays permanently dead by the end
of training, against exp_c36's unclamped run which ended ~40% negative and fully functional.

Two independent knobs, so the two experiments differ only in configuration:

  delay_min          lower clamp bound. 0.0 = upstream. -inf = NO lower bound (exp_c50),
                     reproducing the old BucketLIFDetectorsMHL forward `a = t + delay`
                     while KEEPING the upper t_window cap, so float32 safety in the
                     cumsum membrane is preserved.
  delay_init_const   deterministic positive delay init (exp_c51). Consumes no RNG, so it
                     leaves every downstream parameter's draw sequence untouched -- the
                     same property upstream documents for delay_init_std=0.

Exact-string replacements: if upstream changes shape the patch fails loudly.
"""
import sys

SIG_OLD = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,")
SIG_NEW = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,\n"
           "                 table_init_std: float = 0.1, share_betas: bool = False,\n"
           "                 delay_min: float = 0.0, delay_init_const: float = 0.0,")

INIT_OLD = "        self.delay = nn.Parameter(init)"
INIT_NEW = ("        # exp_c51 SCRATCH: deterministic constant delay init. Consumes no "
            "RNG, so the\n"
            "        # draw order of every later parameter is unchanged.\n"
            "        if float(delay_init_const) != 0.0:\n"
            "            init = init + float(delay_init_const)\n"
            "        self.delay = nn.Parameter(init)\n"
            "        self.delay_min = float(delay_min)")

BETA_OLD = ("        self.beta_base = nn.Parameter(torch.zeros(*(tsh + (1,)), "
            "device=dev))\n"
            "        self.beta_raw = nn.Parameter(torch.full(bsh, "
            "float(inv_softplus_step), device=dev))")
BETA_NEW = ("        _bt = (1, 1) if bool(share_betas) else tsh\n"
            "        self.beta_base = nn.Parameter(torch.zeros(*(_bt + (1,)), "
            "device=dev))\n"
            "        self.beta_raw = nn.Parameter(torch.full(_bt + (M - 1,),\n"
            "                                                "
            "float(inv_softplus_step), device=dev))")

TAB_OLD = ("            self.table = nn.Parameter(0.1 * torch.randn(T, self.cells, O, "
           "device=dev))")
TAB_NEW = ("            self.table = nn.Parameter(float(table_init_std)\n"
           "                                      * torch.randn(T, self.cells, O, "
           "device=dev))")

CLAMP_OLD = ("        a = lat.view(B, 1, 1, -1) + torch.clamp(self.delay, 0.0, "
             "self.t_window).unsqueeze(0)   # (B,T,D,N)")
CLAMP_NEW = ("        # exp_c50 SCRATCH: `delay_min` replaces the hard 0.0 floor. -inf "
             "removes the\n"
             "        # non-negativity trap while KEEPING the upper cap that protects "
             "the cumsum\n"
             "        # membrane from exp(a/tau) overflow.\n"
             "        a = lat.view(B, 1, 1, -1) + torch.clamp(\n"
             "            self.delay, self.delay_min, self.t_window).unsqueeze(0)   "
             "# (B,T,D,N)")

VIEW_A_OLD = "        b = bnd.view(1, T, D, M - 1)"
VIEW_A_NEW = "        b = bnd.unsqueeze(0)        # shared-beta safe"
VIEW_B_OLD = "            b = self.boundaries.view(1, T, D, M - 1)"
VIEW_B_NEW = "            b = self.boundaries.unsqueeze(0)   # shared-beta safe"


def main():
    path = sys.argv[1]
    src = open(path).read()
    for old, new, what in ((SIG_OLD, SIG_NEW, "constructor signature"),
                           (INIT_OLD, INIT_NEW, "delay init line"),
                           (BETA_OLD, BETA_NEW, "beta init"),
                           (TAB_OLD, TAB_NEW, "table init line"),
                           (CLAMP_OLD, CLAMP_NEW, "delay clamp in _membrane"),
                           (VIEW_A_OLD, VIEW_A_NEW, "_bucket boundary view"),
                           (VIEW_B_OLD, VIEW_B_NEW, "forward-eval boundary view")):
        if src.count(old) != 1:
            raise SystemExit(f"patch_torch_ref: expected exactly one {what} to match, "
                             f"found {src.count(old)} — upstream file has changed shape, "
                             f"refusing to guess")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched (scratch, /tmp only): +table_init_std +share_betas +delay_min "
          f"+delay_init_const on {path}")


if __name__ == "__main__":
    main()
