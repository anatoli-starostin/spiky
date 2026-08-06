"""exp_c41 — apply the structured-BOUNDARY init to a SCRATCH copy of the torch reference.

nucstar's branch stays read-only. `run_parity.sh` extracts `lif_multi_head_lut.py` out of
git into /tmp exactly as every parity harness in this chapter does; this patches that
extracted copy in place, so the change exists only in /tmp and only for the duration of the
parity test. Nothing is checked out, nothing on the branch is modified, nothing is committed.

The patch is a pair of exact-string replacements rather than a diff: if the upstream file
changes shape the replacement fails loudly instead of applying somewhere plausible-looking
and wrong.

`boundary_offset` biases `beta_base`, which is the additive base of
`boundaries = beta_base + cumsum(softplus(beta_raw))`. Detector d's whole boundary ladder
slides by d*offset. `beta_raw` is untouched, so the SPACING of a detector's three
boundaries is unchanged -- only where the ladder sits on the time axis.
"""
import sys

SIG_OLD = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,")
SIG_NEW = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,\n"
           "                 boundary_offset: float = 0.0,")

BB_OLD = ("        self.beta_base = nn.Parameter(torch.zeros(*(tsh + (1,)), "
          "device=dev))")
BB_NEW = """        # exp_c41 SCRATCH: per-detector additive boundary offset. Detector d's whole
        # ladder slides by d*boundary_offset; beta_raw (the SPACING) is untouched, and the
        # membrane/spike time is untouched, so no detector can be killed by this.
        # boundary_offset == 0.0 leaves the tensor at zeros and consumes no RNG, so the
        # default is byte-identical to upstream.
        _bb = torch.zeros(*(tsh + (1,)), device=dev)
        if float(boundary_offset) != 0.0:
            _bo = float(boundary_offset) * torch.arange(D, device=dev, dtype=_bb.dtype)
            _bb = _bb + _bo.view(1, D, 1)
        self.beta_base = nn.Parameter(_bb)"""


def main():
    path = sys.argv[1]
    src = open(path).read()
    for old, new, what in ((SIG_OLD, SIG_NEW, "constructor signature"),
                           (BB_OLD, BB_NEW, "beta_base init line")):
        if src.count(old) != 1:
            raise SystemExit(f"patch_torch_ref: expected exactly one {what} to match, "
                             f"found {src.count(old)} — upstream file has changed shape, "
                             f"refusing to guess")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched (scratch, /tmp only): +boundary_offset on {path}")


if __name__ == "__main__":
    main()
