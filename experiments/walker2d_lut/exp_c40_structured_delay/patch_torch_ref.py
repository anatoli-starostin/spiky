"""exp_c40 — apply the structured-delay init to a SCRATCH copy of the torch reference.

nucstar's branch stays read-only. `run_parity.sh` extracts `lif_multi_head_lut.py` out of
git into /tmp exactly as every parity harness in this chapter does; this patches that
extracted copy in place, so the change exists only in /tmp and only for the duration of the
parity test. Nothing is checked out, nothing on the branch is modified, nothing is committed.

The patch is deliberately a pair of exact-string replacements rather than a diff: if the
upstream file changes shape the replacement fails loudly instead of applying somewhere
plausible-looking and wrong.

Usage:
  python patch_torch_ref.py <path to staged lif_multi_head_lut.py>
"""
import sys

SIG_OLD = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,")
SIG_NEW = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,\n"
           "                 delay_offset: float = 0.0,")

INIT_OLD = """        if float(delay_init_std) > 0.0:
            init = (float(delay_init_std) * torch.randn(*dsh, device=dev)).abs()
        else:
            init = torch.zeros(*dsh, device=dev)
        self.delay = nn.Parameter(init)"""

INIT_NEW = """        if float(delay_init_std) > 0.0:
            init = (float(delay_init_std) * torch.randn(*dsh, device=dev)).abs()
        else:
            init = torch.zeros(*dsh, device=dev)
        # exp_c40 SCRATCH: per-detector additive delay bias, shared across that detector's
        # N synapses, on top of the i.i.d. jitter. Clamped to the same [0, t_window] the
        # forward uses. delay_offset == 0.0 leaves the tensor untouched and consumes no
        # RNG, so the default is byte-identical to upstream.
        if float(delay_offset) != 0.0:
            _bias = float(delay_offset) * torch.arange(D, device=dev, dtype=init.dtype)
            init = torch.clamp(init + _bias.view(1, D, 1), 0.0, float(t_window))
        self.delay = nn.Parameter(init)"""


def main():
    path = sys.argv[1]
    src = open(path).read()
    for old, new, what in ((SIG_OLD, SIG_NEW, "constructor signature"),
                           (INIT_OLD, INIT_NEW, "delay init block")):
        if src.count(old) != 1:
            raise SystemExit(f"patch_torch_ref: expected exactly one {what} to match, "
                             f"found {src.count(old)} — upstream file has changed shape, "
                             f"refusing to guess")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched (scratch, /tmp only): +delay_offset on {path}")


if __name__ == "__main__":
    main()
