"""exp_c45 — add `table_init_std` and `share_betas` to a SCRATCH copy of the torch reference.

nucstar's branch stays read-only: `run_parity.sh` extracts `lif_multi_head_lut.py` out of
git into /tmp, and this patches that extracted copy. Nothing is checked out, nothing on the
branch is modified, nothing is committed.

`share_betas` ties the bucket ladder across every (table, detector): ONE `beta_base` scalar
and ONE `beta_raw` vector of length M-1, broadcast everywhere, so every detector of every
table quantises on identical boundaries.

THE ONE NON-OBVIOUS PART. Upstream reshapes the boundaries with

    b = bnd.view(1, T, D, M - 1)

which is an exact-element-count operation and therefore FAILS outright on a shared ladder
(31 elements cannot be viewed as 64x1x31). Replacing it with `bnd.unsqueeze(0)` is
IDENTICAL in the unshared case -- boundaries is (T, D, M-1), so unsqueeze gives exactly
(1, T, D, M-1) -- and broadcasts correctly in the shared case, where it gives
(1, 1, 1, M-1) against a (B, T, D, 1) spike time. Both call sites (`_bucket` and the eval
branch of `forward`) need it.

Exact-string replacements rather than a diff: if upstream changes shape the patch fails
loudly instead of applying somewhere plausible-looking and wrong.
"""
import sys

SIG_OLD = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,")
SIG_NEW = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,\n"
           "                 table_init_std: float = 0.1, share_betas: bool = False,")

BETA_OLD = ("        self.beta_base = nn.Parameter(torch.zeros(*(tsh + (1,)), "
            "device=dev))\n"
            "        self.beta_raw = nn.Parameter(torch.full(bsh, "
            "float(inv_softplus_step), device=dev))")
BETA_NEW = ("        # exp_c45 SCRATCH: share_betas ties the ladder across all (table, "
            "detector) --\n"
            "        # ONE beta_base scalar and ONE beta_raw vector, broadcast "
            "everywhere.\n"
            "        _bt = (1, 1) if bool(share_betas) else tsh\n"
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

# `view` needs an exact element count and breaks on a shared ladder; `unsqueeze(0)` is
# identical when unshared and broadcasts when shared.
VIEW_A_OLD = "        b = bnd.view(1, T, D, M - 1)"
VIEW_A_NEW = ("        b = bnd.unsqueeze(0)        # exp_c45: broadcasts when betas are "
              "shared")
VIEW_B_OLD = "            b = self.boundaries.view(1, T, D, M - 1)"
VIEW_B_NEW = "            b = self.boundaries.unsqueeze(0)   # exp_c45: shared-beta safe"


def main():
    path = sys.argv[1]
    src = open(path).read()
    for old, new, what in ((SIG_OLD, SIG_NEW, "constructor signature"),
                           (BETA_OLD, BETA_NEW, "beta_base/beta_raw init"),
                           (TAB_OLD, TAB_NEW, "table init line"),
                           (VIEW_A_OLD, VIEW_A_NEW, "_bucket boundary view"),
                           (VIEW_B_OLD, VIEW_B_NEW, "forward-eval boundary view")):
        if src.count(old) != 1:
            raise SystemExit(f"patch_torch_ref: expected exactly one {what} to match, "
                             f"found {src.count(old)} — upstream file has changed shape, "
                             f"refusing to guess")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched (scratch, /tmp only): +table_init_std +share_betas on {path}")


if __name__ == "__main__":
    main()
