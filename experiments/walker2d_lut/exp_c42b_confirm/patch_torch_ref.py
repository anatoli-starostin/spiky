"""exp_c42 — add `table_init_std` to a SCRATCH copy of the torch reference.

nucstar's branch stays read-only. `run_parity.sh` extracts `lif_multi_head_lut.py` out of
git into /tmp exactly as every parity harness in this chapter does; this patches that
extracted copy in place, so the change exists only in /tmp and only for the duration of the
parity test. Nothing is checked out, nothing on the branch is modified, nothing is committed.

Upstream hard-codes the table init at `0.1 * torch.randn(...)` with no fan-in or
tables_per_head scaling. Since a row is read one-hot and then SUMMED over `tph` tables, the
head's initial output std is sqrt(tph) * 0.1 -- 0.58 at tph=32, 1.13 at tph=128 -- so the
same constant produces very different initial policies in different configurations. This
exposes the constant as a parameter; `table_init_std = 0.1` reproduces upstream exactly.

Exact-string replacements rather than a diff: if the upstream file changes shape the
replacement fails loudly instead of applying somewhere plausible-looking and wrong.
"""
import sys

SIG_OLD = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,")
SIG_NEW = ("                 *, n_buckets: int = 16, w_max: float = 2.0, "
           "t_window: float = 32.0, delay_init_std: float = 0.0,\n"
           "                 table_init_std: float = 0.1,")

TAB_OLD = ("            self.table = nn.Parameter(0.1 * torch.randn(T, self.cells, O, "
           "device=dev))")
TAB_NEW = ("            # exp_c42 SCRATCH: the hard-coded 0.1 becomes a parameter.\n"
           "            # table_init_std == 0.1 reproduces upstream byte-for-byte "
           "(same RNG draw,\n"
           "            # same scaling), so the default path is unchanged.\n"
           "            self.table = nn.Parameter(float(table_init_std)\n"
           "                                      * torch.randn(T, self.cells, O, "
           "device=dev))")


def main():
    path = sys.argv[1]
    src = open(path).read()
    for old, new, what in ((SIG_OLD, SIG_NEW, "constructor signature"),
                           (TAB_OLD, TAB_NEW, "table init line")):
        if src.count(old) != 1:
            raise SystemExit(f"patch_torch_ref: expected exactly one {what} to match, "
                             f"found {src.count(old)} — upstream file has changed shape, "
                             f"refusing to guess")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched (scratch, /tmp only): +table_init_std on {path}")


if __name__ == "__main__":
    main()
