# lutorch_ex report

Draft report on the `lutorch_ex` library: (1) general architecture, (2) the cartridges and their math +
usage, (3) the 48k champion-topology sweep (7 configs). **Draft — refined iteratively; run numbers are
placeholders until the sweep completes.**

## Build

```
make          # build lutorch_ex_report.pdf (latexmk -pdf, multi-pass)
make clean    # remove aux + PDF
```

The compiled **`lutorch_ex_report.pdf` is committed alongside the source** and must be rebuilt and
re-committed on every `.tex` change, so the PDF always tracks the `.tex`.

## Sources

- Cartridge math: `doc/lutorch/lut_mechanisms.tex` (LightMHL, confidence forms, CompressionMHL).
- Ablation results + quantisation math: `doc/research/lut_ablation/lut_ablation_table_v4.tex`.
- Library code: `src/spiky/lutorch_ex/`.
