# Source material for the lutorch_ex report

The original documents the report draws on, vendored here so the next person doesn't have to hunt for them.

| file | what it is | origin |
| --- | --- | --- |
| `lut_ablation_table_v4.tex` / `.pdf` | The OLD LUT core-module ablation table — three generations of lookup math, their cost, and the ablation results. Source of the ablation framing + the report's preamble style. | `doc/research/lut_ablation/lut_ablation_table_v4.tex` on the research branch (`research/ffn_replacement_fix`). |
| `lut_mechanisms.tex` / `.pdf` | "The Mathematics of Our LUT Mechanisms" — the cartridge math (FastMHL, LightMHL, the confidence forms, CompressionMHL, LookupFFN). Primary source for the per-cartridge math. | `doc/lutorch/lut_mechanisms.tex` on the research branch. |
| `quantisation_simple_norowops.md` | "LUT read and quantisation (simplified)" — the dedicated p2_int8 / power-of-two / straight-through reference (score, blend→shifts, int storage, the decompress fold). Source of the QuantisedConfidenceLUT section's exact constants. | Google Drive PDF shared by Anatoli (`quantisation_simple_norowops_45ef4360.pdf`); extracted to Markdown via the Drive connector. |

The report itself is `../lutorch_ex_report.tex` (+ built `.pdf`); rebuild with `make` in the parent dir.
