# lutorch_ex 48k champion-topology sweep — run artifacts

Seven configs at the champion topology (h8/tph64/nap8/d48, 6 blocks, ~68.2M params, cell-TV λ=10,
head-dropout 0.2, 48k-step cosine, seed 1, faithful small-std init). One subfolder per wandb run.
Each holds the lightweight artifacts: `config.json` (functional config; descriptive `_*_note` prose
with internal ids stripped), `metrics.csv` (full logged history), `summary.json`, and — for the runs
trained on the VM — `train.log` + `run_tag.txt`. The three worker runs (`*_wk_*`) were reconstructed
from wandb (config + history + summary), so they have no local `train.log`. Model checkpoints and raw
wandb directories are intentionally excluded.

| run (subfolder) | cartridge | host | final val_bpb |
| --- | --- | --- | --- |
| lutorch_ex_abl47_conf_1004_1148 | ConfidenceLUT n=2 | VM | 1.10606 |
| lutorch_ex_abl47_quant_1004_1148 | QuantisedConfidenceLUT n=2 (p2_int8) | VM | 1.10724 |
| lutorch_ex_abl47_conf_n1_vm_1004_1806 | ConfidenceLUT n=1 | VM | 1.11737 |
| lutorch_ex_abl47_fss_smooth_vm_1004_1806 | FusedSoftSignSmooth | VM | 1.12941 |
| lutorch_ex_abl47_fms_wk_1004_1510 | FusedManifestoSoft | Worker | 1.13949 |
| lutorch_ex_abl47_fmh_wk_1004_1510 | FusedManifestoHard | Worker | 1.15152 |
| lutorch_ex_abl47_fss_hard_wk_1004_1510 | FusedSoftSignHard | Worker | 1.15647 |

wandb: project `Spiky`, group `lutorch_ex`, entity `anatoli-starostin-relocation`.
