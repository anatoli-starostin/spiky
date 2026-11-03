# lut_grad_spectra: gradient geometry of LUT-FFN tables vs dense layers (no training)

Question: why does Muon help dense layers (lopsided singular spectrum of G = X^T Delta) and what would a "Muon for
lookup tables" have to fix? Results and findings: `../../results/lut-grad-spectra-analysis/README.md`.

## Files
- `lib_spec.py`     model build (reuses `nanochat.checkpoint_manager.build_model` = base_train's LUT rebuild + load_state_dict, then the *same* fp8 conversion/filter as `scripts/base_train.py`), batch loader/cache, spectrum / hit statistics helpers.
- `analyze_model.py` one checkpoint -> raw json (parts A spectra, B hit statistics, C cross-table redundancy, D one-step lookahead, E finite-difference sanity + gradient-norm shares).
- `make_report.py`  raw jsons -> `results.json`, `results.csv`, `summary.md`, PNG plots (numpy + matplotlib only).
- `run_models.sh`   runs `analyze_model.py` for dense2000, dense5568, lut1500, lut500, lut500_nodrop (~20 min on one H100); `run_all.sh` = that + `make_report.py`.

## How
Environment as in earlier tasks (run from `experiments/lut_nanochat/nanochat`):
```
export NANOCHAT_BASE_DIR=$HOME/work/nanochat_base TORCH_EXTENSIONS_DIR=$HOME/work/torch_ext/lut_nanochat CUDA_HOME=/usr/local/cuda PYTHONPATH=$PWD HF_HOME=$HOME/work/.cache/huggingface PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH
python ../analysis/lut_grad_spectra/analyze_model.py --tag lut1500 --out RAW_DIR --n-held 8
```
Report (matplotlib is in `envs/extra-matplotlib`, a pip --target dir for uv CPython 3.12):
`PYTHONPATH=$HOME/work/envs/extra-matplotlib python3.12 make_report.py --raw RAW_DIR --out ../../results/lut-grad-spectra-analysis`

## Setup details
- fp8 tensorwise ON as in the runs, eager (no torch.compile), fused LUT kernel; table dropout = the run's (0 for lut1500, 0.2 for lut500; `lut500_nodrop` = same checkpoint, dropout off).
- Data: ClimbMix train loader (bos-bestfit, 16x2048 per micro-batch), starting at the loader state saved in the dense step-2000 checkpoint (pq 46, rg 77, epoch 1: later than anything the LUT checkpoints saw). 8 micro-batches (262,144 tokens) -> batch gradient, per-micro-batch statistics for noise; then 8 further micro-batches as held-out set for D. Batches are cached once and shared by every model.
- Token loss gradient and cell-TV (lambda 10, summed over layers as in base_train) gradient are kept separate; both are reported.
- Cell indices/scores come from `ManifestoLUT._addresses` + `ConfidenceLUT._score` on the hooked cartridge input; checked against the fused kernel output (rel. err ~2e-7).
