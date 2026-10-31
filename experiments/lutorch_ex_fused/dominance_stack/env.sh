# source me: environment used for every dominance_stack command (nanochat venv: torch 2.9.1+cu128, triton, wandb, rustbpe)
NB=$HOME/work/repos/anatoli-starostin/spiky-lut_nanochat/experiments/lut_nanochat/nanochat
SPIKY=$HOME/work/repos/anatoli-starostin/spiky
export CUDA_HOME=/usr/local/cuda PYTHONPATH=$SPIKY/src PATH=$NB/.venv/bin:/usr/local/cuda/bin:$PATH
export TORCH_EXTENSIONS_DIR=$HOME/work/torch_ext/dominance_stack NANOCHAT_ROOT=$NB NANOCHAT_BASE_DIR=$HOME/work/nanochat_base
export WANDB_API_KEY=$WANDB_SPIKY_API_KEY
