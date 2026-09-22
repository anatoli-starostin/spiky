#!/usr/bin/env bash
# ViT autoencoder sweep at 14x14 (task 6dc2177e). seed 0, full held-out test split, local logging.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_vit_logs

run () {
  local name=$1; shift
  if [ -e "runs_vit/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u vit_autoencoder.py --seed 0 --batch 128 --lr 1e-3 --probe-every 25 \
      --out-dir runs_vit --name "$name" "$@" > "runs_vit_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}
W="--warmup 200 --sched cosine"

# (b) matched linear baseline at both budgets
run "linear-s500"            --arch linear --steps 500
run "linear-s5000"           --arch linear --steps 5000
# (a) control: pooled-global latent, with and without warmup, both depths
run "vit-k1-e2d2-nowarm"     --steps 5000 --enc-layers 2 --dec-layers 2
run "vit-k1-e2d2-warm"       --steps 5000 --enc-layers 2 --dec-layers 2 $W
run "vit-k1-e4d4-nowarm"     --steps 5000 --enc-layers 4 --dec-layers 4
run "vit-k1-e4d4-warm"       --steps 5000 --enc-layers 4 --dec-layers 4 $W
# (c) patch 2 -> 49 tokens of dim 4
run "vit-p2-k1-e4d4-warm"    --steps 5000 --patch 2 --enc-layers 4 --dec-layers 4 $W
# (d) 8 latent tokens x 8 dims, learned queries in, cross-attention out
run "vit-k8-e4d4-warm"       --steps 5000 --latent-tokens 8 --enc-layers 4 --dec-layers 4 $W
run "vit-p2-k8-e4d4-warm"    --steps 5000 --patch 2 --latent-tokens 8 --enc-layers 4 --dec-layers 4 $W
# (e) width sweep at the structured bottleneck
run "vit-k8-e4d4-warm-d32"   --steps 5000 --latent-tokens 8 --enc-layers 4 --dec-layers 4 --d-model 32 $W
run "vit-k8-e4d4-warm-d128"  --steps 5000 --latent-tokens 8 --enc-layers 4 --dec-layers 4 --d-model 128 $W
echo "== VIT SWEEP DONE $(date +%T)"
