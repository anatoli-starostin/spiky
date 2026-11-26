#!/bin/bash
# One arm of experiment/lx-ternin-5k16s: ARM = tern4 | tern16 | ternfree. The seed-2 lutorch_ex control (conf n1, single probes, no input proj; spiky b61555bf recipe, config.json = the
# control seed-2 config) + a ternary 384x384 input matrix T per layer in front of the FFN (TERN_IN / TERN_K of train_multi.py). 5000 steps of the 16K schedule. Waits inside the run: verifies the
# W&B curve, uploads the ckpt to a NEW s3 path. Usage: run.sh ARM   (SMOKE_STEPS=N: smoke, no W&B verification / upload, W&B off)
set -o pipefail
ARM=${1:?ARM}
case $ARM in tern4) export TERN_IN=topk TERN_K=4;; tern16) export TERN_IN=topk TERN_K=16;; ternfree) export TERN_IN=free;; *) echo "bad ARM"; exit 5;; esac
W=$HOME/work; R=$(cd "$(dirname "$0")/.." && pwd); D=$(cd "$(dirname "$0")" && pwd)
NAME=lutorch_ex_conf_n1_single_noproj_${ARM}_5k16s_seed2
export CUDA_HOME=/usr/local/cuda PATH=$W/envs/deepspike-ml-torch291/bin:$PATH TORCH_EXTENSIONS_DIR=$W/torch_ext/deepspike-ml-torch291
export PYTHONPATH=$W/n1_5k_setup/xspiky:$W/n1_5k_setup/stub_rustbpe
export NANOCHAT_ROOT=$W/repos/karpathy/nanochat NANOCHAT_BASE_DIR=$W/nanochat_base TOOLS_DIR=$W/n1_5k_setup/research/experiments/ffn_replacement/tools
export CART=conf_n1 ABL47_CONFIG=$D/config.json OUT_DIR=${OUT_DIR:-$W/runs/$NAME} RUN_TAG=5k16s STOP_AT_STEP=${SMOKE_STEPS:-${STOP_AT_STEP:-5000}} EXP_NAME=$NAME CKPT_EVERY=1000
export NO_INPUT_PROJ=1 ANCHOR_MODE=single WANDB_PROJECT_OVERRIDE=tally_transformer
export EXTRA_TAGS=lutorch_ex,5k,16k-schedule,seed2,ternin,$ARM
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com} WANDB_ENTITY=deepspike-inc WANDB_API_KEY=$WANDB_DEEPSPIKE_API_KEY
[ -n "$SMOKE_STEPS" ] && export WANDB_MODE=disabled
S3=s3://deepspike-data/anatoly/tally_transformer/lutorch_ex/$NAME/ckpt.pt
export AWS_ACCESS_KEY_ID=$DEEPSPIKE_DATA_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$DEEPSPIKE_DATA_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3=https://storage.eu-north1.nebius.cloud AWS_DEFAULT_REGION=eu-north1
if [ -z "$SMOKE_STEPS" ] && [ -n "$(timeout 120 $W/envs/awscli/bin/aws s3 ls $S3)" ]; then echo "REFUSING: $S3 exists"; exit 3; fi
mkdir -p $OUT_DIR
cd $R && python train_multi.py 2>&1 | tee $OUT_DIR/train.log || exit 1
[ -n "$SMOKE_STEPS" ] && exit 0
NAME=$NAME ARM=$ARM python $D/verify_wandb.py $OUT_DIR/metrics.csv 2>&1 | tee $OUT_DIR/verify.log || exit 2
timeout 1800 $W/envs/awscli/bin/aws s3 cp $OUT_DIR/ckpt.pt $S3 2>&1 | tee -a $OUT_DIR/verify.log || exit 4
$W/envs/awscli/bin/aws s3 ls $S3 | tee -a $OUT_DIR/verify.log
