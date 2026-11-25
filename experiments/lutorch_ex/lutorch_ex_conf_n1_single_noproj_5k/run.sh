#!/bin/bash
# Fork of lutorch_ex_conf_n1_16k: no input projection (fan-out h_in=1,h_out=8,d_in=384) + single probes, 5000-step full schedule.
# Waits inside the run, verifies the final W&B values, uploads the checkpoint to a NEW s3 path.
set -o pipefail
W=$HOME/work; R=$W/repos/anatoli-starostin/spiky/experiments/lutorch_ex; D=$R/lutorch_ex_conf_n1_single_noproj_5k
export CUDA_HOME=/usr/local/cuda PATH=$W/envs/deepspike-ml-torch291/bin:$PATH TORCH_EXTENSIONS_DIR=$W/torch_ext/deepspike-ml-torch291
export PYTHONPATH=$W/n1_5k_setup/xspiky:$W/n1_5k_setup/stub_rustbpe
export NANOCHAT_ROOT=$W/repos/karpathy/nanochat NANOCHAT_BASE_DIR=$W/nanochat_base TOOLS_DIR=$W/n1_5k_setup/research/experiments/ffn_replacement/tools
export CART=conf_n1 ABL47_CONFIG=$D/config.json OUT_DIR=${OUT_DIR:-$W/runs/lutorch_ex_conf_n1_single_noproj_5k} RUN_TAG=5k EXP_NAME=lutorch_ex_conf_n1_single_noproj_5k CKPT_EVERY=1000
export NO_INPUT_PROJ=1 ANCHOR_MODE=single WANDB_PROJECT_OVERRIDE=tally_transformer
export EXTRA_TAGS=lutorch_ex,conf_n1,single-probe,no-input-projection,fan-out,5k,fork-of-qwnyrzbh
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com} WANDB_ENTITY=deepspike-inc WANDB_API_KEY=$WANDB_DEEPSPIKE_API_KEY
mkdir -p $OUT_DIR
cd $R && python train_multi.py 2>&1 | tee $OUT_DIR/train.log || exit 1
[ -n "$SMOKE_STEPS" ] && exit 0
# verify final W&B values against the local metrics
python $D/verify_wandb.py $OUT_DIR/metrics.csv 2>&1 | tee $OUT_DIR/verify.log || exit 2
S3=s3://deepspike-data/anatoly/lutorch_ex/lutorch_ex_conf_n1_single_noproj_5k/ckpt.pt
export AWS_ACCESS_KEY_ID=$DEEPSPIKE_DATA_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$DEEPSPIKE_DATA_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3=https://storage.eu-north1.nebius.cloud AWS_DEFAULT_REGION=eu-north1
if [ -n "$(timeout 120 $W/envs/awscli/bin/aws s3 ls $S3)" ]; then echo "REFUSING: $S3 exists"; exit 3; fi
timeout 1800 $W/envs/awscli/bin/aws s3 cp $OUT_DIR/ckpt.pt $S3 2>&1 | tee -a $OUT_DIR/verify.log || exit 4
$W/envs/awscli/bin/aws s3 ls $S3 | tee -a $OUT_DIR/verify.log
