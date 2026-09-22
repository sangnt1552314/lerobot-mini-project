#!/bin/bash
set -euo pipefail

HOME_PATH="/home/n/ntasang"

# Hugging Face cache on persistent home storage so model downloads survive
# across jobs (SLURM_TMPDIR is wiped when the job ends).
export HF_HOME=${HOME_PATH}/cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Scratch tmp on local disk.
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER}/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"

# Run from this script's directory so server.py and server.log resolve here.
cd "$(dirname "$0")"

python server.py > server.log 2>&1 &

~/bin/ngrok http 8000
