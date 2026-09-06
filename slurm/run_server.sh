#!/bin/bash
#SBATCH --job-name=lerobot-mini-project-run-server
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=./logs/server/slurm-%j.out
#SBATCH --error=./logs/server/slurm-%j.err

# ---------------------------------------------------------------------------
# NUS SoC SLURM worker for the lerobot-mini-project inference server.
#
# NOTE: --time is a SAFETY CAP ONLY. Stop the job once you are done; the server
# otherwise runs until the walltime is hit.
# ---------------------------------------------------------------------------

ENV_NAME="py312"
HOME_PATH="/home/n/ntasang"
PROJECT_PATH="${HOME_PATH}/projects/lerobot-mini-project"

cd "${PROJECT_PATH}" || exit 1

LOG_DIR="./logs/server"
mkdir -p "${LOG_DIR}"

# Activate the project virtualenv (must already have torch/torchvision + the
# server deps from requirements_server.txt installed).
source "${HOME_PATH}/${ENV_NAME}/bin/activate"

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

echo "Starting policy worker at $(date)"
echo "Running on host: $(hostname)"
echo "Using Hugging Face cache: $HF_HOME"
df -Th "$HF_HOME"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# This job only starts the FastAPI app on port 8000. It does NOT start an ngrok
# tunnel. Two cases:
#   - If SERVER_URL on the local computer already points to a tunnel that
#     someone/something else manages (pointing at this node's port 8000),
#     nothing else is needed here.
#   - If this job is also responsible for exposing the tunnel, you must also
#     run an ngrok agent on this node forwarding to localhost:8000, e.g.:
#       ngrok http 8000
python server/server.py

echo "Policy worker ended at $(date)"
