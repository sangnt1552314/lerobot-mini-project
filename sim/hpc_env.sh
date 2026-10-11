#!/usr/bin/env bash
# Run a command (or an interactive shell) in the HPC ACT training environment:
# Singularity image + venv from sim/setup_hpc_train.sh + the /scratch cache paths
# (+ MuJoCo EGL rendering, see sim_mujoco/README.md).
#
#   bash sim/hpc_env.sh                                   # interactive shell (e.g. inside qsub -I)
#   bash sim/hpc_env.sh python sim/tools/inspect_dataset.py --repo-id local/x --root /scratch/.../local/x
#   bash sim/hpc_env.sh lerobot-train --help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
IMAGE="${IMAGE:-/app1/common/singularity-img/hopper/pytorch/pytorch_2.7.0_cuda_12.8.sif}"
VENV="${VENV:-$SCRATCH/virtualenvs/lerobot042-act}"

if [ -z "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then
    command -v singularity >/dev/null 2>&1 || { set +u; module load singularity 2>/dev/null || true; set -u; }
    command -v singularity >/dev/null 2>&1 || { echo "singularity not found (module load singularity)" >&2; exit 1; }
    FWD=()
    for v in SCRATCH_ROOT VENV NCPUS PBS_JOBID WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT; do [ -n "${!v+x}" ] && FWD+=(--env "$v=${!v}"); done
    exec singularity exec -e --nv "${FWD[@]}" "$IMAGE" bash --norc "$REPO_ROOT/sim/hpc_env.sh" "$@"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HOME="$SCRATCH/cache" HF_LEROBOT_HOME="$SCRATCH/datasets/lerobot" HF_HUB_OFFLINE=1
export TORCH_HOME="$SCRATCH/cache/torch"
# Offscreen OpenGL for MuJoCo (sim_mujoco/): the image has no EGL libraries and --nv binds the host's
# libEGL + NVIDIA EGL driver but not libGLdispatch or the EGL vendor file; sim_mujoco/setup_hpc.sh
# puts copies of both in $SCRATCH/opt/egl_shim. Harmless for training.
EGL_SHIM="$SCRATCH/opt/egl_shim"
if [ -f "$EGL_SHIM/10_nvidia.json" ]; then
    export LD_LIBRARY_PATH="$EGL_SHIM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_SHIM/10_nvidia.json" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
fi
export TMPDIR="$SCRATCH/tmp" TEMP="$SCRATCH/tmp" TMP="$SCRATCH/tmp"
mkdir -p "$TMPDIR"
cd "$REPO_ROOT"
if [ $# -eq 0 ]; then exec bash --norc -i; else exec "$@"; fi
