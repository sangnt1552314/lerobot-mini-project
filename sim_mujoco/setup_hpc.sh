#!/usr/bin/env bash
# One-time setup of the MuJoCo LiftCube pipeline on the NUS HPC (on top of sim/setup_hpc_train.sh).
#
#   bash sim_mujoco/setup_hpc.sh          # login node is fine (needs internet)
#
#   1. mujoco 3.3.x into the ACT venv (same container numpy/torch pins as training)
#   2. MuJoCo Menagerie's robotstudio_so101 model (Apache-2.0) at a pinned commit
#      -> sim_mujoco/third_party/mujoco_menagerie (gitignored)
#   3. EGL shim for offscreen rendering in the container -> $SCRATCH/opt/egl_shim: a copy of the host's
#      libGLdispatch.so.0 (glvnd; --nv does not bind it) and an EGL vendor file for libEGL_nvidia.so.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
MENAGERIE_REF="${MENAGERIE_REF:-0059d4335f8156206f63a35662313385f7ad6d74}"
MENAGERIE="$REPO_ROOT/sim_mujoco/third_party/mujoco_menagerie"
EGL_SHIM="$SCRATCH/opt/egl_shim"
info() { printf '\033[1;34m[setup-mujoco]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup-mujoco] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$SCRATCH/virtualenvs/lerobot042-act/bin/activate" ] || die "Run the training setup first: qsub pbs/setup_act_env.pbs"

info "installing mujoco into the ACT venv"
bash "$REPO_ROOT/sim/hpc_env.sh" bash -c "HF_HUB_OFFLINE=0 PIP_CONSTRAINT=$REPO_ROOT/sim/constraints_hpc_train.txt \
    pip install -q 'mujoco>=3.3,<3.4' && python -c 'import mujoco; print(\"mujoco\", mujoco.__version__)'"

if [ -f "$MENAGERIE/robotstudio_so101/so101.xml" ]; then
    info "Menagerie SO-101 already at $MENAGERIE"
else
    info "fetching MuJoCo Menagerie robotstudio_so101 @ ${MENAGERIE_REF:0:8}"
    mkdir -p "$(dirname "$MENAGERIE")"
    git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE"
    git -C "$MENAGERIE" sparse-checkout set robotstudio_so101
    git -C "$MENAGERIE" checkout -q "$MENAGERIE_REF"
fi

if [ -f "$EGL_SHIM/10_nvidia.json" ] && [ -e "$EGL_SHIM/lib/libGLdispatch.so.0" ]; then
    info "EGL shim already at $EGL_SHIM"
else
    [ -e /usr/lib64/libGLdispatch.so.0 ] || die "host /usr/lib64/libGLdispatch.so.0 not found (run on a cluster node)"
    mkdir -p "$EGL_SHIM/lib"
    cp -aL /usr/lib64/libGLdispatch.so.0 "$EGL_SHIM/lib/libGLdispatch.so.0"
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : { "library_path" : "libEGL_nvidia.so.0" }\n}\n' \
        > "$EGL_SHIM/10_nvidia.json"
    info "EGL shim written to $EGL_SHIM"
fi

info "checking the scene compiles (physics only; rendering needs a GPU node)"
bash "$REPO_ROOT/sim/hpc_env.sh" python -c "
import sys; sys.path.insert(0, 'sim_mujoco')
from so101_liftcube import LiftCubeEnv
env = LiftCubeEnv(render=False); env.reset(seed=0)
print('scene ok: nq', env.model.nq, 'cameras', [env.model.camera(i).name for i in range(env.model.ncam)])"
info "done. Next: qsub pbs/mujoco_generate.pbs"
