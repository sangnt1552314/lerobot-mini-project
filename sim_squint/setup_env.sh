#!/usr/bin/env bash
# One-time: conda env for squint at $VENV (the pins of third_party/squint/environment.yaml; no lerobot: deploy runs on
# the laptop). Login node is fine (needs internet).   bash sim_squint/setup_env.sh
set -euxo pipefail
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
VENV="${VENV:-$SCRATCH/virtualenvs/squint}"
export TMPDIR=$SCRATCH/tmp PIP_CACHE_DIR=$SCRATCH/pip-cache MAMBA_ROOT_PREFIX=$SCRATCH/opt/micromamba/root
mkdir -p "$TMPDIR"
[ -x "$VENV/bin/python" ] || "$SCRATCH/opt/micromamba/bin/micromamba" create -y -p "$VENV" -c conda-forge python=3.10 pip
"$SCRATCH/opt/micromamba/bin/micromamba" install -y -p "$VENV" -c conda-forge xorg-libx11 xorg-libxext xorg-libxrender libvulkan-loader   # sapien needs libX11 + the Vulkan loader (not on the nodes / in the container)
"$VENV/bin/pip" install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$VENV/bin/pip" install tensordict==0.7.2 torchrl==0.7.2 mani_skill_nightly==2026.1.31.920 sapien==3.0.2 gymnasium==1.2.3 \
    numpy==2.2.6 tyro==1.0.6 tqdm wandb==0.24.2 dacite regex "setuptools<80"   # sapien 3.0.2 imports pkg_resources
# headless OpenCV only (mani_skill pulls opencv-python, which needs libGL; both own cv2/)
"$VENV/bin/pip" uninstall -y opencv-python || true
"$VENV/bin/pip" install --force-reinstall --no-deps opencv-python-headless==4.12.0.88
"$VENV/bin/python" -c "import torch, sapien, mani_skill, tensordict, torchrl, cv2; print('OK', torch.__version__, sapien.__version__)"
