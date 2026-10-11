#!/usr/bin/env bash
# One-time install of the simulation stack into its own conda env (default: leisaac_sim).
# Leaves the real-robot / MolmoAct environments untouched. SIMULATION ONLY.
#
#   bash sim/setup.sh
#
# Versions follow LeIsaac's installation guide at LEISAAC_REF:
#   Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0, PyTorch 2.7.0 (CUDA 12.8), LeRobot 0.4.2

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env

info "1/7 Checking platform"
require_linux_nvidia
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/      GPU: /'
for tool in conda git curl unzip; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' is required but not installed."
done

info "2/7 Conda env '$SIM_CONDA_ENV' (Python 3.11)"
if conda env list | awk '{print $1}' | grep -qx "$SIM_CONDA_ENV"; then
    info "      reusing existing env"
else
    conda create -y -n "$SIM_CONDA_ENV" python=3.11
fi
activate_env
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' \
    || die "env '$SIM_CONDA_ENV' is not Python 3.11; remove it (conda env remove -n $SIM_CONDA_ENV) and rerun."

info "3/7 CUDA toolkit 12.8, FFmpeg, PyTorch 2.7.0 (cu128)"
conda install -y -c "nvidia/label/cuda-12.8.1" cuda-toolkit
conda install -y -c conda-forge ffmpeg=7.1.1   # LeRobot video decoding (torchcodec) needs FFmpeg libs
pip install --upgrade pip
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

info "4/7 LeIsaac source at $LEISAAC_ROOT"
if [ ! -d "$LEISAAC_ROOT/.git" ]; then
    git clone https://github.com/LightwheelAI/leisaac.git "$LEISAAC_ROOT"
fi
case "$LEISAAC_ROOT" in
    "$SIM_DIR"/third_party/*)
        git -C "$LEISAAC_ROOT" fetch --quiet origin
        git -C "$LEISAAC_ROOT" checkout --quiet "$LEISAAC_REF"
        info "      checked out $LEISAAC_REF" ;;
    *)
        warn "Using your own LeIsaac checkout at $(git -C "$LEISAAC_ROOT" rev-parse --short HEAD); tested with ${LEISAAC_REF:0:7}." ;;
esac

info "5/7 Installing Isaac Sim 5.1 + Isaac Lab 2.3 (via leisaac[isaaclab]) and LeRobot 0.4.2"
pip install -e "$LEISAAC_ROOT/source/leisaac[isaaclab]" --extra-index-url https://pypi.nvidia.com
pip install -e "$LEISAAC_ROOT/source/leisaac[lerobot]"
pip install numpy==1.26.0   # LeIsaac docs: Isaac Sim needs numpy 1.26 after installing lerobot
pip check || warn "pip check reported conflicts (above). Isaac Sim's pins often trigger this; continue unless imports fail."

info "6/7 Assets: SO-101 follower + Table with Cube"
mkdir -p "$LEISAAC_ASSETS_ROOT/robots" "$LEISAAC_ASSETS_ROOT/scenes"
if [ ! -f "$LEISAAC_ASSETS_ROOT/robots/so101_follower.usd" ]; then
    curl -fL --retry 3 -o "$LEISAAC_ASSETS_ROOT/robots/so101_follower.usd" \
        https://huggingface.co/LightwheelAI/leisaac_env/resolve/main/assets/robots/so101_follower.usd
fi
if [ ! -f "$LEISAAC_ASSETS_ROOT/scenes/table_with_cube/scene.usd" ]; then
    tmp_zip="$(mktemp -d)/table_with_cube.zip"
    curl -fL --retry 3 -o "$tmp_zip" https://github.com/LightwheelAI/leisaac/releases/download/v0.1.2/table_with_cube.zip
    unzip -q -o "$tmp_zip" -d "$LEISAAC_ASSETS_ROOT/scenes"
    rm -f "$tmp_zip"
fi
require_assets
info "      assets OK in $LEISAAC_ASSETS_ROOT"

info "7/7 Validating imports"
PYTHONPATH="$SIM_DIR${PYTHONPATH:+:$PYTHONPATH}" python - <<'EOF'
import importlib.metadata as md
import torch, numpy, gymnasium, lerobot
print(f"      torch {torch.__version__}  cuda={torch.cuda.is_available()}  ({torch.version.cuda})")
print(f"      numpy {numpy.__version__}  lerobot {md.version('lerobot')}  gymnasium {gymnasium.__version__}")
print(f"      isaacsim {md.version('isaacsim')}  isaaclab {md.version('isaaclab')}  leisaac {md.version('leisaac')}")
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
from isaaclab.app import AppLauncher  # noqa: F401
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
try:
    import torchcodec.decoders  # noqa: F401  needs FFmpeg shared libs; LeRobot decodes videos with it
except Exception as e:
    raise SystemExit(f"torchcodec cannot load FFmpeg ({e}). Run: conda install -c conda-forge ffmpeg=7.1.1")
import leisaac_ext
assert leisaac_ext.TASK_ID in gymnasium.registry, "task registration failed"
from leisaac_ext.config import load_config
load_config()
print(f"      registered {leisaac_ext.TASK_ID}; config OK")
EOF

info "Setup complete. Next (first Isaac launch compiles shaders and can take several minutes):"
echo
echo "    bash sim/preview.sh"
echo
