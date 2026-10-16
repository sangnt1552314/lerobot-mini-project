#!/usr/bin/env bash
# Run a command in the squint env (sim_squint/setup_env.sh) (in the current directory: squint writes runs/ there), with SAPIEN's Vulkan pointed at
# the NVIDIA driver (the nodes ship no nvidia_icd.json).   bash sim_squint/env.sh python sim_squint/train.py ...
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
VENV="${VENV:-$SCRATCH/virtualenvs/squint}"
# NVIDIA's Vulkan driver lives in libGLX_nvidia (bound by --nv in the container, /lib64 on the nodes); point the loader at
# it by absolute path, whatever ICD files the image or node ship (a stale nvidia_icd.json gives ErrorIncompatibleDriver)
NV_LIB=""
for d in /.singularity.d/libs /usr/lib64 /lib64 /usr/lib/x86_64-linux-gnu $(echo "${LD_LIBRARY_PATH:-}" | tr ':' ' '); do
    for f in libGLX_nvidia.so.0 libEGL_nvidia.so.0; do
        if [ -z "$NV_LIB" ] && [ -e "$d/$f" ]; then NV_LIB="$d/$f"; fi
    done
done
[ -n "$NV_LIB" ] || NV_LIB="$(ldconfig -p 2>/dev/null | awk '/libGLX_nvidia.so.0 .*x86-64/ {print $NF; exit}')"
if [ -n "$NV_LIB" ]; then
    ICD="$SCRATCH/opt/vulkan_shim/nvidia_icd_$(echo "$NV_LIB" | tr '/' '_').json"
    mkdir -p "$(dirname "$ICD")"
    echo "{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"$NV_LIB\", \"api_version\": \"1.3.0\"}}" > "$ICD"
    export VK_ICD_FILENAMES="$ICD" VK_DRIVER_FILES="$ICD"
else
    echo "sim_squint/env.sh: no libGLX_nvidia.so.0 found; Vulkan rendering will fail" >&2
fi
# libGLX_nvidia needs libGLdispatch, which the container lacks: sim_mujoco/setup_hpc.sh's copy in egl_shim (+ its EGL
# vendor file); libX11 for sapien from the venv (conda-forge xorg-libx11)
EGL_SHIM="$SCRATCH/opt/egl_shim"
[ -f "$EGL_SHIM/10_nvidia.json" ] && export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_SHIM/10_nvidia.json"
export LD_LIBRARY_PATH="$VENV/lib:$EGL_SHIM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VENV/bin:$PATH" TMPDIR="$SCRATCH/tmp" TEMP="$SCRATCH/tmp" TMP="$SCRATCH/tmp"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/cache/wandb}" MS_ASSET_DIR="${MS_ASSET_DIR:-$SCRATCH/cache/maniskill}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

exec "$@"
