#!/usr/bin/env bash
# One-time setup of the ACT *training-only* environment on the NUS HPC (no Isaac Sim here).
#
#   qsub pbs/setup_act_env.pbs              # as a batch job (needs internet on the compute node)
#   bash sim/setup_hpc_train.sh             # login node or `qsub -I`; re-execs itself in the container
#   FORCE=1 bash sim/setup_hpc_train.sh     # delete and rebuild the venv
#
# What it does (idempotent; finished steps are skipped):
#   1. venv at $VENV with --system-site-packages inside $IMAGE (container torch 2.7.0 / tv 0.22, cu12.8),
#      lerobot==0.4.2 installed --no-deps over sim/requirements_hpc_train.txt, torchcodec==0.4.0
#      (the torchcodec release built for torch 2.7).
#   2. FFmpeg 7.1.1 shared libs from conda-forge (with dav1d/libaom for AV1) at $FFMPEG_PREFIX,
#      via a standalone micromamba. torchcodec needs them; PyAV ships its own FFmpeg.
#   3. ResNet18 ImageNet weights (ACT's default backbone) into $TORCH_HOME, so training needs no internet.
#   4. Checks: versions, GPU, imports used by training/inference, AV1 encode + decode with pyav and
#      torchcodec. Writes a report to logs/setup_act_env/report.txt.
#
# Does not touch sim/setup.sh, the leisaac_sim conda env, or the server venv (my_lebot).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-/app1/common/singularity-img/hopper/pytorch/pytorch_2.7.0_cuda_12.8.sif}"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
VENV="${VENV:-$SCRATCH/virtualenvs/lerobot042-act}"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-$SCRATCH/opt/ffmpeg-7.1.1-conda}"
MAMBA_DIR="${MAMBA_DIR:-$SCRATCH/opt/micromamba}"
export HF_HOME="${HF_HOME:-$SCRATCH/cache}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/cache/torch}"
export TMPDIR="${TMPDIR_OVERRIDE:-$SCRATCH/tmp}" TEMP="$SCRATCH/tmp" TMP="$SCRATCH/tmp"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/pip-cache}"
LOG_DIR="$REPO_ROOT/logs/setup_act_env"

info() { printf '\033[1;34m[setup-hpc]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup-hpc] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$LOG_DIR" "$SCRATCH/opt"

# ---- 0. run inside the container ------------------------------------------------------------
if [ -z "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then
    [ -f "$IMAGE" ] || die "Singularity image not found: $IMAGE"
    command -v singularity >/dev/null 2>&1 || { set +u; module load singularity 2>/dev/null || true; set -u; }
    command -v singularity >/dev/null 2>&1 || die "singularity not found (module load singularity)."
    info "Re-running inside $IMAGE"
    exec singularity exec -e --nv \
        --env FORCE="${FORCE:-0}",VENV="$VENV",FFMPEG_PREFIX="$FFMPEG_PREFIX",MAMBA_DIR="$MAMBA_DIR" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$TORCH_HOME",PIP_CACHE_DIR="$PIP_CACHE_DIR" \
        --env TMPDIR="$TMPDIR",TEMP="$TMPDIR",TMP="$TMPDIR",SCRATCH_ROOT="$SCRATCH" \
        "$IMAGE" bash --norc "$REPO_ROOT/sim/setup_hpc_train.sh"
fi

info "host $(hostname), image ${SINGULARITY_CONTAINER:-$APPTAINER_CONTAINER}"

# ---- internet check (compute nodes may not have it) ------------------------------------------
NET_OK=1
for url in https://pypi.org/simple/pip/ https://huggingface.co https://conda.anaconda.org/conda-forge/ https://download.pytorch.org; do
    if curl -sS -o /dev/null -m 15 "$url"; then info "internet ok: $url"; else info "NO ACCESS: $url"; NET_OK=0; fi
done

# ---- 1. venv + LeRobot 0.4.2 -------------------------------------------------------------------
if [ "${FORCE:-0}" = "1" ] && [ -d "$VENV" ]; then info "FORCE=1: removing $VENV"; rm -rf "$VENV"; fi

# The image's own constraint file pins packaging==23.2 (LeRobot needs >=24.2); use ours instead.
export PIP_CONSTRAINT="$REPO_ROOT/sim/constraints_hpc_train.txt"
python3 - "$PIP_CONSTRAINT" <<'EOF' || die "Container torch/torchvision/numpy/pyarrow differ from sim/constraints_hpc_train.txt; update that file for this image."
import sys, numpy, pyarrow, torch, torchvision
want = dict(l.strip().split("==") for l in open(sys.argv[1]) if "==" in l and not l.startswith("#"))
have = {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__}
# pip normalises "nv25.03" to "nv25.3"; compare the normalised forms.
from packaging.version import Version
bad = {k: (have[k], v) for k, v in want.items() if Version(have[k]) != Version(v)}
print("container:", have)
sys.exit(1 if bad else 0)
EOF

# pip runs every time (a no-op when satisfied) so an existing venv is repaired when the pins change.
if [ "$NET_OK" = "1" ]; then
    [ -d "$VENV" ] || { info "creating venv $VENV"; python3 -m venv --system-site-packages "$VENV"; }
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install -q -U "pip<26"
    info "installing LeRobot 0.4.2 dependencies"
    pip install -q -r "$REPO_ROOT/sim/requirements_hpc_train.txt"
    pip install -q --no-deps "lerobot==0.4.2" "torchcodec==0.4.0"
    deactivate
else
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import importlib.metadata as m; assert m.version('lerobot')=='0.4.2'" 2>/dev/null \
        || die "No internet here and the venv is incomplete. Run this script on the login node instead: bash sim/setup_hpc_train.sh"
    info "No internet: using the existing venv $VENV as is"
fi

# ---- 2. FFmpeg 7.1.1 with AV1 decoders, for torchcodec ----------------------------------------
if [ -f "$FFMPEG_PREFIX/lib/libavcodec.so.61" ]; then
    info "FFmpeg already at $FFMPEG_PREFIX"
elif [ "$NET_OK" = "1" ]; then
    if [ ! -x "$MAMBA_DIR/bin/micromamba" ]; then
        info "downloading micromamba to $MAMBA_DIR"
        mkdir -p "$MAMBA_DIR"
        curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MAMBA_DIR" bin/micromamba
    fi
    info "installing conda-forge ffmpeg=7.1.1 into $FFMPEG_PREFIX"
    MAMBA_ROOT_PREFIX="$MAMBA_DIR/root" "$MAMBA_DIR/bin/micromamba" create -y -q -p "$FFMPEG_PREFIX" \
        -c conda-forge --override-channels "ffmpeg=7.1.1"
else
    info "No internet: skipping FFmpeg install; training must use --dataset.video_backend=pyav."
fi

# ---- 3. ResNet18 ImageNet weights (ACT default: resnet18 / ResNet18_Weights.IMAGENET1K_V1) -----
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python - <<'EOF' || info "Could not fetch ResNet18 weights (no internet?); training will try to download them."
import torchvision
torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
import os; print("resnet18 weights cached in", os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints"))
EOF

# ---- 4. verification ---------------------------------------------------------------------------
REPORT="$LOG_DIR/report.txt"
info "verifying (report: $REPORT)"
FFMPEG_PREFIX="$FFMPEG_PREFIX" python - <<'EOF' 2>&1 | tee "$REPORT"
import ctypes, importlib, importlib.metadata as md, os, subprocess, sys, tempfile, time
from pathlib import Path

ok = True
def check(name, fn):
    global ok
    try:
        r = fn(); print(f"[ok]   {name}: {r}")
    except Exception as e:
        ok = False; print(f"[FAIL] {name}: {type(e).__name__}: {e}")

import numpy, torch, torchvision
print("python", sys.version.split()[0])
for p in ["lerobot", "torch", "torchvision", "torchcodec", "numpy", "av", "accelerate", "datasets",
          "huggingface-hub", "wandb", "draccus", "opencv-python-headless"]:
    try: print(f"  {p:24s} {md.version(p)}")
    except md.PackageNotFoundError: print(f"  {p:24s} MISSING")

from packaging.version import Version
def need(cond, value):
    if not cond: raise RuntimeError(f"got {value}")
    return value
tv, tvv = Version(torch.__version__).release, Version(torchvision.__version__).release
check("torch in LeRobot range >=2.2.1,<2.8", lambda: need((2, 2, 1) <= tv < (2, 8), torch.__version__))
check("torchvision in LeRobot range >=0.21,<0.23", lambda: need((0, 21) <= tvv < (0, 23), torchvision.__version__))
check("numpy 1.x (container torch ABI)", lambda: need(numpy.__version__.startswith("1."), numpy.__version__))

def gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False (login node / no --nv?)")
    p = torch.cuda.get_device_properties(0)
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item()
    return f"{p.name}, {p.total_memory/2**30:.0f} GiB, bf16 matmul ok"
check("GPU", gpu)

for mod in ["lerobot.scripts.lerobot_train", "lerobot.policies.act.modeling_act",
            "lerobot.policies.factory", "lerobot.utils.control_utils", "lerobot.datasets.lerobot_dataset"]:
    check(f"import {mod}", lambda m=mod: importlib.import_module(m) and "ok")
check("rerun not needed by training", lambda: "imported (unexpected but harmless)" if "rerun" in sys.modules else "not imported")

# AV1 round trip, the same codec LeRobot 0.4.2 writes (libsvtav1, yuv420p).
import av
tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
clip = tmp / "test_av1.mp4"
def encode():
    with av.open(str(clip), "w") as c:
        s = c.add_stream("libsvtav1", rate=30); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
        for i in range(30):
            img = numpy.random.default_rng(i).integers(0, 255, (480, 640, 3), dtype=numpy.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")): c.mux(pkt)
        for pkt in s.encode(): c.mux(pkt)
    return f"{clip.stat().st_size} bytes"
check("pyav encode AV1 (libsvtav1)", encode)
check("pyav decode AV1", lambda: f"{sum(1 for _ in av.open(str(clip)).decode(video=0))} frames")

ff = Path(os.environ["FFMPEG_PREFIX"]) / "lib"
def torchcodec_decode():
    # torchcodec dlopens libavutil.so.N etc.; preload the conda-forge copies by absolute path so we
    # do not have to put the whole conda lib dir (its own libstdc++, libz, ...) on LD_LIBRARY_PATH.
    out = subprocess.run([sys.executable, "-c",
        "from torchcodec.decoders import VideoDecoder as V; import sys; d=V(sys.argv[1]); "
        "print(len(d), tuple(d[0].shape), d.metadata.codec)", str(clip)],
        env={**os.environ, "LD_LIBRARY_PATH": f"{ff}:" + os.environ.get("LD_LIBRARY_PATH", "")},
        capture_output=True, text=True, timeout=300)
    if out.returncode: raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr else "failed")
    return out.stdout.strip() + f" (LD_LIBRARY_PATH={ff})"
if ff.is_dir():
    check("torchcodec decode AV1", torchcodec_decode)
else:
    print(f"[skip] torchcodec decode AV1: no FFmpeg at {ff}; use --dataset.video_backend=pyav")

netrc = Path.home() / ".netrc"
wb = bool(os.environ.get("WANDB_API_KEY")) or (netrc.is_file() and "api.wandb.ai" in netrc.read_text())
print(f"[info] W&B credentials: {'found' if wb else 'none (training will log offline)'}")
print("\nRESULT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
status=${PIPESTATUS[0]}
lerobot-train --help > "$LOG_DIR/lerobot-train-help.txt" 2>&1 || true
info "lerobot-train --help saved to $LOG_DIR/lerobot-train-help.txt"
[ "$status" = "0" ] || die "verification failed; see $REPORT"
info "done. venv: $VENV   ffmpeg: $FFMPEG_PREFIX   torch hub: $TORCH_HOME"
