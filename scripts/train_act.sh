#!/usr/bin/env bash
# ACT for the real SO-101 cube pickup on NUS Hopper: fine-tune a pretrained ACT (default
# omiya239532/so101_act_cotrain) or train one from scratch, on a validated LeRobot dataset.
# Submit with pbs/train_act.pbs; also runs as is in an interactive `qsub -I` shell.
#
#   qsub pbs/train_act.pbs                                                        # 1000-step smoke test
#   qsub -v DATASET_REPO_ID=<hf_user>/so101_redcube_lift,STEPS=10000 pbs/train_act.pbs
#   qsub -v DATASET_REPO_ID=<hf_user>/so101_redcube_lift,PRETRAINED=none,STEPS=30000 pbs/train_act.pbs
#
# Steps, each failing the job with a clear message:
#   1/5 environment: GPU (nvidia-smi + a CUDA matmul), LeRobot/torch versions, PyAV AV1 decoder. The env is
#       the LeRobot 0.4.2 venv on pytorch_2.7.0_cuda_12.8.sif from sim/setup_hpc_train.sh (qsub pbs/setup_act_env.pbs);
#       the container's CUDA torch is used, nothing is installed here.
#   2/5 PRETRAINED -> $SCRATCH/checkpoints/act/<name>, made loadable by LeRobot 0.4.2 (scripts/prepare_act_checkpoint.py)
#   3/5 dataset at DATASET_ROOT (downloaded first if missing and the node has internet), validated by
#       scripts/check_act_dataset.py --strict against that checkpoint (camera keys, fps, dims, units, decoding)
#   4/5 lerobot-train via sim/train_act_hpc.sh: checkpoint every SAVE_FREQ steps, an existing OUTPUT_DIR is resumed
#   5/5 the last checkpoint loaded and run on a dummy observation (sim/tools/test_act_checkpoint.py)
#
# Settings (environment variables) [default]:
#   DATASET_REPO_ID   [omiya239532/so101_cube_dr]  smoke-test default (Genesis sim pick-and-place, same format as
#                     the checkpoint). For the real robot use your own recording, see README "ACT".
#   DATASET_ROOT      [$SCRATCH/datasets/lerobot/$DATASET_REPO_ID]
#   PRETRAINED        [omiya239532/so101_act_cotrain]  HF repo id or local checkpoint dir; "none" = from scratch
#   STEPS [1000]   BATCH_SIZE [8]   LR [1e-5] (ACT default and the pretrained run's)   SAVE_FREQ [5000]   LOG_FREQ [100]
#   OUTPUT_DIR        [outputs/act_<dataset name>_<finetune|scratch>]; resumed up to STEPS if it has a checkpoint
#   IMAGE_TRANSFORMS  [true]  colour/sharpness/affine augmentation, as the pretrained run used
#   VIDEO_BACKEND     [pyav]   KEEP_CHECKPOINTS [0 = keep every SAVE_FREQ checkpoint]   WANDB_ENABLE [false]
#   CHECKPOINT_FPS    [30]  fps of the pretrained checkpoint's data (so101_act_cotrain: 30), for the fps check
#   ALLOW_INCOMPATIBLE [0]  1 = fine-tune even if the dataset check finds a mismatch with the checkpoint
# Extra arguments go straight to lerobot-train (e.g. --policy.chunk_size=50 for a scratch run).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"

info() { printf '\033[1;34m[train-act]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[train-act] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$REPO_ROOT/$1" ;; esac; }
inenv() { bash "$REPO_ROOT/sim/hpc_env.sh" "$@"; }   # container + venv + caches, HF_HUB_OFFLINE=1 by default
online() { curl -sS -o /dev/null -m 10 https://huggingface.co 2>/dev/null; }

DATASET_REPO_ID="${DATASET_REPO_ID:-omiya239532/so101_cube_dr}"
DATASET_ROOT="$(abspath "${DATASET_ROOT:-$SCRATCH/datasets/lerobot/$DATASET_REPO_ID}")"
PRETRAINED="${PRETRAINED:-omiya239532/so101_act_cotrain}"
STEPS="${STEPS:-1000}"
LR="${LR:-1e-5}"
IMAGE_TRANSFORMS="${IMAGE_TRANSFORMS:-true}"
CHECKPOINT_FPS="${CHECKPOINT_FPS:-30}"
ALLOW_INCOMPATIBLE="${ALLOW_INCOMPATIBLE:-0}"
DS_NAME="$(basename "$DATASET_REPO_ID")"
MODE=finetune
[ "$PRETRAINED" = "none" ] && MODE=scratch
OUTPUT_DIR="$(abspath "${OUTPUT_DIR:-outputs/act_${DS_NAME}_${MODE}}")"
export DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR
export BATCH_SIZE="${BATCH_SIZE:-8}" SAVE_FREQ="${SAVE_FREQ:-5000}" LOG_FREQ="${LOG_FREQ:-100}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}" KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
export WANDB_ENABLE="${WANDB_ENABLE:-false}" JOB_NAME="${JOB_NAME:-act_${DS_NAME}_${MODE}}"
[[ "$STEPS" =~ ^[0-9]+$ ]] || die "STEPS must be an integer, got '$STEPS'"
OFFLINE=1
online && OFFLINE=0

info "$(date)  host $(hostname)  job ${PBS_JOBID:-interactive}"
info "dataset $DATASET_REPO_ID ($DATASET_ROOT)"
info "mode $MODE$([ "$MODE" = finetune ] && echo " from $PRETRAINED"), $STEPS steps, batch $BATCH_SIZE, lr $LR, save every $SAVE_FREQ -> $OUTPUT_DIR"
info "internet: $([ "$OFFLINE" = 0 ] && echo yes || echo 'no (everything must already be on /scratch)')"

# ---- 1/5 environment ------------------------------------------------------------------------
info "==== 1/5 environment"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv \
    || die "nvidia-smi failed: no GPU here. Submit with qsub (pbs/train_act.pbs), not on the login node."
inenv python - <<'EOF' || die "environment check failed (above). Build/repair the venv: qsub pbs/setup_act_env.pbs"
import importlib.metadata as md
import av, torch
print(f"lerobot {md.version('lerobot')} | torch {torch.__version__} (CUDA {torch.version.cuda}) | av {av.__version__}")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False (no --nv / no GPU)"
x = torch.randn(2048, 2048, device="cuda"); (x @ x).sum().item()
print(f"GPU {torch.cuda.get_device_name(0)}: CUDA matmul ok")
decoders = []
for name in ("libdav1d", "libaom-av1", "av1"):
    try:
        av.codec.Codec(name, "r"); decoders.append(name)
    except Exception:
        pass
assert decoders, "PyAV has no AV1 decoder; LeRobot v3 datasets are AV1 video"
print(f"AV1 decoders: {decoders}")
import lerobot.scripts.lerobot_train, lerobot.policies.act.modeling_act  # noqa: F401
EOF

# ---- 2/5 pretrained checkpoint ------------------------------------------------------------
PREP_DIR=""
if [ "$MODE" = finetune ]; then
    info "==== 2/5 pretrained checkpoint $PRETRAINED"
    SRC="$PRETRAINED"   # HF repo id, unless it names a local dir
    if [ -e "$PRETRAINED" ] || [ -e "$REPO_ROOT/$PRETRAINED" ]; then SRC="$(abspath "$PRETRAINED")"; fi
    PREP_DIR="${PREPARED_DIR:-$SCRATCH/checkpoints/act/$(echo "$PRETRAINED" | sed 's#^/##; s#[/ ]#_#g')}"
    inenv env HF_HUB_OFFLINE="$OFFLINE" python scripts/prepare_act_checkpoint.py "$SRC" --out "$PREP_DIR" \
        || die "could not prepare $PRETRAINED. Offline node and not cached? Run this once on the login node:
    bash sim/hpc_env.sh env HF_HUB_OFFLINE=0 python scripts/prepare_act_checkpoint.py $PRETRAINED --out $PREP_DIR"
else
    info "==== 2/5 no pretrained checkpoint: ACT from scratch (ResNet18 ImageNet backbone from \$TORCH_HOME)"
fi

# ---- 3/5 dataset ----------------------------------------------------------------------------
info "==== 3/5 dataset check"
if [ ! -f "$DATASET_ROOT/meta/info.json" ] || [ ! -d "$DATASET_ROOT/videos" ]; then
    [ "$OFFLINE" = 0 ] || die "no dataset at $DATASET_ROOT and no internet here. Download it on the login node:
    bash sim/hpc_env.sh env HF_HUB_OFFLINE=0 python scripts/check_act_dataset.py --repo-id $DATASET_REPO_ID --root $DATASET_ROOT"
fi
CHECK=(--repo-id "$DATASET_REPO_ID" --root "$DATASET_ROOT" --video-backend "$VIDEO_BACKEND")
if [ "$MODE" = finetune ]; then
    CHECK+=(--checkpoint "$PREP_DIR" --checkpoint-fps "$CHECKPOINT_FPS")
    [ "$ALLOW_INCOMPATIBLE" = 1 ] || CHECK+=(--strict)
fi
inenv env HF_HUB_OFFLINE="$OFFLINE" python scripts/check_act_dataset.py "${CHECK[@]}" \
    || die "dataset check failed (above). ALLOW_INCOMPATIBLE=1 overrides only a checkpoint mismatch."

# ---- 4/5 train --------------------------------------------------------------------------------
info "==== 4/5 lerobot-train ($(date))"
EXTRA=()
if [ -f "$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" ]; then
    export RESUME=1
    info "resuming $OUTPUT_DIR up to step $STEPS (its saved config, lr and data are kept)"
elif [ -e "$OUTPUT_DIR" ]; then
    die "$OUTPUT_DIR exists but holds no checkpoint (a run that died before its first save?). Remove it or set OUTPUT_DIR."
else
    EXTRA=(--policy.optimizer_lr="$LR" --policy.optimizer_lr_backbone="$LR"
           --dataset.image_transforms.enable="$IMAGE_TRANSFORMS")
    [ -n "$PREP_DIR" ] && export POLICY_PATH="$PREP_DIR"
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
bash "$REPO_ROOT/sim/train_act_hpc.sh" "$STEPS" ${EXTRA[@]+"${EXTRA[@]}"} "$@"

# ---- 5/5 load like the server -------------------------------------------------------------
info "==== 5/5 load $OUTPUT_DIR/checkpoints/last/pretrained_model"
inenv python sim/tools/test_act_checkpoint.py "$OUTPUT_DIR/checkpoints/last/pretrained_model" \
    || die "the trained checkpoint does not load"
info "done ($(date)). Copy $OUTPUT_DIR/checkpoints/last/pretrained_model to the laptop and run:"
info "  python local/run_act.py --checkpoint <that dir>"
