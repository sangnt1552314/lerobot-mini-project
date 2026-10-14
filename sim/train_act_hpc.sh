#!/usr/bin/env bash
# Train LeRobot 0.4.2 ACT on one H200 of the NUS HPC (data from sim_mujoco/scripts/generate.py; pbs/mujoco_pipeline.pbs runs both).
# Works inside a PBS job (pbs/train_act_h200.pbs) and in an interactive `qsub -I` shell; when started
# outside the Singularity image it re-runs itself inside it. Env built by sim/setup_hpc_train.sh.
#
#   bash sim/train_act_hpc.sh                    # STEPS (default 20000)
#   bash sim/train_act_hpc.sh 500                # short run
#   RESUME=1 bash sim/train_act_hpc.sh 40000     # continue OUTPUT_DIR up to 40000 steps
#   BATCH_SIZE=64 MIXED_PRECISION=bf16 bash sim/train_act_hpc.sh 2500
#
# Settings (environment variables; defaults in brackets):
#   DATASET_REPO_ID  [local/so101_liftcube_dualcam_sim]
#   DATASET_ROOT     [/scratch/e1583535/datasets/lerobot/$DATASET_REPO_ID]
#   OUTPUT_DIR       [outputs/act_liftcube_h200]  (relative = repo root). Never overwritten.
#   JOB_NAME [act_so101_liftcube_sim]  BATCH_SIZE [8]  SAVE_FREQ [5000]  LOG_FREQ [200]
#   NUM_WORKERS      [NCPUS-2: PBS gives 12 CPUs per GPU, so 10]
#   VIDEO_BACKEND    [pyav] or torchcodec (uses the FFmpeg from setup_hpc_train.sh)
#   MIXED_PRECISION  [no] or bf16 (accelerate autocast; LeRobot 0.4.2's trainer reads it from
#                    ACCELERATE_MIXED_PRECISION; policy.use_amp is ignored by lerobot-train)
#   STAGE_DATASET    [0] or 1: copy the dataset to node-local RAM disk (/dev/shm) first
#   WANDB_ENABLE [true]  WANDB_ENTITY  WANDB_PROJECT [cs6283]  WANDB_API_KEY
#                    (read from the gitignored repo-root .env if present; offline automatically without
#                    credentials or when api.wandb.ai is unreachable)
#   USAGE_CSV        if set, sample GPU util / CPU cores of the run into this CSV
#   KEEP_CHECKPOINTS [2] keep only the newest N checkpoints/<step> dirs (0 = keep all). LeRobot has no
#                    such option, so older ones are deleted while training runs and once at the end;
#                    the one `last` points to is never deleted.
# Extra arguments after STEPS go straight to lerobot-train. SIMULATION DATA ONLY.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
IMAGE="${IMAGE:-/app1/common/singularity-img/hopper/pytorch/pytorch_2.7.0_cuda_12.8.sif}"
VENV="${VENV:-$SCRATCH/virtualenvs/lerobot042-act}"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-$SCRATCH/opt/ffmpeg-7.1.1-conda}"

info() { printf '\033[1;34m[train-hpc]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[train-hpc] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[train-hpc] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$REPO_ROOT/$1" ;; esac; }

# Secrets/config (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT) from the gitignored .env, as in the
# robocolosseum PBS scripts. Variables already set in the environment win.
if [ -f "$REPO_ROOT/.env" ]; then
    while IFS='=' read -r key val; do
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [ -n "${!key+x}" ] || export "$key=$val"
    done < "$REPO_ROOT/.env"
fi

STEPS="${1:-${STEPS:-20000}}"
[[ "$STEPS" =~ ^[0-9]+$ ]] || die "steps must be a non-negative integer, got '$STEPS'"
shift $(( $# > 0 ? 1 : 0 ))

DATASET_REPO_ID="${DATASET_REPO_ID:-local/so101_liftcube_dualcam_sim}"
DATASET_ROOT="$(abspath "${DATASET_ROOT:-$SCRATCH/datasets/lerobot/$DATASET_REPO_ID}")"
OUTPUT_DIR="$(abspath "${OUTPUT_DIR:-outputs/act_liftcube_h200}")"
JOB_NAME="${JOB_NAME:-act_so101_liftcube_sim}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"
NCPUS="${NCPUS:-$(nproc)}"
NUM_WORKERS="${NUM_WORKERS:-$(( NCPUS > 3 ? NCPUS - 2 : 1 ))}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
STAGE_DATASET="${STAGE_DATASET:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-cs6283}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RESUME="${RESUME:-0}"
USAGE_CSV="${USAGE_CSV:-}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
[[ "$KEEP_CHECKPOINTS" =~ ^[0-9]+$ ]] || die "KEEP_CHECKPOINTS must be a non-negative integer"

# ---- run inside the container ---------------------------------------------------------------
if [ -z "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then
    [ -f "$IMAGE" ] || die "Singularity image not found: $IMAGE"
    command -v singularity >/dev/null 2>&1 || { set +u; module load singularity 2>/dev/null || true; set -u; }
    command -v singularity >/dev/null 2>&1 || die "singularity not found (module load singularity)."
    FWD=()
    for v in SCRATCH_ROOT VENV FFMPEG_PREFIX DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR JOB_NAME BATCH_SIZE \
             SAVE_FREQ LOG_FREQ NCPUS NUM_WORKERS VIDEO_BACKEND MIXED_PRECISION STAGE_DATASET STAGE_DIR \
             WANDB_ENABLE WANDB_ENTITY WANDB_PROJECT WANDB_API_KEY RESUME USAGE_CSV KEEP_CHECKPOINTS PBS_JOBID; do
        [ -n "${!v+x}" ] && FWD+=(--env "$v=${!v}")
    done
    FWD+=(--env "SCRATCH_ROOT=$SCRATCH" --env "NCPUS=$NCPUS" --env "NUM_WORKERS=$NUM_WORKERS")
    exec singularity exec -e --nv "${FWD[@]}" "$IMAGE" bash --norc "$REPO_ROOT/sim/train_act_hpc.sh" "$STEPS" "$@"
fi

# ---- environment (same cache conventions as pbs/run_server.pbs) -------------------------------
[ -f "$VENV/bin/activate" ] || die "No venv at $VENV. Run: qsub pbs/setup_act_env.pbs"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HOME="$SCRATCH/cache"
export HF_LEROBOT_HOME="$SCRATCH/datasets/lerobot"
export HF_HUB_OFFLINE=1                       # local dataset; nothing to fetch from the Hub
export TORCH_HOME="$SCRATCH/cache/torch"      # ResNet18 ImageNet weights pre-downloaded by setup
export TMPDIR="$SCRATCH/tmp" TEMP="$SCRATCH/tmp" TMP="$SCRATCH/tmp"
export ACCELERATE_MIXED_PRECISION="$MIXED_PRECISION"
mkdir -p "$TMPDIR"

case "$MIXED_PRECISION" in no|bf16|fp16) ;; *) die "MIXED_PRECISION must be no, bf16 or fp16" ;; esac
case "$VIDEO_BACKEND" in
    pyav) ;;
    torchcodec)
        [ -f "$FFMPEG_PREFIX/lib/libavcodec.so.61" ] || die "torchcodec needs FFmpeg at $FFMPEG_PREFIX (sim/setup_hpc_train.sh) or use VIDEO_BACKEND=pyav"
        export LD_LIBRARY_PATH="$FFMPEG_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    *) die "VIDEO_BACKEND must be pyav or torchcodec" ;;
esac

[ -f "$DATASET_ROOT/meta/info.json" ] || die "No LeRobot dataset at $DATASET_ROOT (generate it first: pbs/mujoco_pipeline.pbs)."
[ -f "$DATASET_ROOT/SYNTHETIC_DATA_NOT_REAL.txt" ] && warn "$DATASET_ROOT is the SYNTHETIC test dataset; this checkpoint is for plumbing tests only."

TRAIN_ROOT="$DATASET_ROOT"
STAGED=""
if [ "$STAGE_DATASET" = "1" ]; then
    STAGED="${STAGE_DIR:-/dev/shm/${USER:-$(id -un)}-act-${PBS_JOBID:-$$}}"
    info "Staging dataset to $STAGED ($(du -sh "$DATASET_ROOT" | cut -f1))"
    mkdir -p "$STAGED/$DATASET_REPO_ID"
    cp -a "$DATASET_ROOT/." "$STAGED/$DATASET_REPO_ID/"
    TRAIN_ROOT="$STAGED/$DATASET_REPO_ID"
    trap 'rm -rf "$STAGED"' EXIT
fi

WANDB_ARGS=(--wandb.enable="$WANDB_ENABLE")
if [ "$WANDB_ENABLE" = "true" ]; then
    WANDB_ARGS+=(--wandb.project="$WANDB_PROJECT")
    [ -n "$WANDB_ENTITY" ] && WANDB_ARGS+=(--wandb.entity="$WANDB_ENTITY")
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
        warn "No W&B credentials; logging offline (sync later: wandb sync $OUTPUT_DIR/wandb/offline-run-*)."
        WANDB_ARGS+=(--wandb.mode=offline)
    elif ! curl -sS -o /dev/null -m 10 https://api.wandb.ai; then
        warn "api.wandb.ai unreachable from $(hostname); logging offline."
        WANDB_ARGS+=(--wandb.mode=offline)
    fi
fi

# Settings that only affect speed / logging; safe to change when resuming.
RUNTIME_ARGS=(
    --dataset.root="$TRAIN_ROOT"
    --dataset.video_backend="$VIDEO_BACKEND"
    --num_workers="$NUM_WORKERS"
    --steps="$STEPS"
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
)

info "host $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1), NCPUS=$NCPUS"
info "batch $([ "$RESUME" = "1" ] && echo "from checkpoint" || echo "$BATCH_SIZE"), workers $NUM_WORKERS, video $VIDEO_BACKEND, mixed precision $MIXED_PRECISION, data $TRAIN_ROOT"

if [ -e "$OUTPUT_DIR" ]; then
    [ "$RESUME" = "1" ] || die "$OUTPUT_DIR already exists. Use RESUME=1 to continue it, or set OUTPUT_DIR to a new path."
    CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
    [ -f "$CONFIG" ] || die "Cannot resume: $CONFIG not found."
    info "Resuming $OUTPUT_DIR from step $(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["step"])' "$OUTPUT_DIR/checkpoints/last/training_state/training_step.json") up to $STEPS"
    CMD=(lerobot-train --config_path="$CONFIG" --resume=true "${RUNTIME_ARGS[@]}" "${WANDB_ARGS[@]}" "$@")
else
    [ "$RESUME" = "1" ] && die "RESUME=1 but $OUTPUT_DIR does not exist."
    info "Training ACT on $DATASET_REPO_ID for $STEPS steps -> $OUTPUT_DIR"
    CMD=(lerobot-train
        --dataset.repo_id="$DATASET_REPO_ID"
        --policy.type=act
        --policy.device=cuda
        --policy.push_to_hub=false
        --output_dir="$OUTPUT_DIR"
        --job_name="$JOB_NAME"
        --batch_size="$BATCH_SIZE"
        "${RUNTIME_ARGS[@]}"
        "${WANDB_ARGS[@]}"
        "$@")
fi

# Delete all but the newest KEEP_CHECKPOINTS checkpoints/<step> dirs. The newest dir may still be
# being written, and the dir `last` points to is skipped, so a resumable checkpoint always survives.
prune_checkpoints() {
    local dir="$OUTPUT_DIR/checkpoints" last="" steps=() i
    [ "$KEEP_CHECKPOINTS" -gt 0 ] && [ -d "$dir" ] || return 0
    [ -L "$dir/last" ] && last="$(basename "$(readlink "$dir/last")")"
    mapfile -t steps < <(find "$dir" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' | sort -n)
    for (( i = 0; i < ${#steps[@]} - KEEP_CHECKPOINTS; i++ )); do
        [ "${steps[i]}" = "$last" ] && continue
        info "Removing old checkpoint ${steps[i]} (keeping newest $KEEP_CHECKPOINTS)"
        rm -rf "${dir:?}/${steps[i]}"
    done
}

echo "[train-hpc] $(printf '%q ' "${CMD[@]}")"
"${CMD[@]}" &
TRAIN_PID=$!
if [ -n "$USAGE_CSV" ]; then
    python "$REPO_ROOT/sim/tools/monitor_usage.py" --pid "$TRAIN_PID" --out "$USAGE_CSV" &
fi
if [ "$KEEP_CHECKPOINTS" -gt 0 ]; then
    ( while kill -0 "$TRAIN_PID" 2>/dev/null; do prune_checkpoints; sleep 60; done ) &
fi
status=0
wait "$TRAIN_PID" || status=$?
wait || true
prune_checkpoints
[ "$status" = "0" ] || die "lerobot-train exited with status $status"

if [[ " $* " != *" --save_checkpoint=false "* ]]; then
    CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"
    [ -f "$CKPT/config.json" ] || die "Training finished but $CKPT/config.json is missing."
    info "Checkpoint: $CKPT"
fi
