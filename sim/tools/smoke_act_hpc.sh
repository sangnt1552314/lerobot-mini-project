#!/usr/bin/env bash
# End-to-end plumbing test of the HPC ACT training path (run on a GPU node: pbs/test_act_h200.pbs or qsub -I).
#   env OK -> dataset found -> 100 training steps (checkpoints at 50, 100) -> files eval needs exist
#   -> refuses to overwrite -> resume to 150 -> LocalACTPolicy loads the checkpoint, (6,) action.
#
#   bash sim/tools/smoke_act_hpc.sh [WORK_DIR]
#
# Uses the SYNTHETIC dataset by default (DATASET_REPO_ID/DATASET_ROOT override it). Training on it
# tests plumbing only; the checkpoint means nothing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
WORK="${1:-$SCRATCH/tmp/act_smoke_${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/synthetic_liftcube_smoke}"
export DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/datasets/lerobot/$DATASET_REPO_ID}"
export OUTPUT_DIR="$WORK/act_smoke"
export BATCH_SIZE=8 SAVE_FREQ=50 LOG_FREQ=10
export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

step() { printf '\n\033[1;35m==== SMOKE: %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m==== SMOKE FAILED: %s\033[0m\n' "$*" >&2; exit 1; }
mkdir -p "$WORK"

step "1/6 environment"
bash "$REPO_ROOT/sim/hpc_env.sh" python -c "
import torch, lerobot, importlib.metadata as m
assert torch.cuda.is_available(), 'no CUDA'
print('lerobot', m.version('lerobot'), '| torch', torch.__version__, '|', torch.cuda.get_device_name(0))
assert m.version('lerobot') == '0.4.2'" || fail "environment"

step "2/6 dataset $DATASET_ROOT"
[ -f "$DATASET_ROOT/meta/info.json" ] || fail "no dataset at $DATASET_ROOT"
bash "$REPO_ROOT/sim/hpc_env.sh" python sim/tools/inspect_dataset.py --repo-id "$DATASET_REPO_ID" \
    --root "$DATASET_ROOT" --video-backend "$VIDEO_BACKEND" --out-dir "$WORK/inspect" || fail "inspect_dataset"

step "3/6 train 100 steps (W&B offline so the smoke test does not create online runs)"
bash "$REPO_ROOT/sim/train_act_hpc.sh" 100 --wandb.mode=offline || fail "training"
for s in 000050 000100; do
    d="$OUTPUT_DIR/checkpoints/$s/pretrained_model"
    ls "$d"/config.json "$d"/model.safetensors "$d"/train_config.json >/dev/null || fail "missing files in $d"
    ls "$d"/policy_preprocessor*.json "$d"/policy_postprocessor*.json >/dev/null || fail "missing processor files in $d"
done
ls -l "$OUTPUT_DIR/checkpoints/" "$OUTPUT_DIR/checkpoints/000100/pretrained_model/"

step "4/6 refuses to overwrite an existing output dir"
if bash "$REPO_ROOT/sim/train_act_hpc.sh" 100 >"$WORK/overwrite.log" 2>&1; then fail "overwrote $OUTPUT_DIR"; fi
grep -q "already exists" "$WORK/overwrite.log" && echo "refused, as expected: $(grep -m1 'already exists' "$WORK/overwrite.log")"

step "5/6 resume to 150 steps"
RESUME=1 bash "$REPO_ROOT/sim/train_act_hpc.sh" 150 --wandb.mode=offline || fail "resume"
d="$OUTPUT_DIR/checkpoints/000150"
[ -f "$d/pretrained_model/model.safetensors" ] || fail "no checkpoint at $d"
grep -q '"step": 150' "$d/training_state/training_step.json" || fail "training_step.json is not 150"
echo "last -> $(readlink "$OUTPUT_DIR/checkpoints/last"); $(cat "$d/training_state/training_step.json")"

step "6/6 load with LocalACTPolicy + forward pass on a dummy observation"
bash "$REPO_ROOT/sim/hpc_env.sh" python sim/tools/test_act_checkpoint.py "$OUTPUT_DIR/checkpoints/last/pretrained_model" \
    || fail "LocalACTPolicy"

printf '\n\033[1;32m==== SMOKE PASSED\033[0m (work dir %s)\n' "$WORK"
