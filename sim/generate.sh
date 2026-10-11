#!/usr/bin/env bash
# Record N *successful* scripted demonstrations straight into a LeRobot dataset, then validate it.
#
#   bash sim/generate.sh          # NUM_DEMOS from sim/.env (default 100)
#   bash sim/generate.sh 20       # quick first dataset
#   bash sim/generate.sh 500
#   GENERATE_GUI=1 bash sim/generate.sh 20      # watch while recording (slower)
#
# Extra arguments after N go to sim/scripts/generate.py (e.g. --seed 1).
# The dataset goes to $HF_LEROBOT_HOME/$HF_USER/$SIM_DATASET_NAME. SIMULATION ONLY.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env

N="${1:-$NUM_DEMOS}"
require_int "number of demos" "$N"
[ "$N" -gt 0 ] || die "number of demos must be > 0"
shift $(( $# > 0 ? 1 : 0 ))

require_linux_nvidia
activate_env
require_assets

if [ -e "$DATASET_ROOT" ]; then
    die "Dataset already exists: $DATASET_ROOT
    Delete it (rm -rf \"$DATASET_ROOT\") or set a different SIM_DATASET_NAME in sim/.env."
fi

HEADLESS=(--headless)
[ "${GENERATE_GUI:-0}" = "1" ] && HEADLESS=()

info "Recording $N successful demos of $SIM_TASK -> $DATASET_REPO_ID ($DATASET_FPS fps)"
info "Dataset dir: $DATASET_ROOT"
python "$SIM_DIR/scripts/generate.py" \
    --task "$SIM_TASK" \
    "${HEADLESS[@]}" \
    --enable_cameras \
    --device "$DEVICE" \
    --record \
    --num_demos "$N" \
    --lerobot_dataset_repo_id "$DATASET_REPO_ID" \
    --lerobot_dataset_fps "$DATASET_FPS" \
    "$@" \
    || warn "generate.py exited non-zero; validating whatever was written"

info "Validating dataset"
python "$SIM_DIR/tools/inspect_dataset.py" --repo-id "$DATASET_REPO_ID" --root "$DATASET_ROOT" --min-episodes "$N"
info "Done. Next: bash sim/train_act.sh"
