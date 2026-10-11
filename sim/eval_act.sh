#!/usr/bin/env bash
# Run an ACT checkpoint in the dual-camera LiftCube sim and report the success rate.
#
#   bash sim/eval_act.sh outputs/act_liftcube/checkpoints/last/pretrained_model
#   bash sim/eval_act.sh <checkpoint> 50          # 50 episodes instead of EVAL_EPISODES
#   EVAL_GUI=1 bash sim/eval_act.sh <checkpoint>  # watch it
#
# Extra arguments after the episode count go to sim/scripts/eval_act.py. SIMULATION ONLY.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env

CKPT="${1:-$OUTPUT_DIR/checkpoints/last/pretrained_model}"
EPISODES="${2:-$EVAL_EPISODES}"
require_int "episodes" "$EPISODES"
shift $(( $# > 1 ? 2 : $# ))
CKPT="$(abspath "$CKPT")"
[ -f "$CKPT/config.json" ] || die "No checkpoint at $CKPT (expected .../checkpoints/<step>/pretrained_model)."

require_linux_nvidia
activate_env
require_assets

HEADLESS=(--headless)
STEP_HZ=0
if [ "${EVAL_GUI:-0}" = "1" ]; then HEADLESS=(); STEP_HZ="$DATASET_FPS"; fi

RESULTS="$(dirname "$CKPT")/eval_sim_$(date +%Y%m%d_%H%M%S).json"
info "Evaluating $CKPT for $EPISODES episodes"
python "$SIM_DIR/scripts/eval_act.py" \
    --task "$SIM_TASK" \
    --checkpoint "$CKPT" \
    --episodes "$EPISODES" \
    "${HEADLESS[@]}" \
    --enable_cameras \
    --device "$DEVICE" \
    --step_hz "$STEP_HZ" \
    --results_file "$RESULTS" \
    "$@"

[ -f "$RESULTS" ] || die "Evaluation did not write $RESULTS; see the log above."
info "Results: $RESULTS"
