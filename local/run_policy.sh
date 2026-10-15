#!/usr/bin/env bash
# Shared part of run_home.sh / run_office.sh: run local/run_act_policy.py on one arm with a sim policy.
# Those scripts set ROBOT_ID, PORT and GRIPPER_FLOOR (properties of the arm), then exec this. Any policy runs on
# any arm: its start pose and task come from the sim setup it was trained on.
#
#   bash local/run_policy.sh <policy> [control|preview]
#
# <policy>: a run dir or pretrained_model dir, as a path or just its name under outputs/ (act_office_cube).
# The policy's sim profile and object come from its training data (see run_act_policy.py --checkpoint).

set -euo pipefail

: "${ROBOT_ID:?}" "${PORT:?}" "${GRIPPER_FLOOR:?}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${SCRIPT_NAME:-$0}"

if [ $# -eq 0 ] || [ "$1" = "control" ] || [ "$1" = "preview" ]; then
    echo "usage: bash $SCRIPT <policy> [control|preview]   (default: control)" >&2
    echo "policies in outputs/: $(cd "$REPO_ROOT/outputs" 2>/dev/null && ls -d act_* 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi
POLICY="$1"
shift
for cand in "$POLICY" "$REPO_ROOT/$POLICY" "$REPO_ROOT/outputs/$POLICY"; do
    if [ -d "$cand" ]; then POLICY="$(cd "$cand" && pwd)"; break; fi
done
[ -d "$POLICY" ] || { echo "No policy dir '$POLICY' (here, in the repo, or in outputs/)" >&2; exit 1; }

MODE=control
if [ "${1:-}" = "control" ] || [ "${1:-}" = "preview" ]; then
    MODE="$1"
    shift
fi

python -c "import importlib.util as u, sys; sys.exit(not (u.find_spec('lerobot') and u.find_spec('cv2')))" 2>/dev/null \
    || { echo "$(command -v python || echo python) has no lerobot/cv2: run 'conda activate lerobot' first" >&2; exit 1; }
[ -e "$PORT" ] || { echo "Arm $ROBOT_ID not found at $PORT (plugged in? list ports: ls /dev/tty.usbmodem*; then PORT=... )" >&2; exit 1; }

echo "$ROBOT_ID on $PORT, policy $POLICY, $MODE"
cd "$REPO_ROOT/local"
exec python run_act_policy.py --mode "$MODE" --checkpoint "$POLICY" --gripper-floor "$GRIPPER_FLOOR" \
    --robot-id "$ROBOT_ID" --port "$PORT"
