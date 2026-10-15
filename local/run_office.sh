#!/usr/bin/env bash
# Office arm (my_follower) with any sim policy (office, home, ...); its start pose and object come from the policy.
#
#   bash local/run_office.sh act_office_cylinder             # home, then run the policy (keep a hand near the arm)
#   bash local/run_office.sh act_office2_cube_v2             # office2: starts folded with the wrist at -87.6 deg
#   bash local/run_office.sh act_office_cube preview         # cameras + model only, the arm never moves
#   bash local/run_office.sh outputs/act_office_cube control --seconds 10   # extra flags go to run_act_policy.py
#
# Overrides: PORT, GRIPPER_FLOOR. FOLLOWER_PORT from a .env is ignored on purpose, so the other arm's port can't be
# picked up. Run inside `conda activate lerobot`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOT_ID=my_follower
export PORT="${PORT:-/dev/tty.usbmodem5B610366661}"
export GRIPPER_FLOOR="${GRIPPER_FLOOR:-8}"   # this arm closes at ~0; the cylinder policy closes to 8
export SCRIPT_NAME="${BASH_SOURCE[0]}"
exec bash "$REPO_ROOT/local/run_policy.sh" "$@"
