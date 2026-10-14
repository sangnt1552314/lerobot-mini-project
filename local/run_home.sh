#!/usr/bin/env bash
# Home arm (home_follower) with any sim policy (home, real, office, ...); its start pose and object come from the policy.
#
#   bash local/run_home.sh act_home_cube                     # home, then run the policy (keep a hand near the arm)
#   bash local/run_home.sh act_mujoco_real_v2 preview        # cameras + model only, the arm never moves
#   bash local/run_home.sh act_mujoco_real_v2 control --seconds 10 --hold-wrist-roll   # extra flags go to run_act_policy.py
#
# Overrides: PORT, GRIPPER_FLOOR. FOLLOWER_PORT from a .env is ignored on purpose, so the other arm's port can't be
# picked up. Run inside `conda activate lerobot`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOT_ID=home_follower
export PORT="${PORT:-/dev/tty.usbmodem5C821064861}"
export GRIPPER_FLOOR="${GRIPPER_FLOOR:-15}"  # this arm stalls below ~15
export SCRIPT_NAME="${BASH_SOURCE[0]}"
exec bash "$REPO_ROOT/local/run_policy.sh" "$@"
