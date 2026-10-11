# shellcheck shell=bash
# Shared helpers, sourced by the sim/*.sh scripts. Not meant to be run directly.
# SIMULATION ONLY: nothing here talks to a physical robot.

set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SIM_DIR/.." && pwd)"

info() { printf '\033[1;34m[sim]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[sim] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[sim] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Resolve a path relative to the repository root.
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$REPO_ROOT/$1" ;; esac; }

# KEY=VALUE lines from sim/.env (or .env.example). Variables already set in the
# environment win, so `OUTPUT_DIR=outputs/x bash sim/train_act.sh` works.
load_env() {
    local env_file="$SIM_DIR/.env" line key val
    [ -f "$env_file" ] || env_file="$SIM_DIR/.env.example"
    while IFS= read -r line || [ -n "$line" ]; do
        [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key="${BASH_REMATCH[1]}"
        val="${BASH_REMATCH[2]}"
        val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
        [ -n "${!key+x}" ] || export "$key=$val"
    done < "$env_file"

    SIM_CONDA_ENV="${SIM_CONDA_ENV:-leisaac_sim}"
    LEISAAC_ROOT="$(abspath "${LEISAAC_ROOT:-sim/third_party/leisaac}")"
    LEISAAC_REF="${LEISAAC_REF:-24d3bcd3f1e4585740fc79921782c41617237812}"
    HF_USER="${HF_USER:-local}"
    SIM_TASK="${SIM_TASK:-LeIsaac-SO101-LiftCube-DualCam-v0}"
    SIM_DATASET_NAME="${SIM_DATASET_NAME:-so101_liftcube_dualcam_sim}"
    SIM_DATA_ROOT="$(abspath "${SIM_DATA_ROOT:-sim/data}")"
    NUM_DEMOS="${NUM_DEMOS:-100}"
    DATASET_FPS="${DATASET_FPS:-30}"
    TRAIN_STEPS="${TRAIN_STEPS:-20000}"
    SMOKE_TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-10000}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SAVE_FREQ="${SAVE_FREQ:-5000}"
    DEVICE="${DEVICE:-cuda}"
    OUTPUT_DIR="$(abspath "${OUTPUT_DIR:-outputs/act_liftcube}")"
    WANDB_ENABLE="${WANDB_ENABLE:-true}"
    WANDB_PROJECT="${WANDB_PROJECT:-so101_liftcube_sim}"
    EVAL_EPISODES="${EVAL_EPISODES:-20}"

    DATASET_REPO_ID="$HF_USER/$SIM_DATASET_NAME"
    # LeIsaac finds assets via the git root of the *current* directory unless this is set,
    # which would point at this repository instead of the LeIsaac checkout.
    export LEISAAC_ROOT
    export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
    export HF_LEROBOT_HOME="$SIM_DATA_ROOT/lerobot"
    export DATASET_REPO_ID DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
    cd "$REPO_ROOT"  # relative paths in sim/.env are relative to the repository root
}

activate_env() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "$SIM_CONDA_ENV" ]; then return 0; fi
    command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda, then run: bash sim/setup.sh"
    set +u  # conda's activation scripts reference unset variables
    eval "$(conda shell.bash hook)"
    conda activate "$SIM_CONDA_ENV" || { set -u; die "conda env '$SIM_CONDA_ENV' not found. Run: bash sim/setup.sh"; }
    set -u
}

require_linux_nvidia() {
    [ "$(uname -s)" = "Linux" ] || die "Isaac Sim needs Linux with an NVIDIA RTX GPU (this machine: $(uname -s) $(uname -m))."
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install the NVIDIA driver first."
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >/dev/null \
        || die "nvidia-smi failed; the NVIDIA driver is not working."
}

require_assets() {
    [ -f "$LEISAAC_ASSETS_ROOT/robots/so101_follower.usd" ] \
        || die "Missing $LEISAAC_ASSETS_ROOT/robots/so101_follower.usd. Run: bash sim/setup.sh"
    [ -f "$LEISAAC_ASSETS_ROOT/scenes/table_with_cube/scene.usd" ] \
        || die "Missing $LEISAAC_ASSETS_ROOT/scenes/table_with_cube/scene.usd. Run: bash sim/setup.sh"
}

# Read {"ok": ...} from a JSON report written by a Python tool.
report_ok() {
    python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") else 1)' "$1" 2>/dev/null
}

require_int() {
    [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer, got '$2'"
}
