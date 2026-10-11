#!/usr/bin/env bash
# Look at the scene before generating data.
#
#   bash sim/preview.sh            # headless check + save camera frames, then open the GUI
#   bash sim/preview.sh --no-gui   # headless check only (remote machines)
#
# Step 1 checks the robot, cube, physics and both cameras, and saves
# sim/debug/observations/{front,wrist}.png. Step 2 opens Isaac Sim and runs the scripted
# expert in a loop without recording, so you can watch it lift the cube. Close the window
# or press Ctrl+C to stop. SIMULATION ONLY.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env
require_linux_nvidia
activate_env
require_assets

info "Step 1/2: headless observation check"
REPORT="$SIM_DIR/debug/observations/report.json"
rm -f "$REPORT"
python "$SIM_DIR/tools/check_observations.py" --task "$SIM_TASK" --headless --enable_cameras --device "$DEVICE" \
    --fps "$DATASET_FPS"
report_ok "$REPORT" || die "Observation check failed; see $REPORT"
info "Camera frames: $SIM_DIR/debug/observations/front.png and wrist.png"

if [ "${1:-}" = "--no-gui" ]; then
    exit 0
fi

info "Step 2/2: GUI with the scripted expert (not recording). Ctrl+C or close the window to stop."
python "$SIM_DIR/scripts/generate.py" --task "$SIM_TASK" --enable_cameras --device "$DEVICE" \
    --num_demos 0 --step_hz "$DATASET_FPS"
