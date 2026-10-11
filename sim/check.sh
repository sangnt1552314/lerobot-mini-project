#!/usr/bin/env bash
# Validate the simulation module.
#
#   bash sim/check.sh            # static checks + launch Isaac headless and check observations
#   bash sim/check.sh --static   # static checks only (any OS, any Python with pyyaml + gymnasium)
#
# SIMULATION ONLY.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env

MODE="${1:-full}"

static_checks() {
    info "Shell syntax"
    for f in "$SIM_DIR"/*.sh; do bash -n "$f" || die "syntax error in $f"; done
    info "Python syntax"
    python - "$SIM_DIR" <<'EOF' || die "Python syntax error"
import pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).rglob("*.py")):
    if "third_party" in path.parts:
        continue
    compile(path.read_text(), str(path), "exec")
EOF
    info "Config + task registration"
    PYTHONPATH="$SIM_DIR${PYTHONPATH:+:$PYTHONPATH}" python - <<'EOF' || die "config/registration check failed"
import gymnasium as gym
import leisaac_ext
from leisaac_ext.config import CAMERA_NAMES, load_config
cfg = load_config()
spec = gym.spec(leisaac_ext.TASK_ID)
entry = spec.kwargs["env_cfg_entry_point"]
assert entry == "leisaac_ext.lift_cube_dualcam.env_cfg:LiftCubeDualCamEnvCfg", entry
cams = {c: (cfg["cameras"][c]["height"], cfg["cameras"][c]["width"], cfg["cameras"][c]["fps"]) for c in CAMERA_NAMES}
print(f"      {spec.id} -> {entry}")
print(f"      cameras (h, w, fps): {cams}")
EOF
}

static_checks
if [ "$MODE" = "--static" ]; then
    info "Static checks passed."
    exit 0
fi

info "Runtime checks"
require_linux_nvidia
activate_env
require_assets
python - <<'EOF' || die "package check failed"
import importlib.metadata as md
for pkg, want in (("isaacsim", "5.1"), ("isaaclab", "2.3"), ("lerobot", "0.4.2"), ("torch", "2.7")):
    have = md.version(pkg)
    print(f"      {pkg:<9} {have}" + ("" if have.startswith(want) else f"   (expected {want}.x)"))
EOF

info "Launching Isaac Sim headless (first launch can take several minutes)"
REPORT="$SIM_DIR/debug/observations/report.json"
rm -f "$REPORT"
python "$SIM_DIR/tools/check_observations.py" --task "$SIM_TASK" --headless --enable_cameras --device "$DEVICE" \
    --fps "$DATASET_FPS"
report_ok "$REPORT" || die "Observation check failed; see $REPORT"
info "All checks passed. Camera frames: $SIM_DIR/debug/observations/{front,wrist}.png"
