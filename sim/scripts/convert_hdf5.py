"""Fallback only: run LeIsaac's isaaclab2lerobotv3.py with our task registered.

The upstream converter looks the task up with parse_env_cfg, which only knows tasks
inside the LeIsaac package. Registering ours imports nothing but gymnasium, so it is safe
to do before the converter launches Isaac Sim. All arguments pass through unchanged:

    python sim/scripts/convert_hdf5.py --task_name LeIsaac-SO101-LiftCube-DualCam-v0 \
        --repo_id local/so101_liftcube_dualcam_sim --hdf5_root ./datasets \
        --hdf5_files liftcube_dualcam.hdf5 --fps 30 --headless --enable_cameras
"""

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leisaac_ext  # noqa: E402,F401

converter = Path(os.environ["LEISAAC_ROOT"]) / "scripts" / "convert" / "isaaclab2lerobotv3.py"
sys.argv[0] = str(converter)
runpy.run_path(str(converter), run_name="__main__")
