"""vision_pick: pick up a red cube with the real SO-101 using OpenCV, a touch-calibrated camera->table homography
and inverse kinematics. No learning, no simulator, no server. See README.md here.

The rest of local/ uses flat imports (`from safety import ...`), so local/ goes on sys.path first.
"""

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
LOCAL_DIR = PACKAGE_DIR.parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))
