"""Export office3's object placement points (sim_mujoco/so101_liftcube/placements.py) for the squint envs.

Writes sim_squint/assets/office3_points_<object>.npz: points (N, 2) x, y of the object's centre on the 5 mm grid (in the
region, clear of the folded arm for the largest object at 45 deg yaw), split (N,) 0 train / 1 val / 2 test (held-out
cells without their rim, as in placements.py). Same robot frame as squint (base at the origin, +x forward), so the
points are used as they are. Run in the MuJoCo env:

    bash sim/hpc_env.sh python sim_squint/tools/export_office3_points.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim_mujoco"))

from so101_liftcube.config import make_config  # noqa: E402
from so101_liftcube.env import LiftCubeEnv  # noqa: E402
from so101_liftcube.placements import Layout  # noqa: E402

for obj in ("cube", "cylinder"):
    layout = Layout(LiftCubeEnv(make_config("office3", obj), render=False))
    split = np.full(len(layout.points), -1)
    split[np.isin(layout.label, layout.cells["train"])] = 0
    for s, code in (("val", 1), ("test", 2)):
        for cell in layout.cells[s]:
            keep = [tuple(p) for p in layout._held_out_points[cell]]
            for i in np.flatnonzero(layout.label == cell):
                if tuple(layout.points[i]) in keep:
                    split[i] = code
    keep = split >= 0
    out = REPO / "sim_squint" / "assets" / f"office3_points_{obj}.npz"
    np.savez(out, points=layout.points[keep].astype(np.float32), split=split[keep].astype(np.int8))
    print(obj, out.name, {n: int((split == c).sum()) for n, c in (("train", 0), ("val", 1), ("test", 2))})
