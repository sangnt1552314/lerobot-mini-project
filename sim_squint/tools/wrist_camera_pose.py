"""office3's wrist camera (Menagerie camera_mount + office2's fitted 9 deg extra rotation) as a pose in squint's
gripper_link frame (= the MJCF "gripper" body), in SAPIEN's camera convention (x forward, y left, z up).

    bash sim/hpc_env.sh python sim_squint/tools/wrist_camera_pose.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim_mujoco"))
from so101_liftcube.config import make_config  # noqa: E402
from so101_liftcube.env import LiftCubeEnv  # noqa: E402

env = LiftCubeEnv(make_config("office3_plain", "cube"), render=False)
m, d = env.model, env.data
for q in (env.cfg.home_qpos, (0.3, -0.5, 0.8, 0.4, -1.0, 0.5)):   # the pose in the gripper frame is the same at any qpos
    d.qpos[env.qadr] = q
    mujoco.mj_forward(m, d)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    Rg, pg = d.xmat[g].reshape(3, 3), d.xpos[g]
    Rc, pc = d.cam_xmat[env.wrist_cam].reshape(3, 3), d.cam_xpos[env.wrist_cam]
    p = Rg.T @ (pc - pg)
    Rm = Rg.T @ Rc                                   # MuJoCo camera: x right, y up, looks along -z
    Rs = np.stack([-Rm[:, 2], -Rm[:, 0], Rm[:, 1]], axis=1)   # SAPIEN: x forward, y left, z up
    qs = np.zeros(4)
    mujoco.mju_mat2Quat(qs, Rs.flatten())
    print("pos", np.round(p, 5).tolist(), "quat_wxyz", np.round(qs, 5).tolist(), "fovy", round(float(m.cam_fovy[env.wrist_cam]), 2))
