"""Run squint's deploy.py (third_party/squint, unmodified) on the office arm with an office3 squint policy. Laptop:

    python sim_squint/deploy_office3.py --checkpoint=path/to/ckpt.pt --env_id=SO101LiftCubeOffice3Both-v1 \
        --no-continuous_eval            # first runs: asks before every step (deploy.py's flag)

Env vars (as local/run_office.sh): PORT (default the office arm's), ROBOT_ID (my_follower), FRONT_CAM (OpenCV index of
the D435's RGB, 1), WRIST_CAM (0), GRIPPER_FLOOR (lowest gripper command in LeRobot units, 0).

What is replaced in squint's deploy (it is written for the authors' arm and WowRobo wrist camera):
- robot: SOFollower with use_degrees=True and your LeRobot calibration (ROBOT_ID), not squint's calibration file in
  deploy_utils/; check it isn't squint's that sits in the motors (README: "the motors' calibration differs").
- cameras: the sim sensors' names: base_camera = the D435 (front), wrist_camera = the wrist camera; deploy.py
  centre-crops each frame to a square and resizes it, as the sim cameras are set up.
- gripper: squint maps servo degrees through constants measured on its arm. Here: LeRobot 0..100 <-> the model's
  -10..100 deg, the mapping office3 / sim_mujoco use (arm joints: degrees = model radians, both unchanged).
- the end-of-run "rest" pose: office3's folded rest instead of squint's (wrist rolled to -180 deg).
"""

import os
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SQUINT = HERE.parent / "third_party" / "squint"
sys.path.insert(0, str(SQUINT))
sys.path.append(str(HERE))

import office3_env  # noqa: E402  (registers the office3 envs; imports squint's envs)
from envs.robot.so101 import SO101  # noqa: E402

GRIPPER_LO, GRIPPER_HI = -0.174533, 1.7453292     # rad, the model's gripper range = LeRobot 0..100
GRIPPER_FLOOR = float(os.environ.get("GRIPPER_FLOOR", "0"))


def create_real_robot():
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cams = {"base_camera": OpenCVCameraConfig(index_or_path=int(os.environ.get("FRONT_CAM", "1")), fps=30, width=640, height=480),
            "wrist_camera": OpenCVCameraConfig(index_or_path=int(os.environ.get("WRIST_CAM", "0")), fps=30, width=640, height=480)}
    env_id = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--env_id=")), "")
    keep = ("base_camera", "wrist_camera") if "Both" in env_id else ("wrist_camera",) if "Wrist" in env_id else ("base_camera",)
    cfg = SOFollowerRobotConfig(port=os.environ.get("PORT", "/dev/tty.usbmodem5B610366661"),
                                id=os.environ.get("ROBOT_ID", "my_follower"), use_degrees=True,
                                cameras={k: v for k, v in cams.items() if k in keep})
    return SOFollower(cfg)


# deploy.py imports create_real_robot from deploy_utils.robot_config, whose imports need squint's lerobot version
sys.modules["deploy_utils.robot_config"] = types.SimpleNamespace(create_real_robot=create_real_robot)

from deploy_utils import manipulator  # noqa: E402


class Office3RealAgent(manipulator.LeRobotRealAgent):
    """LeRobotRealAgent with the gripper in LeRobot 0..100 units (no servo-degree remapping)."""

    def __init__(self, robot, **kwargs):
        super().__init__(robot, **kwargs)
        robot.bus.motors["gripper"].norm_mode = manipulator.MotorNormMode.RANGE_0_100

    def set_target_qpos(self, qpos):
        self._cached_qpos = None
        qpos = manipulator.common.to_cpu_tensor(qpos).flatten()
        target = {f"{k}.pos": float(np.rad2deg(qpos[i])) for i, k in enumerate(self._motor_keys)}
        units = (float(qpos[-1]) - GRIPPER_LO) / (GRIPPER_HI - GRIPPER_LO) * 100
        target["gripper.pos"] = float(np.clip(units, GRIPPER_FLOOR, 100))
        self.real_robot.send_action(target)

    def get_qpos(self):
        if self.use_cached_qpos and self._cached_qpos is not None:
            return self._cached_qpos.clone()
        read = self.real_robot.bus.sync_read("Present_Position")
        if self._motor_keys is None:
            self._motor_keys = list(read.keys())
        rad = [np.deg2rad(read[k]) for k in self._motor_keys[:-1]]
        rad.append(GRIPPER_LO + read["gripper"] / 100 * (GRIPPER_HI - GRIPPER_LO))
        self._cached_qpos = torch.tensor(rad, dtype=torch.float32).unsqueeze(0)
        return self._cached_qpos.clone()


manipulator.LeRobotRealAgent = Office3RealAgent
SO101.keyframes["rest"].qpos[:] = office3_env.HOME_QPOS

if __name__ == "__main__":
    sys.argv[0] = str(SQUINT / "deploy.py")
    runpy.run_path(str(SQUINT / "deploy.py"), run_name="__main__")
