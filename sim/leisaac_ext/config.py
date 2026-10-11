"""Loads sim/configs/lift_cube_dualcam.yaml (or $SIM_CONFIG) and checks it.

Pure Python + PyYAML so it can be tested without Isaac Sim.
"""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lift_cube_dualcam.yaml"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERA_NAMES = ["front", "wrist"]
PHASES = ["approach", "hover", "descend", "grasp", "lift", "hold"]


def config_path() -> Path:
    return Path(os.environ.get("SIM_CONFIG") or DEFAULT_CONFIG_PATH).expanduser().resolve()


def _require(cfg: dict, dotted: str):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"{config_path()}: missing required key '{dotted}'")
        node = node[key]
    return node


def _check_range(name: str, value) -> None:
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and value[0] <= value[1]):
        raise ValueError(f"{config_path()}: '{name}' must be [min, max], got {value!r}")


def validate(cfg: dict) -> dict:
    _require(cfg, "task.description")
    if _require(cfg, "task.success_height_above_base") >= _require(cfg, "expert.lift_cube_height_above_base"):
        raise ValueError("expert.lift_cube_height_above_base must exceed task.success_height_above_base")
    if _require(cfg, "sim.decimation") < 1:
        raise ValueError("sim.decimation must be >= 1")

    if sorted(_require(cfg, "robot.init_joint_pos_deg")) != sorted(JOINT_NAMES):
        raise ValueError(f"robot.init_joint_pos_deg must list exactly {JOINT_NAMES}")
    for part in ("arm", "gripper"):
        for key in ("stiffness", "damping", "effort_limit"):
            _require(cfg, f"robot.{part}.{key}")

    for cam in CAMERA_NAMES:
        for key in ("parent", "pos", "rot", "convention", "focal_length", "horizontal_aperture", "width", "height", "fps"):
            _require(cfg, f"cameras.{cam}.{key}")
        if len(cfg["cameras"][cam]["pos"]) != 3 or len(cfg["cameras"][cam]["rot"]) != 4:
            raise ValueError(f"cameras.{cam}: pos needs 3 values and rot needs 4 (w, x, y, z)")
        if cfg["cameras"][cam]["convention"] not in ("ros", "opengl", "world"):
            raise ValueError(f"cameras.{cam}.convention must be ros, opengl or world")

    for key in ("x", "y", "yaw_deg"):
        _check_range(f"randomization.cube.{key}", _require(cfg, f"randomization.cube.{key}"))
    for key in ("xyz", "rpy_deg"):
        _check_range(f"randomization.front_camera.{key}", _require(cfg, f"randomization.front_camera.{key}"))

    if len(_require(cfg, "expert.grasp_offset")) != 3:
        raise ValueError("expert.grasp_offset needs 3 values")
    if _require(cfg, "expert.ik_command") not in ("pose", "position"):
        raise ValueError("expert.ik_command must be 'pose' or 'position'")
    if len(_require(cfg, "expert.target_rpy_deg")) != 3:
        raise ValueError("expert.target_rpy_deg needs 3 values")
    for phase in PHASES:
        if _require(cfg, f"expert.phase_s.{phase}") <= 0:
            raise ValueError(f"expert.phase_s.{phase} must be > 0")
    return cfg


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else config_path()
    with open(path) as f:
        return validate(yaml.safe_load(f))
