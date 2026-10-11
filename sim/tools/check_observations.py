"""Launch the dual-camera LiftCube env, report observation/action shapes, save camera frames.

    python sim/tools/check_observations.py --headless --enable_cameras

Checks that the SO-101 and the cube load, that physics runs (the cube rests on the table),
that both cameras render, and what the LeRobot recorder will write. Shapes are read from
the live env, nothing is assumed. Writes <out_dir>/report.json with "ok": true/false
(Isaac Sim does not reliably propagate exit codes, so the shell scripts read this file).
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SIM_DIR = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Check observations of the dual-camera LiftCube env.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-DualCam-v0")
parser.add_argument("--steps", type=int, default=60, help="Env steps to run before checking (60 = 2 s).")
parser.add_argument("--out_dir", type=str, default=str(SIM_DIR / "debug" / "observations"))
parser.add_argument("--fps", type=int, default=30, help="Dataset fps passed to the feature builder.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES
from leisaac.utils.robot_utils import build_feature_from_env
from PIL import Image

sys.path.insert(0, str(SIM_DIR))
import leisaac_ext  # noqa: E402,F401
from leisaac_ext.config import load_config  # noqa: E402

TABLE_TOP_Z = 0.041  # from table_with_cube/scene.usd
CUBE_HALF = 0.015


def main():
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    errors = []

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    env_cfg.recorders = None
    env_cfg.terminations.success = None  # don't auto-reset mid-check
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    robot = env.scene["robot"]
    hold = robot.data.joint_pos.clone()  # absolute joint targets == current pose -> arm holds still
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            obs, *_ = env.step(hold)

    policy_obs = obs["policy"]
    print(f"\nTask: {args_cli.task}   env step: {env.step_dt:.4f} s ({1 / env.step_dt:.1f} Hz)")
    print("\nPolicy observation terms (LeIsaac names):")
    for key, value in policy_obs.items():
        print(f"  {key:<20} {tuple(value.shape[1:])}  {value.dtype}")

    features = build_feature_from_env(env, LeRobotDatasetCfg(fps=args_cli.fps))
    print("\nLeRobot dataset features the recorder will write:")
    for key, feat in features.items():
        print(f"  {key:<28} {tuple(feat['shape'])}")
    print(f"  {'task':<28} '{env.cfg.task_description}'")

    # --- checks -------------------------------------------------------------------
    if list(robot.data.joint_names) != SINGLE_ARM_JOINT_NAMES:
        errors.append(f"robot joints {robot.data.joint_names} != {SINGLE_ARM_JOINT_NAMES}")
    for key, dim in (("observation.state", 6), ("action", 6)):
        if key not in features or tuple(features[key]["shape"]) != (dim,):
            errors.append(f"{key} shape {features.get(key, {}).get('shape')} != ({dim},)")
    for cam in ("front", "wrist"):
        key = f"observation.images.{cam}"
        want = (cfg["cameras"][cam]["height"], cfg["cameras"][cam]["width"], 3)
        if key not in features:
            errors.append(f"{key} missing from dataset features")
            continue
        if tuple(features[key]["shape"]) != want:
            errors.append(f"{key} shape {tuple(features[key]['shape'])} != {want}")
        image = policy_obs[cam][0].cpu().numpy().astype("uint8")
        Image.fromarray(image).save(out_dir / f"{cam}.png")
        if image.max() == image.min():
            errors.append(f"{cam} camera image is constant ({image.max()}); is --enable_cameras set?")

    cube_pos = env.scene["cube"].data.root_pos_w[0] - env.scene.env_origins[0]
    cube_z = float(cube_pos[2])
    print(f"\nCube position (env frame): {[round(float(v), 4) for v in cube_pos]}")
    if not (TABLE_TOP_Z + CUBE_HALF - 0.01 < cube_z < TABLE_TOP_Z + CUBE_HALF + 0.01):
        errors.append(f"cube z={cube_z:.4f} is not resting on the table (~{TABLE_TOP_Z + CUBE_HALF:.3f})")

    report = {
        "ok": not errors,
        "errors": errors,
        "env_step_hz": 1 / env.step_dt,
        "policy_obs": {k: list(v.shape[1:]) for k, v in policy_obs.items()},
        "features": {k: list(v["shape"]) for k, v in features.items()},
        "cube_pos_env": [float(v) for v in cube_pos],
        "images": [str(out_dir / f"{c}.png") for c in ("front", "wrist")],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved camera frames to {out_dir}/front.png and wrist.png")
    print("\nRESULT:", "PASS" if not errors else "FAIL\n  - " + "\n  - ".join(errors))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
