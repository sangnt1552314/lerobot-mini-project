"""Evaluate a LeRobot ACT checkpoint in LeIsaac-SO101-LiftCube-DualCam-v0.

Follows LeIsaac scripts/evaluation/policy_inference.py (same env setup, `so101leader`
joint-position actions, success = the task's `success` termination, failure = time-out),
but runs ACT in-process instead of through LeIsaac's `lerobot-<type>` gRPC client.

Why not the gRPC path: LeIsaac's LeRobotServicePolicyClient targets LeRobot v0.3.3 and
pickles its own leisaac...RemotePolicyConfig. The LeRobot 0.4.2 policy server (the
version LeIsaac's recorder requires) rejects anything that is not its own
RemotePolicyConfig, and the `lerobot-async` extra pins protobuf 6.x against Isaac Lab's
protobuf 3.x. Loading the checkpoint here uses the exact LeRobot code that trained it
(ACTPolicy + saved pre/post-processors + lerobot.utils.control_utils.predict_action).
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an ACT checkpoint in simulation.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-DualCam-v0")
parser.add_argument("--checkpoint", type=str, required=True, help=".../checkpoints/<step>/pretrained_model")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--episode_length_s", type=float, default=None, help="Default: task.episode_length_s in YAML.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--step_hz", type=int, default=0, help="Real-time rate limit in Hz; 0 disables it.")
parser.add_argument("--n_action_steps", type=int, default=None, help="Override ACT's open-loop steps per chunk.")
parser.add_argument("--results_file", type=str, default=None, help="Write a JSON summary here.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac  # noqa: F401
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leisaac_ext  # noqa: E402,F401
from leisaac_ext.act_policy import LocalACTPolicy  # noqa: E402


def to_lerobot_observation(policy_obs: dict, camera_keys: list[str]) -> dict:
    """Same conversion the LeRobot recorder applies when writing the dataset."""
    obs = {
        "observation.state": convert_leisaac_action_to_lerobot(policy_obs["joint_pos"])[0].astype(np.float32),
    }
    for key in camera_keys:
        obs[f"observation.images.{key}"] = policy_obs[key][0].cpu().numpy().astype(np.uint8)
    return obs


def main():
    policy = LocalACTPolicy(args_cli.checkpoint, device=args_cli.device, n_action_steps=args_cli.n_action_steps)
    checkpoint = str(policy.checkpoint)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.recorders = None
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    camera_keys = policy.camera_keys
    if not set(camera_keys) <= set(env.scene.sensors):
        raise RuntimeError(f"Checkpoint expects cameras {camera_keys}, env has sensors {list(env.scene.sensors)}")
    task = env.cfg.task_description
    print(f"[eval] checkpoint={checkpoint}\n[eval] cameras={camera_keys} task='{task}' fps={1 / env.step_dt:.1f}")

    obs_dict, _ = env.reset()
    policy.reset()
    results = []
    while len(results) < args_cli.episodes and simulation_app.is_running():
        steps = 0
        outcome = None
        while outcome is None and simulation_app.is_running():
            with torch.inference_mode():
                observation = to_lerobot_observation(obs_dict["policy"], camera_keys)
                action = policy(observation, task=task, robot_type=env.cfg.robot_name)
                joint_targets = convert_lerobot_action_to_leisaac(action.reshape(1, -1))
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
                # The env auto-resets on termination/time-out, so obs_dict is already the next episode's.
                obs_dict, _, terminated, truncated, _ = env.step(torch.as_tensor(joint_targets, device=env.device))
                steps += 1
                if terminated[0]:
                    outcome = "success"
                elif truncated[0]:
                    outcome = "timeout"
            if args_cli.step_hz > 0:
                time.sleep(1.0 / args_cli.step_hz)
        if outcome is None:
            break
        policy.reset()
        results.append({"episode": len(results) + 1, "outcome": outcome, "steps": steps})
        n_ok = sum(r["outcome"] == "success" for r in results)
        print(f"[eval] episode {len(results)}/{args_cli.episodes}: {outcome} after {steps} steps "
              f"(running success rate {n_ok}/{len(results)})")

    successes = sum(r["outcome"] == "success" for r in results)
    total = len(results)
    summary = {
        "checkpoint": checkpoint,
        "task": args_cli.task,
        "successes": successes,
        "failures": total - successes,
        "total_episodes": total,
        "success_rate": successes / total if total else 0.0,
        "episodes": results,
    }
    print("\n========== ACT simulation evaluation ==========")
    print(f"successes      {successes}")
    print(f"failures       {total - successes}")
    print(f"total episodes {total}")
    print(f"success rate   {summary['success_rate']:.1%}")
    print("===============================================")
    if args_cli.results_file:
        Path(args_cli.results_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.results_file).write_text(json.dumps(summary, indent=2))
        print(f"[eval] wrote {args_cli.results_file}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
