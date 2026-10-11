"""Record state-machine demonstrations for LeIsaac-SO101-LiftCube-DualCam-v0.

Adapted from LeIsaac scripts/datagen/state_machine/generate.py (commit 24d3bcd). Upstream
keeps TASK_REGISTRY as a literal dict inside that script and imports only `leisaac.tasks`,
so a task registered outside the LeIsaac package cannot be added without editing the
installed checkout. This copy keeps the upstream loop (state machine -> env.step ->
success check -> auto_terminate -> recorder export) and LeIsaac's own recorder managers,
and changes only:
  * registers our task from sim/leisaac_ext and uses LiftCubeStateMachine,
  * uses the joint-position action space (`so101leader`) so actions are joint targets,
  * keeps a `success` termination even when not recording (upstream crashes there),
  * --step_hz 0 runs as fast as possible (useful headless),
  * stops after --max_attempts and prints a success-rate summary.
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import signal
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State-machine data generation for the dual-camera LiftCube task.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-DualCam-v0")
parser.add_argument("--seed", type=int, default=None, help="Defaults to the current time.")
parser.add_argument("--record", action="store_true", help="Save successful episodes.")
parser.add_argument("--num_demos", type=int, default=20, help="Successful demos to record (0 = run forever).")
parser.add_argument("--max_attempts", type=int, default=None, help="Give up after this many episodes. Default 3*N+10.")
parser.add_argument("--step_hz", type=int, default=0, help="Real-time rate limit in Hz; 0 disables it.")
parser.add_argument("--recorder", choices=["lerobot", "hdf5"], default="lerobot")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None)
parser.add_argument("--lerobot_dataset_fps", type=int, default=30)
parser.add_argument("--dataset_file", type=str, default="./datasets/liftcube_dualcam.hdf5", help="HDF5 recorder only.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac  # noqa: F401  registers upstream tasks, applies LeIsaac monkey patches
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leisaac_ext  # noqa: E402,F401  registers LeIsaac-SO101-LiftCube-DualCam-v0
from leisaac_ext.lift_cube_dualcam.state_machine import LiftCubeStateMachine  # noqa: E402

# Same shape as upstream: gym id -> (state machine class, action-space device type).
TASK_REGISTRY = {
    "LeIsaac-SO101-LiftCube-DualCam-v0": (LiftCubeStateMachine, "so101leader"),
}


class RateLimiter:
    """Upstream helper: sleep to hold a real-time rate, rendering while waiting."""

    def __init__(self, hz):
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def constant_success(value: bool) -> TerminationTermCfg:
    return TerminationTermCfg(func=lambda env: torch.full((env.num_envs,), value, dtype=torch.bool, device=env.device))


def auto_terminate(env: ManagerBasedRLEnv, success: bool):
    """Upstream trick: flip the `success` term so the recorder exports (or drops) the episode on reset."""
    env.termination_manager.set_term_cfg("success", constant_success(success))
    env.termination_manager.compute()


def configure_env_cfg(env_cfg):
    # The state machine decides when an episode ends and whether it succeeded.
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = constant_success(False)
    if not args_cli.record:
        env_cfg.recorders = None
        return
    if args_cli.recorder == "lerobot":
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    else:
        if os.path.exists(args_cli.dataset_file):
            raise FileExistsError(f"{args_cli.dataset_file} already exists; delete it or pick another --dataset_file")
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.dataset_file)), exist_ok=True)
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args_cli.dataset_file))
        env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]


def replace_recorder_manager(env, env_cfg):
    del env.recorder_manager
    if args_cli.recorder == "lerobot":
        from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
        from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager

        dataset_cfg = LeRobotDatasetCfg(repo_id=args_cli.lerobot_dataset_repo_id, fps=args_cli.lerobot_dataset_fps)
        env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        print(f"[generate] LeRobot features: {dataset_cfg.features}")
    else:
        from leisaac.enhance.managers import StreamingRecorderManager

        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"


def main():
    if args_cli.task not in TASK_REGISTRY:
        raise ValueError(f"Task '{args_cli.task}' is not in TASK_REGISTRY: {list(TASK_REGISTRY)}")
    if args_cli.record and args_cli.recorder == "lerobot" and not args_cli.lerobot_dataset_repo_id:
        raise ValueError("--lerobot_dataset_repo_id is required with --record")
    sm_class, device = TASK_REGISTRY[args_cli.task]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    configure_env_cfg(env_cfg)

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    env_fps = 1.0 / env.step_dt
    if args_cli.record and args_cli.recorder == "lerobot" and abs(env_fps - args_cli.lerobot_dataset_fps) > 1e-3:
        raise ValueError(
            f"Env steps at {env_fps:.2f} Hz but --lerobot_dataset_fps={args_cli.lerobot_dataset_fps}. "
            "Each env step is one dataset frame; change sim.decimation or DATASET_FPS so they match."
        )
    if args_cli.record:
        replace_recorder_manager(env, env_cfg)

    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None
    max_attempts = args_cli.max_attempts or (3 * args_cli.num_demos + 10 if args_cli.num_demos > 0 else 0)

    sm = sm_class()
    sm.setup(env)
    env.reset()
    sm.reset()

    attempts, successes, recorded = 0, 0, 0
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[generate] Ctrl+C: finishing up...")

    original_sigint = signal.signal(signal.SIGINT, on_sigint)
    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, device)

                if sm.is_episode_done:
                    attempts += 1
                    success = sm.check_success(env)
                    successes += int(success)
                    auto_terminate(env, success and args_cli.record)
                    env.reset()  # recorder exports or drops the episode here
                    sm.reset()
                    if args_cli.record:
                        recorded = env.recorder_manager.exported_successful_episode_count
                    print(
                        f"[generate] episode {attempts}: {'SUCCESS' if success else 'FAILED'} "
                        f"(expert success {successes}/{attempts}, recorded {recorded}/{args_cli.num_demos or 'inf'})"
                    )
                    if args_cli.record and args_cli.num_demos > 0 and recorded >= args_cli.num_demos:
                        print(f"[generate] Recorded all {args_cli.num_demos} demonstrations.")
                        break
                    if max_attempts and attempts >= max_attempts:
                        print(f"[generate] Stopping after {attempts} attempts (--max_attempts).")
                        break
                    continue

                sm.pre_step(env)
                env.step(sm.get_action(env))
                sm.advance()

                if rate_limiter:
                    rate_limiter.sleep(env)
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        rate = successes / attempts if attempts else 0.0
        print(f"[generate] SUMMARY attempts={attempts} expert_successes={successes} rate={rate:.2f} recorded={recorded}")
        env.close()
        simulation_app.close()

    if args_cli.record and args_cli.num_demos > 0 and recorded < args_cli.num_demos:
        sys.exit(f"[generate] Only {recorded}/{args_cli.num_demos} demos recorded. Tune the expert in the YAML config.")


if __name__ == "__main__":
    main()
