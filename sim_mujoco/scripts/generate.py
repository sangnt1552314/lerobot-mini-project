"""Record scripted Lift demos in MuJoCo as a LeRobot v3 dataset (only successful episodes kept).

    python sim_mujoco/scripts/generate.py --num_demos 100 [--profile office --object cylinder] \
        --repo_id local/so101_liftcube_mujoco --root /scratch/.../lerobot/local/so101_liftcube_mujoco

Same features as the original Isaac pipeline: observation.images.front / .wrist (480x640x3 video, 30 fps),
observation.state (6,), action (6,) with names shoulder_pan.pos..gripper.pos, robot_type so101_follower,
task from the object ("Lift the red cube.", "Lift the red cylinder.", ...). Each frame stores the
observation *before* the action, as lerobot-record does. Episode i uses seed --seed + attempt index, so a
dataset is reproducible. SIMULATION ONLY.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_liftcube import JOINTS, OBJECTS, ROBOT_TYPE, LiftCubeConfig, LiftCubeEnv, PROFILES, make_config, ScriptedLiftExpert  # noqa: E402


def features(cfg: LiftCubeConfig) -> dict:
    names = [f"{j}.pos" for j in JOINTS]
    video = {"dtype": "video", "shape": (cfg.height, cfg.width, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.front": dict(video),
        "observation.images.wrist": dict(video),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_demos", type=int, required=True, help="number of SUCCESSFUL episodes to save")
    ap.add_argument("--repo_id", default="local/so101_liftcube_mujoco")
    ap.add_argument("--root", default=None, help="dataset dir (must not exist). Default $HF_LEROBOT_HOME/<repo_id>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_attempts", type=int, default=None, help="default 3*N+10")
    ap.add_argument("--image_writer_threads", type=int, default=8)
    ap.add_argument("--profile", choices=PROFILES, default="default",
                    help="physical setup the sim is matched to (start pose, camera, textures, randomization)")
    ap.add_argument("--object", choices=OBJECTS, default="cube", help="object to lift (so101_liftcube/config.py OBJECTS)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        sys.exit(f"{root} already exists; delete it or choose another --root/--repo_id.")
    cfg = make_config(args.profile, args.object)
    env = LiftCubeEnv(cfg, render=True)
    expert = ScriptedLiftExpert(env)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=cfg.fps, features=features(cfg), root=root,
                               robot_type=ROBOT_TYPE, use_videos=True, image_writer_threads=args.image_writer_threads)

    max_attempts = args.max_attempts or 3 * args.num_demos + 10
    saved = attempts = 0
    episodes = []   # seed and object size of each saved episode
    failed = []     # the dropped attempts and why
    t0 = time.time()
    while saved < args.num_demos and attempts < max_attempts:
        seed = args.seed + attempts
        attempts += 1
        obs = env.reset(seed=seed)
        expert.plan()
        frames = []
        for t in range(len(expert)):
            action = expert.action(t)
            frames.append({**obs, "action": action, "task": cfg.obj.task})
            obs = env.step(action)
        size = f"{200 * env.object_half_width:.1f} x {200 * env.object_half_height:.1f} cm"   # width/diameter x height
        if not expert.safe:  # the plan brings the wrist camera too close to the arm (setup's min_camera_clearance)
            failed.append({"seed": seed, "reason": "unsafe", "camera_clearance_m": round(float(expert.min_clearance), 4)})
            print(f"attempt {attempts} (seed {seed}, {size}): UNSAFE (wrist camera {1000 * expert.min_clearance:.0f} mm "
                  f"from the arm), dropped", flush=True)
            continue
        if not env.is_success():
            failed.append({"seed": seed, "reason": "not_lifted", "object_z_m": round(float(env.object_pos[2]), 4),
                           "object_xy_m": np.round(env.object_pos[:2], 4).tolist(), "object_width_m": round(2 * env.object_half_width, 4)})
            print(f"attempt {attempts} (seed {seed}, {size}): FAILED (object z {env.object_pos[2]:.3f}), dropped", flush=True)
            if saved == 0 and attempts >= 20:
                sys.exit(f"the expert failed the first {attempts} attempts of --profile {args.profile} --object "
                         f"{args.object}: this setup cannot do this object (e.g. the setup's gripper_closed leaves the "
                         f"jaws wider than the object); see sim_mujoco/scripts/preview.py")
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        saved += 1
        episodes.append({"seed": seed, "object_width_m": round(2 * env.object_half_width, 4),
                         "object_height_m": round(2 * env.object_half_height, 4)})
        print(f"attempt {attempts} (seed {seed}, {size}): SUCCESS, episode {saved}/{args.num_demos} saved "
              f"({len(frames)} frames, {time.time() - t0:.0f}s elapsed)", flush=True)
    if hasattr(ds, "stop_image_writer"):   # LeRobot 0.4.2 (Hopper); 0.6 stops it in finalize()
        ds.stop_image_writer()
    ds.finalize()

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = ""
    (root / "sim_mujoco_generation.json").write_text(json.dumps({
        "generator": "sim_mujoco/scripts/generate.py (MuJoCo, scripted expert)",
        "git_commit": commit, "profile": args.profile, "object": args.object, "seed": args.seed,
        "attempts": attempts, "saved": saved,
        "expert_success_rate": saved / max(attempts, 1), "config": dataclasses.asdict(cfg),
        "mjcf": str(os.environ.get("SO101_MJCF", "sim_mujoco/third_party/mujoco_menagerie/robotstudio_so101/so101.xml")),
        "episodes": episodes, "failed_attempts": failed,
    }, indent=2))
    print(f"\nsaved {saved}/{args.num_demos} episodes in {attempts} attempts "
          f"(expert success rate {100 * saved / max(attempts, 1):.1f}%), {time.time() - t0:.0f}s -> {root}")
    if saved < args.num_demos:
        sys.exit(f"only {saved}/{args.num_demos} demos recorded")


if __name__ == "__main__":
    main()
