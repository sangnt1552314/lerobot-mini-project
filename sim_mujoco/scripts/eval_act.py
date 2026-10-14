"""Evaluate an ACT checkpoint in the MuJoCo SO-101 Lift task and report the success rate.

    python sim_mujoco/scripts/eval_act.py --checkpoint outputs/.../checkpoints/last/pretrained_model \
        --episodes 50 [--profile office --object cylinder] [--save_videos 3] [--results_file eval.json]

Use the --profile / --object the checkpoint's dataset was generated with. --wandb logs the success rate and
the saved videos to W&B, into the checkpoint's own training run (the run id in its train_config.json), so
they sit next to the training curves; credentials as for training (WANDB_* in the environment or in the
repo-root .env, or `wandb login`).

ACT runs in-process with the same LeRobot code that trained it (sim/leisaac_ext/act_policy.py ->
LocalACTPolicy: pre/post-processors from the checkpoint, action-chunk queue via predict_action).
Success: the object centre stays above success_height (0.10 m) for 5 steps; failure: time-out after
episode_length_s (12 s). Eval seeds start at --seed (default 100000) so they never overlap the
generation seeds (0, 1, 2, ...). SIMULATION ONLY.
"""

import argparse
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                      # sim_mujoco/
sys.path.insert(0, str(HERE.parents[2] / "sim"))              # sim/ -> leisaac_ext (no Isaac imports)
from so101_liftcube import OBJECTS, ROBOT_TYPE, LiftCubeConfig, LiftCubeEnv, PROFILES, make_config  # noqa: E402


def log_to_wandb(ckpt: Path, summary: dict, video_dir: Path) -> None:
    """Success rate + eval videos to W&B, into the checkpoint's training run when train_config.json names one."""
    import wandb

    env_file = HERE.parents[2] / ".env"   # WANDB_* as sim/train_act_hpc.sh reads them; the environment wins
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, sep, val = line.strip().partition("=")
            if sep and key.startswith("WANDB_") and key not in os.environ:
                os.environ[key] = val.strip().strip("'\"")
    try:
        wb = json.loads((ckpt / "train_config.json").read_text()).get("wandb") or {}
    except (OSError, ValueError):
        wb = {}
    kwargs = {"project": wb.get("project") or os.environ.get("WANDB_PROJECT", "cs6283"),
              "entity": wb.get("entity") or os.environ.get("WANDB_ENTITY") or None, "mode": wb.get("mode") or None}
    if wb.get("run_id"):
        kwargs.update(id=wb["run_id"], resume="allow")
    else:
        kwargs.update(job_type="eval", name=f"eval_{ckpt.parents[2].name}")
    videos = sorted(video_dir.glob("*.mp4")) if video_dir.is_dir() else []
    try:
        run = wandb.init(**kwargs)
        metrics = {f"eval/{k}": summary[k] for k in ("success_rate", "successes", "episodes")}
        run.log({**metrics, **{f"eval/video_{v.stem}": wandb.Video(str(v), format="mp4") for v in videos}})
        run.summary.update(metrics)
        print(f"[eval] logged success rate and {len(videos)} videos to W&B run {run.url or run.id}")
        run.finish()
    except Exception as e:  # the local results are already saved; W&B is a copy
        print(f"[eval] W&B logging failed ({type(e).__name__}: {e}); results and videos are in {video_dir.parent}")


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_action_steps", type=int, default=None, help="override ACT n_action_steps (default: checkpoint)")
    ap.add_argument("--save_videos", type=int, default=3, help="save front|wrist videos of the first N episodes")
    ap.add_argument("--results_file", default=None)
    ap.add_argument("--wandb", action="store_true", help="log the success rate and the saved videos to W&B")
    ap.add_argument("--profile", choices=PROFILES, default="default",
                    help="physical setup the sim is matched to (start pose, camera, textures, randomization)")
    ap.add_argument("--object", choices=OBJECTS, default="cube", help="object to lift (so101_liftcube/config.py OBJECTS)")
    args = ap.parse_args()

    from leisaac_ext.act_policy import LocalACTPolicy

    ckpt = Path(args.checkpoint).resolve()
    policy = LocalACTPolicy(ckpt, device=args.device, n_action_steps=args.n_action_steps)
    if policy.camera_keys != ["front", "wrist"]:
        sys.exit(f"Checkpoint expects cameras {policy.camera_keys}; this env provides ['front', 'wrist'].")
    cfg = make_config(args.profile, args.object)
    env = LiftCubeEnv(cfg, render=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.results_file) if args.results_file else ckpt.parent / f"eval_mujoco_{stamp}.json"
    video_dir = results_file.with_suffix("")
    print(f"[eval] checkpoint={ckpt}\n[eval] episodes={args.episodes} seeds {args.seed}.. profile {args.profile} object {args.object} task='{cfg.obj.task}' fps={cfg.fps}")

    episodes = []
    t0 = time.time()
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs = env.reset(seed=seed)
        policy.reset()
        frames = []
        max_z = 0.0
        while not env.is_success() and not env.timed_out():
            if ep < args.save_videos:
                frames.append(np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1))
            action = policy(obs, task=cfg.obj.task, robot_type=ROBOT_TYPE)
            obs = env.step(action)
            max_z = max(max_z, float(env.object_pos[2]))
        ok = env.is_success()
        episodes.append({"episode": ep, "seed": seed, "success": ok, "steps": env.steps, "max_object_z": round(max_z, 4)})
        print(f"[eval] episode {ep + 1}/{args.episodes} seed {seed}: {'SUCCESS' if ok else 'FAILED '} "
              f"after {env.steps} steps (max object z {max_z:.3f} m)", flush=True)
        if frames:
            video_dir.mkdir(parents=True, exist_ok=True)
            save_video(frames, video_dir / f"episode_{ep:03d}_{'success' if ok else 'fail'}.mp4", cfg.fps)

    n_ok = sum(e["success"] for e in episodes)
    summary = {
        "checkpoint": str(ckpt), "env": "MuJoCo SO-101 Lift (sim_mujoco)", "profile": args.profile,
        "object": args.object, "episodes": len(episodes),
        "successes": n_ok, "failures": len(episodes) - n_ok, "success_rate": n_ok / max(len(episodes), 1),
        "success_height_m": cfg.success_height, "episode_length_s": cfg.episode_length_s,
        "seed": args.seed, "wall_s": round(time.time() - t0, 1), "per_episode": episodes,
    }
    results_file.write_text(json.dumps(summary, indent=2))
    print("\n========== ACT MuJoCo evaluation ==========")
    print(f"successes      {n_ok}\nfailures       {len(episodes) - n_ok}\ntotal episodes {len(episodes)}")
    print(f"success rate   {100 * summary['success_rate']:.1f}%")
    print("===========================================")
    print(f"[eval] results: {results_file}" + (f"\n[eval] videos:  {video_dir}" if args.save_videos else ""))
    if args.wandb:
        log_to_wandb(ckpt, summary, video_dir)


if __name__ == "__main__":
    main()
