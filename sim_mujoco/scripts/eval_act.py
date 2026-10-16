"""Evaluate an ACT checkpoint in the MuJoCo SO-101 Lift task and report the success rate.

    python sim_mujoco/scripts/eval_act.py --checkpoint outputs/.../checkpoints/last/pretrained_model \
        --episodes 50 [--profile office --object cylinder] [--save_videos 3] [--results_file eval.json]

Use the --profile / --object the checkpoint's dataset was generated with. --wandb logs the success rate and
the saved videos to W&B, into the checkpoint's own training run (the run id in its train_config.json), so
they sit next to the training curves: eval/success_rate is plotted against eval/train_step (the checkpoint's
training step) and eval/videos holds the videos of every evaluation (step slider in the media panel).
Credentials as for training (WANDB_* in the environment or in the repo-root .env, or `wandb login`).
Do not run it while lerobot-train is still writing to the same run (pbs/mujoco_pipeline.pbs alternates them).

ACT runs in-process with the same LeRobot code that trained it (sim/leisaac_ext/act_policy.py ->
LocalACTPolicy: pre/post-processors from the checkpoint, action-chunk queue via predict_action).
Success: the object centre stays above success_height (0.10 m) for 5 steps; failure: time-out after
episode_length_s (12 s). Eval seeds start at --seed (default 100000) so they never overlap the
generation seeds (0, 1, 2, ...). SIMULATION ONLY.

--split val|test (profiles with an object_region, e.g. office3): episode k uses the k-th held-out placement of that
split (so101_liftcube/placements.py: never a training cell, the object's real size, the same placements at every
checkpoint) instead of --seed. Use val (20 episodes) to compare checkpoints and pick one, test (50) once, on the picked
checkpoint. The IK expert is not used: ACT acts from the two images and the joint state.
Each failure gets a category from the object's and fingertips' motion: knocked_over (tilted > 45 deg), dropped
(lifted >= 2 cm, then back on the table), lifted_too_low, pushed_away (moved >= 3 cm, never lifted), missed_grasp
(fingertips came within 3 cm), never_reached. --save_videos N keeps up to N/2 successes and N/2 failures (the rest
from whichever there are more of).
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
sys.path.insert(0, str(HERE.parents[2] / "sim"))              # sim/ -> leisaac_ext.act_policy
from so101_liftcube import OBJECTS, ROBOT_TYPE, LiftCubeConfig, LiftCubeEnv, PROFILES, make_config  # noqa: E402
from so101_liftcube.placements import Layout  # noqa: E402

FAILURES = ("knocked_over", "dropped", "lifted_too_low", "pushed_away", "missed_grasp", "never_reached")


def train_step(ckpt: Path) -> int | None:
    """Training step of a lerobot-train checkpoint (<run>/checkpoints/<step>/pretrained_model), else None."""
    try:
        return int(json.loads((ckpt.parent / "training_state" / "training_step.json").read_text())["step"])
    except (OSError, ValueError, KeyError):
        return None


def log_to_wandb(ckpt: Path, summary: dict, video_dir: Path, prefix: str = "eval") -> None:
    """Success rate + eval videos to W&B, into the checkpoint's training run when train_config.json names one
    (prefix "eval" for validation, "test" for the final test of the picked checkpoint)."""
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
    if ckpt.parent.parent.name == "checkpoints":   # local files next to the training run's (offline runs too)
        kwargs["dir"] = str(ckpt.parents[2])
    videos = sorted(video_dir.glob("*.mp4")) if video_dir.is_dir() else []
    step = summary.get("train_step")
    try:
        run = wandb.init(**kwargs)
        metrics = {f"{prefix}/{k}": summary[k] for k in ("success_rate", "successes", "episodes")}
        metrics.update({f"{prefix}/failures/{k}": v for k, v in summary.get("failure_categories", {}).items()})
        if step is not None:
            # lerobot-train already used W&B's own step up to this checkpoint, so eval gets its own x-axis
            run.define_metric(f"{prefix}/train_step")
            run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/train_step")
            metrics[f"{prefix}/train_step"] = step
        media = {f"{prefix}/videos": [wandb.Video(str(v), caption=v.stem, format="mp4") for v in videos]} if videos else {}
        run.log({**metrics, **media})
        run.summary.update(metrics)
        print(f"[eval] W&B: success rate {summary['success_rate']:.2f} at train step {step} and {len(videos)} videos "
              f"-> run {run.url or f'{run.id} (offline, upload with: wandb sync <run dir>/wandb/offline-run-*)'}")
        run.finish()
    except Exception as e:  # the local results are already saved; W&B is a copy
        print(f"[eval] W&B logging FAILED ({type(e).__name__}: {e}); results and videos are in {video_dir.parent}")


def prune_videos(video_dir: Path, episodes: list[dict], n: int) -> None:
    """Keep the videos of the first n // 2 failures and the first successes up to n (more of either kind when the other
    has fewer); delete the rest."""
    ok = [e for e in episodes if e["success"]]
    bad = [e for e in episodes if not e["success"]]
    n_bad = min(len(bad), max(n // 2, n - len(ok)))
    keep = {id(e) for e in ok[: n - n_bad] + bad[:n_bad]}
    for e in episodes:
        if id(e) not in keep:
            (video_dir / e.pop("video")).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_action_steps", type=int, default=None, help="override ACT n_action_steps (default: checkpoint)")
    ap.add_argument("--save_videos", type=int, default=3, help="keep front|wrist videos of N episodes (successes and failures)")
    ap.add_argument("--results_file", default=None)
    ap.add_argument("--wandb", action="store_true", help="log the success rate and the saved videos to W&B")
    ap.add_argument("--profile", choices=PROFILES, default="default",
                    help="physical setup the sim is matched to (start pose, camera, textures, randomization)")
    ap.add_argument("--object", choices=OBJECTS, default="cube", help="object to lift (so101_liftcube/config.py OBJECTS)")
    ap.add_argument("--split", choices=["val", "test"], default=None, help="held-out placements (office3) instead of --seed")
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

    layout = Layout(env) if args.split else None
    episodes = []
    t0 = time.time()
    for ep in range(args.episodes):
        placement = layout.placement(args.split, ep) if layout else None
        seed = placement.seed if placement else args.seed + ep
        obs = env.reset(seed=seed, placement=placement)
        policy.reset()
        writer = None
        if args.save_videos:   # every episode streams to disk; prune_videos keeps a mix of successes and failures
            import imageio.v2 as imageio

            video_dir.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_dir / "current.mp4", fps=cfg.fps, codec="libx264", quality=7, macro_block_size=8)
        z0, xy0 = float(env.object_pos[2]), env.object_pos[:2].copy()
        max_lift = max_shift = 0.0
        min_tip = np.inf
        while not env.is_success() and not env.timed_out():
            if writer:
                writer.append_data(np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1))
            action = policy(obs, task=cfg.obj.task, robot_type=ROBOT_TYPE)
            obs = env.step(action)
            pos = env.object_pos
            max_lift = max(max_lift, float(pos[2]) - z0)
            max_shift = max(max_shift, float(np.linalg.norm(pos[:2] - xy0)))
            min_tip = min(min_tip, float(np.linalg.norm(env.data.site("gripperframe").xpos - pos)))
        ok = env.is_success()
        tilt = float(np.degrees(np.arccos(np.clip(env.data.xmat[env.object_body].reshape(3, 3)[2, 2], -1, 1))))
        lift = float(env.object_pos[2]) - z0
        if ok:
            category = "success"
        elif tilt > 45:
            category = "knocked_over"
        elif max_lift >= 0.02 and lift < 0.01:
            category = "dropped"
        elif max_lift >= 0.01:
            category = "lifted_too_low"
        elif max_shift >= 0.03:
            category = "pushed_away"
        elif min_tip < 0.03:
            category = "missed_grasp"
        else:
            category = "never_reached"
        width, height = 200 * env.object_half_width, 200 * env.object_half_height   # cm (width or diameter)
        episodes.append({"episode": ep, "seed": seed, "success": ok, "category": category, "steps": env.steps,
                         "max_object_z": round(max_lift + z0 - cfg.table_z, 4), "max_lift_m": round(max_lift, 4),
                         "max_shift_m": round(max_shift, 4), "min_fingertip_dist_m": round(min_tip, 4), "final_tilt_deg": round(tilt, 1),
                         "object_width_cm": round(width, 2), "object_height_cm": round(height, 2),
                         **({"placement": placement.as_dict()} if placement else {}), "env": env.episode_info})
        where = f" {args.split} cell {placement.cell} ({placement.x:.3f}, {placement.y:+.3f})" if placement else ""
        print(f"[eval] episode {ep + 1}/{args.episodes} seed {seed}{where} ({width:.1f} x {height:.1f} cm): "
              f"{'SUCCESS' if ok else 'FAILED '} after {env.steps} steps ({category}, max lift {max_lift:.3f} m)", flush=True)
        if writer:
            writer.close()
            name = f"episode_{ep:03d}_{width:.1f}x{height:.1f}cm_{'success' if ok else 'fail_' + category}.mp4"
            (video_dir / "current.mp4").rename(video_dir / name)
            episodes[-1]["video"] = name

    if args.save_videos:
        prune_videos(video_dir, episodes, args.save_videos)
    n_ok = sum(e["success"] for e in episodes)
    summary = {
        "checkpoint": str(ckpt), "train_step": train_step(ckpt), "env": "MuJoCo SO-101 Lift (sim_mujoco)", "profile": args.profile,
        "object": args.object, "split": args.split, "episodes": len(episodes),
        "successes": n_ok, "failures": len(episodes) - n_ok, "success_rate": n_ok / max(len(episodes), 1),
        "failure_categories": {k: sum(e["category"] == k for e in episodes) for k in FAILURES},
        "success_height_m": cfg.success_height, "episode_length_s": cfg.episode_length_s,
        "seed": args.seed, "wall_s": round(time.time() - t0, 1), "per_episode": episodes,
    }
    results_file.write_text(json.dumps(summary, indent=2))
    print("\n========== ACT MuJoCo evaluation ==========")
    print(f"successes      {n_ok}\nfailures       {len(episodes) - n_ok}\ntotal episodes {len(episodes)}")
    print(f"success rate   {100 * summary['success_rate']:.1f}%")
    print("failures       " + (", ".join(f"{k} {v}" for k, v in summary["failure_categories"].items() if v) or "none"))
    print("===========================================")
    print(f"[eval] results: {results_file}" + (f"\n[eval] videos:  {video_dir}" if args.save_videos else ""))
    if args.wandb:
        log_to_wandb(ckpt, summary, video_dir, prefix="test" if args.split == "test" else "eval")


if __name__ == "__main__":
    main()
