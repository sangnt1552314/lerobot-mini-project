"""Validate a recorded LeRobot dataset and save the frames ACT will see.

    python sim/tools/inspect_dataset.py --repo-id local/so101_liftcube_dualcam_sim

Prints episodes, fps, frames, feature shapes and task strings, and exits non-zero if a
camera is missing, a dimension is wrong, values contain NaN, the actions never change, or
there are fewer than --min-episodes episodes. Saves the first/middle/last frame of
episode 0 from both cameras to --out-dir.

Needs only LeRobot.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[1]
CAMERAS = ("front", "wrist")


def fail(errors: list[str]) -> None:
    print("\nRESULT: FAIL")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None, help="Dataset dir. Default: $HF_LEROBOT_HOME/<repo-id>.")
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--out-dir", default=str(SIM_DIR / "debug" / "dataset"))
    parser.add_argument("--video-backend", default=None, help="torchcodec (LeRobot default) or pyav.")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from PIL import Image

    root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    if not (root / "meta" / "info.json").is_file():
        fail([f"no LeRobot dataset at {root} (missing meta/info.json). HF_LEROBOT_HOME={os.environ.get('HF_LEROBOT_HOME')}"])

    ds = LeRobotDataset(args.repo_id, root=root, video_backend=args.video_backend)
    meta = ds.meta
    features = meta.features
    tasks = list(meta.tasks.index) if meta.tasks is not None else []
    errors: list[str] = []

    def shape(key):
        return tuple(features[key]["shape"]) if key in features else None

    print(f"Dataset: {args.repo_id}\nRoot:    {root}\n")
    print(f"number of episodes     {meta.total_episodes}")
    print(f"fps                    {meta.fps}")
    print(f"number of frames       {meta.total_frames}")
    print(f"observation.state      {shape('observation.state')}")
    print(f"action                 {shape('action')}")
    for cam in CAMERAS:
        print(f"{'observation.images.' + cam:<22} {shape('observation.images.' + cam)}")
    print(f"task                   {tasks}")
    if "action" in features:
        print(f"action names           {features['action'].get('names')}")

    # --- structural checks ----------------------------------------------------------
    if meta.total_episodes == 0:
        errors.append("dataset has zero episodes (only successful episodes are saved)")
    elif meta.total_episodes < args.min_episodes:
        errors.append(f"only {meta.total_episodes} episodes, expected at least {args.min_episodes}")
    for cam in CAMERAS:
        if f"observation.images.{cam}" not in features:
            errors.append(f"{cam} camera missing (observation.images.{cam})")
    if shape("observation.state") != (args.state_dim,):
        errors.append(f"observation.state shape {shape('observation.state')} != ({args.state_dim},)")
    if shape("action") != (args.action_dim,):
        errors.append(f"action shape {shape('action')} != ({args.action_dim},)")
    if not tasks or not all(t.strip() for t in tasks):
        errors.append(f"missing/empty task string: {tasks}")
    if errors or meta.total_episodes == 0:
        fail(errors)

    # --- numeric checks over every frame --------------------------------------------
    table = ds.hf_dataset.with_format("numpy")
    for key in ("observation.state", "action"):
        values = np.asarray(table[key], dtype=np.float64)
        if np.isnan(values).any():
            errors.append(f"{key} contains {int(np.isnan(values).sum())} NaN values")
            continue
        print(f"\n{key} min  {np.round(values.min(0), 2)}")
        print(f"{key} max  {np.round(values.max(0), 2)}")
        if key == "action":
            if np.allclose(values.std(0), 0.0):
                errors.append("action is constant across the dataset; the expert never moved")
            if (np.abs(values[:, :5]) > 100.5).any() or (values[:, 5] < -0.5).any() or (values[:, 5] > 100.5).any():
                print("WARNING: some actions fall outside LeRobot's SO-101 range (arm +-100, gripper 0..100)")
    lengths = np.asarray(meta.episodes["length"])
    print(f"\nepisode length (frames) min {lengths.min()}  mean {lengths.mean():.0f}  max {lengths.max()}")

    # --- decode frames of episode 0 and save what ACT sees ---------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep0_len = int(lengths[0])
    for label, idx in (("first", 0), ("middle", ep0_len // 2), ("last", ep0_len - 1)):
        frame = ds[idx]
        for cam in CAMERAS:
            img = frame[f"observation.images.{cam}"]  # (C, H, W) float in [0, 1]
            h, w, c = shape(f"observation.images.{cam}")
            if tuple(img.shape) != (c, h, w):
                errors.append(f"decoded {cam} frame {tuple(img.shape)} != {(c, h, w)}")
                continue
            if img.isnan().any():
                errors.append(f"decoded {cam} frame {idx} contains NaN")
                continue
            path = out_dir / f"ep0_{label}_{cam}.png"
            Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)
    print(f"\nSaved episode-0 frames to {out_dir}/ep0_{{first,middle,last}}_{{front,wrist}}.png")

    if errors:
        fail(errors)
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
