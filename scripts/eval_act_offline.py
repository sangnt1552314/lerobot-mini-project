"""Held-out check of a trained ACT checkpoint on the validation episodes it was not trained on: how far its actions
are from the recorded ones, now and 1 s ahead in its chunk, next to "hold still" (keep the current state) as the
baseline a useless policy gets. Recorded actions are slow ramps right next to the state, so the 1 s column is the
informative one. Only the real robot measures success (README: evaluate with local/run_act.py).

    python scripts/eval_act_offline.py --checkpoint outputs/act_so101_redcube_lift_finetune
    python scripts/eval_act_offline.py --checkpoint <dir> --repo-id local/so101_redcube_lift --episodes 9,19 --stride 5

Validation episodes: "val" in <dataset>/collect/split.json (written by local/collect_act.py) unless --episodes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

from leisaac_ext.act_policy import LocalACTPolicy  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def resolve_checkpoint(path):
    p = Path(path).expanduser()
    for cand in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
        if (cand / "config.json").is_file():
            return cand
    sys.exit(f"no checkpoint in {p} (config.json there or in checkpoints/last/pretrained_model)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="run dir or pretrained_model dir")
    ap.add_argument("--repo-id", default="local/so101_redcube_lift")
    ap.add_argument("--root", default=None, help="dataset dir (default $HF_LEROBOT_HOME/<repo-id>)")
    ap.add_argument("--episodes", default=None, help="comma-separated episode indices (default: split.json val)")
    ap.add_argument("--stride", type=int, default=10, help="evaluate every N-th frame")
    ap.add_argument("--ahead", type=int, default=30, help="steps ahead in the chunk for the second column (30 = 1 s)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = Path(args.root) if args.root else Path(HF_LEROBOT_HOME) / args.repo_id
    if args.episodes:
        episodes = [int(e) for e in args.episodes.split(",")]
    else:
        episodes = json.loads((root / "collect" / "split.json").read_text())["val"]
    ds = LeRobotDataset(args.repo_id, root=root, episodes=episodes, video_backend="pyav")
    policy = LocalACTPolicy(resolve_checkpoint(args.checkpoint), device=args.device)
    policy.policy.config.n_action_steps = policy.policy.config.chunk_size  # pop a whole chunk from one observation
    h = args.ahead
    print(f"checkpoint {policy.checkpoint}\nvalidation episodes {episodes}: {len(ds)} frames, every {args.stride}th")

    now, ahead, hold = [], [], []
    for i in range(0, len(ds) - h, args.stride):
        item, later = ds[i], ds[i + h]
        if int(later["episode_index"]) != int(item["episode_index"]):
            continue
        state = item["observation.state"].numpy()
        obs = {"observation.state": state}
        for k in policy.camera_keys:
            img = item[f"observation.images.{k}"].permute(1, 2, 0).numpy()
            obs[f"observation.images.{k}"] = (img * 255).round().astype(np.uint8)
        policy.reset()
        chunk = [policy(obs, task=item["task"], robot_type="so101_follower") for _ in range(h + 1)]
        now.append(np.abs(chunk[0] - item["action"].numpy()))
        ahead.append(np.abs(chunk[h] - later["action"].numpy()))
        hold.append(np.abs(state - later["action"].numpy()))
    if not now:
        sys.exit("no frames to evaluate (episodes shorter than --ahead?)")

    n = len(now)
    now, ahead, hold = (np.mean(x, axis=0) for x in (now, ahead, hold))
    print(f"\nmean |error| over {n} samples (degrees; gripper 0..100)")
    print(f"{'joint':14s} {'action now':>11s} {f'{h} steps ahead':>15s} {'hold still':>11s}")
    for j, name in enumerate(JOINTS):
        print(f"{name:14s} {now[j]:11.2f} {ahead[j]:15.2f} {hold[j]:11.2f}")
    print(f"{'all':14s} {now.mean():11.2f} {ahead.mean():15.2f} {hold.mean():11.2f}")
    verdict = "beats" if ahead.mean() < 0.5 * hold.mean() else "does NOT clearly beat"
    print(f"\n{h} steps ahead the policy {verdict} holding still ({ahead.mean():.1f} vs {hold.mean():.1f})")


if __name__ == "__main__":
    main()
