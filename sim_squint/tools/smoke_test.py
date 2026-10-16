"""Smoke test of the office3 squint envs (GPU node): builds each camera setup, saves what the cameras see, checks that
the front camera matches office3's (the rendered cube's centre vs its projection through office3's MuJoCo camera),
steps random actions.

    python sim_squint/tools/smoke_test.py --out sim_squint/debug/smoke
"""

import argparse
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "third_party" / "squint"))
sys.path.append(str(HERE))
import envs  # noqa: E402,F401
import office3_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "debug" / "smoke"))
ap.add_argument("--num_envs", type=int, default=8)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

for env_id in ("SO101LiftCubeOffice3-v1", "SO101LiftCubeOffice3Wrist-v1", "SO101LiftCylinderOffice3Both-v1"):
    for dr in (False, True):
        env = gym.make(env_id, num_envs=a.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu", render_mode="all",
                       domain_randomization=dr, sensor_configs=dict(width=128, height=128))
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        tiles = []
        for name, cam in obs["sensor_data"].items():
            rgb = cam["rgb"].cpu().numpy()
            tiles.append(np.concatenate(list(rgb[: min(8, len(rgb))]), axis=1))
            print(f"{env_id} dr={dr} {name}: rgb {tuple(cam['rgb'].shape)}")
        cv2.imwrite(str(out / f"{env_id}_dr{int(dr)}.png"), cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))
        if not dr and "base_camera" in obs["sensor_data"]:
            # the cube's pixel: red in the overlaid image vs office3's projection, in 128 px of the 480 px crop
            rgb = obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy().astype(int)
            item = u.item.pose.p.cpu().numpy()
            uv = office3_env.front_camera_pixels(item)
            uv = (uv - [office3_env.CROP_PX[0], 0]) * 128 / 480
            errs = []
            for i in range(len(rgb)):
                red = (rgb[i, ..., 0] > 120) & (rgb[i, ..., 1] < 60) & (rgb[i, ..., 2] < 60)
                if red.sum() > 3:
                    vs, us = np.nonzero(red)
                    errs.append(np.hypot(us.mean() - uv[i, 0], vs.mean() - uv[i, 1]))
            print(f"  front camera vs office3: cube pixel error {np.round(errs, 1)} px of 128 (cube ~12 px wide)")
        info = u.evaluate()
        print("  touching table at start:", info["robot_touching_table"].float().mean().item(),
              " gripper tip z:", np.round(u.agent.tcp_pos[:, 2].cpu().numpy()[:4], 3))
        for _ in range(20):
            obs, rew, term, trunc, info = env.step(env.action_space.sample())
        print(f"  20 random steps ok, reward mean {rew.float().mean().item():.3f}")
        env.close()
print("SMOKE TEST PASSED")
