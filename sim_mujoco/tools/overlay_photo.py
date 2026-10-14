"""Overlay a sim camera render on a real photo, to fit a profile's camera, start pose and object region.

    python sim_mujoco/tools/overlay_photo.py --profile office --object cylinder \
        --photo local/debug_frames/act_front.png --out overlay.png \
        [--camera front|wrist] [--object-xy 0.20 0.05] [--qpos-deg PAN LIFT ELBOW WFLEX ROLL GRIP] \
        [--cam-pos X Y Z --cam-lookat X Y Z --fovy F --cam-roll DEG] [--wrist-fovy F]

Renders the profile's start pose (or --qpos-deg, arm-frame degrees + gripper 0..100 as
local/capture_setup.py prints them) without noise or randomization, with the profile's mean colour cast,
and writes photo | sim | 50 % blend side by side. Move the camera / pose until the robot and the table
edges line up, then copy the values into so101_liftcube/config.py. Runs on a laptop (no GPU needed).
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_liftcube import OBJECTS, PROFILES, LiftCubeEnv, make_config  # noqa: E402
from so101_liftcube.config import CameraConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--object", choices=OBJECTS, default="cube")
    ap.add_argument("--photo", required=True, help="real camera image (any size; resized to the sim's)")
    ap.add_argument("--camera", choices=["front", "wrist"], default="front")
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--object-xy", type=float, nargs=2, default=None, help="object position (default: spawn centre)")
    ap.add_argument("--qpos-deg", type=float, nargs=6, default=None, help="arm-frame start pose (default: profile)")
    ap.add_argument("--cam-pos", type=float, nargs=3, default=None)
    ap.add_argument("--cam-lookat", type=float, nargs=3, default=None)
    ap.add_argument("--fovy", type=float, default=None, help="front camera fovy (deg)")
    ap.add_argument("--cam-roll", type=float, default=None, help="front camera roll about its viewing axis (deg)")
    ap.add_argument("--wrist-fovy", type=float, default=None, help="default: middle of the profile's range")
    args = ap.parse_args()

    cfg = make_config(args.profile, args.object)
    front = cfg.front
    if args.cam_pos or args.cam_lookat or args.fovy or args.cam_roll is not None:
        front = CameraConfig(pos=tuple(args.cam_pos or front.pos), lookat=tuple(args.cam_lookat or front.lookat),
                             fovy_deg=args.fovy or front.fovy_deg,
                             roll_deg=front.roll_deg if args.cam_roll is None else args.cam_roll)
    still = dict(tint_jitter=0.0, light_jitter=0.0, light_dir_jitter=0.0, front_fovy_jitter_deg=0.0, camera_gain_jitter=0.0)
    cfg = dataclasses.replace(cfg, front=front, home_noise=0.0, camera_jitter_pos=0.0, camera_jitter_deg=0.0,
                              appearance=dataclasses.replace(cfg.appearance, **still))
    env = LiftCubeEnv(cfg, render=True)
    env.reset(seed=0)

    q = np.array(cfg.home_qpos, float)
    if args.qpos_deg:
        q[:5] = np.radians(args.qpos_deg[:5])
        lo, hi = env.jrange[5]
        q[5] = lo + args.qpos_deg[5] / 100 * (hi - lo)
    env.data.qpos[env.qadr] = q
    xy = args.object_xy or [np.mean(cfg.object_x), np.mean(cfg.object_y)]
    a = env.object_qadr
    env.data.qpos[a : a + 7] = [*xy, cfg.obj.half_height, 1.0, 0.0, 0.0, 0.0]
    wr = cfg.appearance.wrist_fovy_range
    if args.wrist_fovy or wr:
        env.model.cam_fovy[env.wrist_cam] = args.wrist_fovy or float(np.mean(wr))
    mujoco.mj_forward(env.model, env.data)
    sim = env.render("front" if args.camera == "front" else "wrist_cam")

    photo = np.asarray(Image.open(args.photo).convert("RGB").resize((cfg.width, cfg.height)))
    blend = ((photo.astype(np.float32) + sim) / 2).astype(np.uint8)
    Image.fromarray(np.concatenate([photo, sim, blend], axis=1)).save(args.out)
    deg = [round(math.degrees(v), 1) for v in q[:5]]
    if args.camera == "front":
        print(f"front camera: pos {front.pos} lookat {front.lookat} fovy {front.fovy_deg} roll {front.roll_deg}")
    else:
        print(f"wrist camera: fovy {env.model.cam_fovy[env.wrist_cam]:.1f}")
    print(f"pose arm deg {deg} gripper rad {q[5]:.3f}; object at {np.round(xy, 3).tolist()} -> {args.out}")


if __name__ == "__main__":
    main()
