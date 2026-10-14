"""Preview the MuJoCo LiftCube scene and the scripted expert without recording anything.

    python sim_mujoco/scripts/preview.py [--episodes 10] [--out sim_mujoco/debug/preview]

Saves what ACT will see (front + wrist PNGs at the start, the grasp and the end of episode 0, and a
front|wrist mp4 of episode 0), prints the expert's success rate and rendering speed.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_liftcube import OBJECTS, LiftCubeConfig, LiftCubeEnv, PROFILES, make_config, ScriptedLiftExpert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "debug" / "preview"))
    ap.add_argument("--profile", choices=PROFILES, default="default",
                    help="physical setup the sim is matched to (start pose, camera, textures, randomization)")
    ap.add_argument("--object", choices=OBJECTS, default="cube", help="object to lift (so101_liftcube/config.py OBJECTS)")
    args = ap.parse_args()

    import imageio.v2 as imageio
    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_config(args.profile, args.object)
    env = LiftCubeEnv(cfg, render=True)
    expert = ScriptedLiftExpert(env)
    ok, n_frames, t_render = 0, 0, 0.0
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        expert.plan()
        video, snaps = [], {}
        t0 = time.perf_counter()
        for t in range(len(expert)):
            if ep == 0:
                video.append(np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1))
                if t in (0, int(len(expert) * 0.55)):
                    snaps["start" if t == 0 else "grasp"] = obs
            obs = env.step(expert.action(t))
        t_render += time.perf_counter() - t0
        n_frames += len(expert)
        ok += env.is_success()
        print(f"episode {ep}: {'SUCCESS' if env.is_success() else 'FAILED'} ({len(expert)} frames, object z {env.object_pos[2]:.3f})")
        if ep == 0:
            snaps["end"] = obs
            for name, o in snaps.items():
                for cam in ("front", "wrist"):
                    Image.fromarray(o[f"observation.images.{cam}"]).save(out / f"ep0_{name}_{cam}.png")
            imageio.mimwrite(out / "ep0_front_wrist.mp4", video, fps=cfg.fps, codec="libx264", quality=7, macro_block_size=8)
    print(f"\nexpert success {ok}/{args.episodes}; {n_frames / t_render:.0f} env steps/s incl. physics + 2 cameras")
    print(f"saved {out}/ep0_{{start,grasp,end}}_{{front,wrist}}.png and ep0_front_wrist.mp4")


if __name__ == "__main__":
    main()
