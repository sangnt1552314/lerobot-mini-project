"""Map where a profile's object can spawn, to set config.py's object_region from the setup instead of guessing it.

    python sim_mujoco/tools/workspace_scan.py --profile office3 --object cube [--step 0.015] [--workers 8] \
        [--out sim_mujoco/debug/workspace_office3_cube]

A grid point (the object's centre) is valid when the object, at its real and at its largest size:
  - is on the table, >= --edge_margin from its edges,
  - is fully inside the front camera's image (nominal pose, >= --pixel_margin px from the border at 640x480),
  - is >= object_clearance from the robot at the start pose (the folded gripper; placements.py re-checks this),
  - is lifted by the scripted expert (both sizes, a random yaw each) with a plan that keeps the wrist camera clear.
Saves map.png (green valid; grey off the table / out of view; blue too close to the start pose; orange expert unsafe;
red expert failed) and scan.json, and prints the convex hull of the valid points and the share of invalid points
inside it (too-close points excluded: placements.py removes those).
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_liftcube import OBJECTS, PROFILES, LiftCubeEnv, ScriptedLiftExpert, make_config  # noqa: E402
from so101_liftcube.env import point_in_polygon, rot_z  # noqa: E402
from so101_liftcube.placements import Placement  # noqa: E402

_worker = {}


def _init(profile, obj):
    from dataclasses import replace

    cfg = make_config(profile, obj)
    cfg = replace(cfg, object_region=None)   # the scan must not depend on the region it is computing
    env = LiftCubeEnv(cfg, render=False)
    _worker.update(cfg=cfg, env=env, expert=ScriptedLiftExpert(env))


def sizes(cfg):
    o = cfg.obj
    real = (o.half_width, o.half_height)
    if not o.half_width_range:
        return [real]
    return [real, (o.half_width_range[1], (o.half_height_range or o.half_width_range)[1])]


def corners(x, y, z0, hw, hh, yaw_deg):
    """The 8 corners of the object's bounding box (a cylinder's too) at this yaw."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    pts = []
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for z in (z0, z0 + 2 * hh):
            pts.append((x + c * dx * hw - s * dy * hw, y + s * dx * hw + c * dy * hw, z))
    return np.array(pts)


def geometry_check(xy, args):
    """'off_table', 'not_visible', 'too_close' or None, from the nominal scene (no randomization)."""
    import mujoco

    cfg, env = _worker["cfg"], _worker["env"]
    ap = cfg.appearance
    x, y = xy
    tc, th = np.array(ap.table_center), np.array(ap.table_half)
    if not (np.abs(np.array([x, y]) - tc) <= th - args.edge_margin).all():
        return "off_table"
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    d.qpos[env.qadr] = cfg.home_qpos
    mujoco.mj_forward(m, d)
    cam = env.front_cam
    R, cp, fovy = d.cam_xmat[cam].reshape(3, 3), d.cam_xpos[cam], m.cam_fovy[cam]
    W, H = cfg.width, cfg.height
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    for hw, hh in sizes(cfg):
        pts = corners(x, y, cfg.table_z, hw, hh, 45.0)
        local = (pts - cp) @ R            # camera frame: x right, y up, -z forward
        if (local[:, 2] >= 0).any():
            return "not_visible"
        u = W / 2 + f * local[:, 0] / -local[:, 2]
        v = H / 2 - f * local[:, 1] / -local[:, 2]
        if (u < args.pixel_margin).any() or (u > W - args.pixel_margin).any() or (v < args.pixel_margin).any() or (v > H - args.pixel_margin).any():
            return "not_visible"
    hw, hh = sizes(cfg)[-1]
    if cfg.obj.half_width_range:
        env.set_object_size(hw, hh)
    a = env.object_qadr
    d.qpos[a : a + 3] = [x, y, cfg.table_z + hh]
    d.qpos[a + 3 : a + 7] = rot_z(45.0)
    close = cfg.object_clearance > 0 and env._object_robot_distance() < cfg.object_clearance
    if cfg.obj.half_width_range:
        env.set_object_size(cfg.obj.half_width, cfg.obj.half_height)
    return "too_close" if close else None


def scan_point(job):
    i, xy, args = job
    cfg, env, expert = _worker["cfg"], _worker["env"], _worker["expert"]
    why = geometry_check(xy, args)
    if why:
        return i, why, {}
    info = {}
    r = np.random.default_rng([7, i])
    for k, (hw, hh) in enumerate(sizes(cfg)):
        yaw = float(r.uniform(-cfg.object_yaw_deg, cfg.object_yaw_deg))
        p = Placement("scan", i, 0, -1, 900_000_000 + 10 * i + k, float(xy[0]), float(xy[1]), yaw, hw, hh, k > 0)
        env.reset(seed=p.seed, placement=p)
        expert.plan()
        for t in range(len(expert)):
            env.step(expert.action(t))
        info[f"size{k}"] = {"ik_pos_err_m": round(expert.ik_error[0], 4), "camera_clearance_m": round(float(expert.min_clearance), 4),
                            "lifted": bool(env.is_success())}
        if not expert.safe:
            return i, "unsafe", info
        if not env.is_success():
            return i, "failed", info
    return i, "ok", info


def convex_hull(pts):
    pts = sorted(map(tuple, pts))

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and np.cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--object", choices=OBJECTS, default="cube")
    ap.add_argument("--step", type=float, default=0.015)
    ap.add_argument("--x", type=float, nargs=2, default=(-0.05, 0.34))
    ap.add_argument("--y", type=float, nargs=2, default=(-0.40, 0.34))
    ap.add_argument("--edge_margin", type=float, default=0.03)
    ap.add_argument("--pixel_margin", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or Path(__file__).resolve().parents[1] / "debug" / f"workspace_{args.profile}_{args.object}")
    out.mkdir(parents=True, exist_ok=True)

    xs = np.arange(args.x[0], args.x[1] + 1e-9, args.step)
    ys = np.arange(args.y[0], args.y[1] + 1e-9, args.step)
    grid = [(float(x), float(y)) for x in xs for y in ys]
    jobs = [(i, xy, args) for i, xy in enumerate(grid)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.profile, args.object)) as pool:
        results = sorted(pool.map(scan_point, jobs, chunksize=4))
    status = {i: s for i, s, _ in results}
    counts = {s: sum(v == s for v in status.values()) for s in sorted(set(status.values()))}
    print(f"{len(grid)} grid points ({args.step * 100:.1f} cm): {counts}")

    ok = np.array([grid[i] for i, s in status.items() if s == "ok"])
    hull = convex_hull(ok)
    inside_bad = [(grid[i], s) for i, s in status.items() if s in ("failed", "unsafe", "not_visible", "off_table")
                  and point_in_polygon(grid[i], hull)]
    inside = sum(point_in_polygon(grid[i], hull) for i, s in status.items() if s != "too_close")
    print(f"convex hull of the valid points ({len(hull)} vertices): {np.round(hull, 3).tolist()}")
    print(f"invalid points inside it: {len(inside_bad)}/{inside} {[(np.round(p, 3).tolist(), s) for p, s in inside_bad]}")

    # map: x up (away from the robot), y left, as seen from above with the robot at the bottom
    import cv2

    px = 12
    img = np.full((len(xs) * px, len(ys) * px, 3), 255, np.uint8)
    colour = {"ok": (60, 180, 60), "off_table": (200, 200, 200), "not_visible": (150, 150, 150), "too_close": (220, 120, 40),
              "unsafe": (0, 140, 255), "failed": (40, 40, 220)}
    for i, s in status.items():
        xi, yi = divmod(i, len(ys))
        r0, c0 = (len(xs) - 1 - xi) * px, (len(ys) - 1 - yi) * px
        img[r0 + 1 : r0 + px - 1, c0 + 1 : c0 + px - 1] = colour[s]
    poly = [((len(ys) - 1 - (p[1] - ys[0]) / args.step) * px + px // 2, (len(xs) - 1 - (p[0] - xs[0]) / args.step) * px + px // 2) for p in hull]
    cv2.polylines(img, [np.array(poly, np.int32)], True, (0, 0, 0), 1)
    bx, by = (len(ys) - 1 - (0 - ys[0]) / args.step) * px + px // 2, (len(xs) - 1 - (0 - xs[0]) / args.step) * px + px // 2
    cv2.circle(img, (int(bx), int(by)), 6, (0, 0, 0), -1)   # robot base
    cv2.imwrite(str(out / "map.png"), img)
    (out / "scan.json").write_text(json.dumps({
        "profile": args.profile, "object": args.object, "args": {k: v for k, v in vars(args).items() if k != "out"},
        "counts": counts, "hull": np.round(hull, 4).tolist(),
        "points": [{"xy": grid[i], "status": s, **info} for i, s, info in results]}, indent=1))
    print(f"saved {out}/map.png and scan.json")


if __name__ == "__main__":
    main()
