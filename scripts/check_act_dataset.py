"""Validate a LeRobot dataset before ACT training: what it contains, whether it is usable, and whether
it fits a pretrained checkpoint. Do not train on a dataset because of its name.

    # laptop / login node (downloads to $HF_LEROBOT_HOME/<repo-id> unless --root is given)
    python scripts/check_act_dataset.py --repo-id omiya239532/so101_cube_dr --checkpoint omiya239532/so101_act_cotrain
    # Hopper login node, into the training container's dataset dir (internet needed once):
    bash sim/hpc_env.sh env HF_HUB_OFFLINE=0 python scripts/check_act_dataset.py --repo-id <id> \
        --root /scratch/e1583535/datasets/lerobot/<id> --checkpoint omiya239532/so101_act_cotrain

Prints: episodes, frames, fps, robot type, task strings, camera keys and sizes, joint names, per-joint
state/action statistics (also in degrees when the data is in radians), and evidence for
  * units            radians vs LeRobot degrees / gripper 0..100
  * action type      absolute targets (a_t ~ s_t+1) vs deltas (a_t ~ s_t+1 - s_t)
  * real vs sim      identical start poses / episode lengths, bare joint names (no ".pos"), robot_type
  * grasps           per episode: gripper commanded closed, jaw blocked by an object (state stays
                     above the command) and the arm moving while it is blocked (= lifted/carried)
and with --checkpoint whether the dataset can fine-tune it (camera keys, image size, fps, dims, units).
Saves a contact sheet of start / grasp / lift / end frames of a few episodes to --out-dir (--show
also opens it). Exit status 1 on structural problems, and with --strict also when fine-tuning the
checkpoint on this dataset would be invalid.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# so101_act_cotrain's gripper (%, rad) line, as in local/run_act.py
AUTHOR_GRIPPER = ((0.0, -0.0314), (100.0, 1.2006))
GRIP = JOINT_ORDER.index("gripper")
LIFT_JOINTS = [JOINT_ORDER.index("shoulder_lift"), JOINT_ORDER.index("elbow_flex")]


def detect_units(arm_min, arm_max, gripper_min, gripper_max):
    """'rad' or 'deg' (arm frame: LeRobot degrees, gripper 0..100). Radian SO-101 data stays within the joint
    limits (|q| < 2.9); degree data spans tens of degrees. Degrees and LeRobot's default -100..100 units
    (use_degrees=false) look alike and cannot be told apart from the numbers."""
    arm = "rad" if max(np.abs(arm_min).max(), np.abs(arm_max).max()) < 3.5 else "deg"
    grip = "rad" if max(abs(gripper_min), abs(gripper_max)) < 5.0 else "deg"
    if arm != grip:
        raise ValueError(f"arm joints look like {arm} but the gripper like {grip}: unknown units")
    return arm


def deg_to_rad_frame(state):
    (p0, m0), (p1, m1) = AUTHOR_GRIPPER
    return [math.radians(v) for v in state[:5]] + [m0 + (state[5] - p0) * (m1 - m0) / (p1 - p0)]


def rad_to_deg_frame(state):
    (p0, m0), (p1, m1) = AUTHOR_GRIPPER
    return [math.degrees(v) for v in state[:5]] + [p0 + (state[5] - m0) * (p1 - p0) / (m1 - m0)]


def resolve_checkpoint(name):
    p = Path(name).expanduser()
    if p.exists():
        for cand in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
            if (cand / "config.json").is_file():
                return cand.resolve()
        raise FileNotFoundError(f"no config.json in {p}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, allow_patterns=["config.json", "model.safetensors", "policy_*"]))


def joint_names(feature):
    names = feature.get("names")
    while isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]  # some datasets nest the list
    if isinstance(names, dict):
        names = next(iter(names.values()))
    return list(names) if names else None


def per_episode(values, episode_index):
    order = np.argsort(episode_index, kind="stable")
    eps, starts = np.unique(episode_index[order], return_index=True)
    return {int(e): values[order][s:e2] for e, s, e2 in zip(eps, starts, list(starts[1:]) + [len(order)])}


def action_kind(states, actions):
    """'absolute' if a_t tracks s_t+1, 'delta' if a_t tracks s_t+1 - s_t (arm joints only)."""
    err_abs, err_delta = [], []
    for s, a in zip(states.values(), actions.values()):
        if len(s) < 3:
            continue
        err_abs.append(np.abs(a[:-1, :5] - s[1:, :5]).mean())
        err_delta.append(np.abs(a[:-1, :5] - (s[1:, :5] - s[:-1, :5])).mean())
    ea, ed = float(np.mean(err_abs)), float(np.mean(err_delta))
    return ("absolute" if ea < ed else "delta"), ea, ed


def grasp_analysis(states, actions, units):
    """Per episode: closed (gripper commanded shut), grasp (jaw blocked by an object for >= 10 frames)
    and lift (arm joints move >= 10 deg while blocked). Joint space only: confirm on the frames."""
    all_cmd = np.concatenate([a[:, GRIP] for a in actions.values()])
    lo, hi = np.percentile(all_cmd, 2), np.percentile(all_cmd, 98)
    span = max(hi - lo, 1e-6)
    close_thr, block_margin = lo + 0.25 * span, 0.05 * span
    to_deg = 180.0 / math.pi if units == "rad" else 1.0
    out = {}
    for ep in states:
        s, a = states[ep], actions[ep]
        closed = a[:, GRIP] < close_thr
        blocked = closed & (s[:, GRIP] - a[:, GRIP] > block_margin)
        # longest run of blocked frames
        best, run, best_end = 0, 0, -1
        for i, b in enumerate(blocked):
            run = run + 1 if b else 0
            if run > best:
                best, best_end = run, i
        grasp = best >= 10
        lift_deg, grasp_frame = 0.0, None
        if grasp:
            start = best_end - best + 1
            grasp_frame = int(start)
            seg = s[start:best_end + 1][:, LIFT_JOINTS] * to_deg
            lift_deg = float(np.abs(seg - seg[0]).sum(axis=1).max())
        out[ep] = {"closed": bool(closed.any()), "grasp": grasp, "grasp_frames": int(best), "grasp_frame": grasp_frame,
                   "lift": grasp and lift_deg >= 10.0, "arm_motion_while_grasped_deg": round(lift_deg, 1),
                   "length": int(len(s))}
    return out, {"close_threshold": float(close_thr), "block_margin": float(block_margin)}


def sim_evidence(meta, states, lengths, names):
    ev = []
    starts = np.stack([s[0] for s in states.values()])
    spread = starts.std(axis=0)
    ranges = np.ptp(np.concatenate(list(states.values())), axis=0) + 1e-9
    if (spread / ranges).max() < 1e-3:
        ev.append(f"every episode starts from the identical pose {np.round(starts[0], 3).tolist()}")
    if len(set(lengths)) == 1 and len(lengths) > 1:
        ev.append(f"every episode has exactly {lengths[0]} frames")
    if names and not any(n.endswith(".pos") for n in names):
        ev.append("joint names have no '.pos' suffix (lerobot-record on a real SO-101 writes 'shoulder_pan.pos')")
    if meta.robot_type and "follower" not in str(meta.robot_type):
        ev.append(f"robot_type '{meta.robot_type}' (real recordings say so101_follower / so_follower)")
    return ev


def checkpoint_compat(ckpt, meta, data_units, state_mean, ckpt_fps):
    """Can this dataset fine-tune the checkpoint? Returns (finetune_ok, notes, problems)."""
    from safetensors.numpy import load_file

    ckpt_dir = resolve_checkpoint(ckpt)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    stats = {}
    for f in sorted(ckpt_dir.glob("policy_*processor*.safetensors")):
        stats.update(load_file(str(f)))
    problems, notes = [], []
    ck_images = {k: v["shape"] for k, v in cfg["input_features"].items() if v["type"] == "VISUAL"}
    ds_images = {k: meta.features[k]["shape"] for k in meta.camera_keys}
    if set(ck_images) != set(ds_images):
        problems.append(f"camera keys differ: checkpoint {sorted(ck_images)} vs dataset {sorted(ds_images)} "
                        f"(record with the checkpoint's names, or pass --rename_map; never silently)")
    for k in set(ck_images) & set(ds_images):
        c, h, w = ck_images[k]
        if tuple(ds_images[k][:2]) != (h, w):
            notes.append(f"{k}: dataset {ds_images[k][1]}x{ds_images[k][0]} vs checkpoint {w}x{h} (ACT's ResNet "
                         f"accepts it, but local/run_act.py feeds {w}x{h})")
    for key in ("observation.state", "action"):
        want = cfg["input_features" if key != "action" else "output_features"][key]["shape"]
        if list(meta.features[key]["shape"]) != list(want):
            problems.append(f"{key} shape {meta.features[key]['shape']} != checkpoint {want}")
    fps = ckpt_fps or 30.0
    if float(meta.fps) != float(fps):
        problems.append(f"fps {meta.fps} != checkpoint {fps}: its 100-step chunks would mean "
                        f"{100 / meta.fps:.1f} s instead of {100 / fps:.1f} s")
    ck_units = detect_units(stats["action.min"][:5], stats["action.max"][:5],
                            float(stats["action.min"][5]), float(stats["action.max"][5]))
    if ck_units != data_units:
        notes.append(f"units differ (checkpoint {ck_units}, dataset {data_units}). Fine-tuning still works: "
                     f"lerobot-train replaces the normalization stats with this dataset's and z-scores are unit-"
                     f"free; but never MIX this dataset with {ck_units} data or reuse the old stats")
    mean_in_ck = np.asarray(state_mean, dtype=float)
    if ck_units == "rad" and data_units == "deg":
        mean_in_ck = np.asarray(deg_to_rad_frame(list(mean_in_ck)))
    if ck_units == "deg" and data_units == "rad":
        mean_in_ck = np.asarray(rad_to_deg_frame(list(mean_in_ck)))
    z = (mean_in_ck - stats["observation.state.mean"]) / np.maximum(stats["observation.state.std"], 1e-6)
    far = [f"{n} (z={v:+.1f})" for n, v in zip(JOINT_ORDER, z) if abs(v) > 2.0]
    if far:
        notes.append("state distribution far from the checkpoint's (different zero/sign convention or workspace?): "
                     + ", ".join(far))
    return not problems, notes, problems, str(ckpt_dir)


def contact_sheet(ds, meta, episodes_info, ep_ranges, sample_eps, out_path):
    from PIL import Image, ImageDraw

    cams = list(meta.camera_keys)
    rows = []
    for ep in sample_eps:
        start, length = ep_ranges[ep]
        g = episodes_info[ep]
        picks = [("start", 0)]
        if g["grasp_frame"] is not None:
            picks += [("grasp", g["grasp_frame"]), ("lift", min(g["grasp_frame"] + g["grasp_frames"] // 2, length - 1))]
        picks += [("end", length - 1)]
        tiles = []
        for label, offset in picks:
            item = ds[start + offset]
            for cam in cams:
                img = item[cam]
                arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                tile = Image.fromarray(arr).resize((256, 192))
                ImageDraw.Draw(tile).text((4, 4), f"ep{ep} {label} f{offset} {cam.split('.')[-1]}", fill=(255, 255, 0))
                tiles.append(tile)
        rows.append(tiles)
    width = max(len(r) for r in rows) * 256
    sheet = Image.new("RGB", (width, 192 * len(rows)))
    for y, r in enumerate(rows):
        for x, tile in enumerate(r):
            sheet.paste(tile, (x * 256, y * 192))
    sheet.save(out_path)
    return out_path


def download(repo_id, root, with_videos):
    """Plain snapshot of the dataset repo into root. LeRobot's own downloader refuses repos without a
    codebase-version tag (e.g. Li-2025hang/so101-pick-lift); a local copy loads without that check."""
    from huggingface_hub import snapshot_download

    patterns = ["meta/*", "data/*"] + (["videos/*"] if with_videos else [])
    print(f"downloading {repo_id} ({', '.join(patterns)}) -> {root}")
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(root), allow_patterns=patterns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=None, help="dataset dir (default $HF_LEROBOT_HOME/<repo-id>; downloaded if missing)")
    ap.add_argument("--checkpoint", default=None, help="ACT checkpoint to check fine-tuning compatibility against")
    ap.add_argument("--checkpoint-fps", type=float, default=None, help="fps the checkpoint was trained at (default 30)")
    ap.add_argument("--video-backend", default="pyav", help="pyav (default) or torchcodec")
    ap.add_argument("--no-frames", action="store_true", help="skip videos (no download, no decoding)")
    ap.add_argument("--sample-episodes", type=int, default=3)
    ap.add_argument("--out-dir", default=None, help="default dataset_checks/<repo-id with __>")
    ap.add_argument("--show", action="store_true", help="open the contact sheet in a window")
    ap.add_argument("--strict", action="store_true", help="exit 1 if --checkpoint cannot be fine-tuned on it")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    out_dir = Path(args.out_dir or REPO_ROOT / "dataset_checks" / args.repo_id.replace("/", "__"))
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root) if args.root else Path(HF_LEROBOT_HOME) / args.repo_id
    if not (root / "meta" / "info.json").is_file() or (not args.no_frames and not (root / "videos").is_dir()):
        download(args.repo_id, root, with_videos=not args.no_frames)
    ds = LeRobotDataset(args.repo_id, root=root, download_videos=not args.no_frames,
                        video_backend=args.video_backend)
    meta = ds.meta
    errors = []

    table = ds.hf_dataset.with_format("numpy")
    S = np.asarray(table["observation.state"], dtype=np.float64)
    A = np.asarray(table["action"], dtype=np.float64)
    E = np.asarray(table["episode_index"]).reshape(-1)
    states, actions = per_episode(S, E), per_episode(A, E)
    lengths = [len(v) for v in states.values()]
    names = joint_names(meta.features["action"])
    tasks = list(meta.tasks.index) if meta.tasks is not None else []

    print(f"dataset        {args.repo_id}\nroot           {ds.root}")
    print(f"codebase       {meta.info.get('codebase_version')}   robot_type {meta.robot_type}")
    print(f"episodes       {meta.total_episodes}   frames {meta.total_frames}   fps {meta.fps}   "
          f"duration {meta.total_frames / meta.fps / 60:.1f} min")
    print(f"episode length min {min(lengths)}  mean {np.mean(lengths):.0f}  max {max(lengths)} frames "
          f"({min(lengths) / meta.fps:.1f}-{max(lengths) / meta.fps:.1f} s)")
    print(f"tasks          {tasks}")
    for cam in meta.camera_keys:
        f = meta.features[cam]
        info = f.get("info", {}) or {}
        print(f"camera         {cam}  {f['shape']} (H, W, C)  codec {info.get('video.codec', '?')}")
    print(f"joint names    {names}")

    if len(meta.camera_keys) == 0:
        errors.append("no camera features")
    for key, arr in (("observation.state", S), ("action", A)):
        if arr.shape[1] != len(JOINT_ORDER):
            errors.append(f"{key} has {arr.shape[1]} dims, expected {len(JOINT_ORDER)}")
        if not np.isfinite(arr).all():
            errors.append(f"{key} has {int((~np.isfinite(arr)).sum())} non-finite values")
    if names and [n.removesuffix(".pos") for n in names] != JOINT_ORDER:
        errors.append(f"joint order {names} != {JOINT_ORDER}")
    if errors:
        print("\nRESULT: FAIL\n  - " + "\n  - ".join(errors))
        sys.exit(1)

    units = detect_units(A.min(0)[:5], A.max(0)[:5], float(A.min(0)[GRIP]), float(A.max(0)[GRIP]))
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print(f"\nunits          {units}" + ("  (arm joints in radians; gripper is a jaw angle, NOT a 0..100 value)"
                                         if units == "rad" else "  (arm frame, gripper 0..100)"))
    unit_warning = None
    if units == "deg" and np.abs(np.concatenate([S[:, :5], A[:, :5]])).max() <= 100.5:
        unit_warning = ("every arm value is within +-100: these may be LeRobot's default -100..100 units "
                        "(use_degrees=false), not degrees. The client and server need degrees; only use this "
                        "dataset if you know it was recorded with --robot.use_degrees=true")
        print(f"WARNING        {unit_warning}")
    for key, arr in (("state", S), ("action", A)):
        print(f"{key:7s} min    {arr.min(0)}\n{key:7s} max    {arr.max(0)}\n{key:7s} mean   {arr.mean(0)}\n"
              f"{key:7s} std    {arr.std(0)}")
        if units == "rad":
            print(f"{key:7s} deg    min {np.degrees(arr.min(0)[:5]).round(1)} max {np.degrees(arr.max(0)[:5]).round(1)} (arm)")

    kind, ea, ed = action_kind(states, actions)
    print(f"\naction type    {kind} joint targets (mean |a_t - s_t+1| {ea:.4f} vs |a_t - (s_t+1 - s_t)| {ed:.4f})")
    if kind != "absolute":
        errors.append("actions look like deltas; the server/client expect absolute joint targets")

    evidence = sim_evidence(meta, states, lengths, names)
    print("real vs sim    " + ("likely SIMULATED / scripted:\n  - " + "\n  - ".join(evidence) if len(evidence) >= 2
                               else "no strong sim indicators (" + ("; ".join(evidence) or "none") + ") - check the frames"))

    grasps, thresholds = grasp_analysis(states, actions, units)
    n_closed = sum(g["closed"] for g in grasps.values())
    n_grasp = sum(g["grasp"] for g in grasps.values())
    n_lift = sum(g["lift"] for g in grasps.values())
    print(f"grasps         gripper closed in {n_closed}/{len(grasps)} episodes, jaw blocked by an object in "
          f"{n_grasp}, arm moved >= 10 deg while holding it in {n_lift}  (joint-space evidence; confirm on frames)")
    weak = [e for e, g in grasps.items() if not g["lift"]]
    if weak:
        print(f"               episodes without grasp+move evidence: {weak[:20]}{' ...' if len(weak) > 20 else ''}")

    summary = {"repo_id": args.repo_id, "root": str(ds.root), "episodes": meta.total_episodes,
               "frames": meta.total_frames, "fps": meta.fps, "robot_type": meta.robot_type, "tasks": tasks,
               "cameras": {c: meta.features[c]["shape"] for c in meta.camera_keys}, "joint_names": names,
               "units": units, "unit_warning": unit_warning, "action_type": kind, "sim_evidence": evidence, "grasp_thresholds": thresholds,
               "grasps": {"closed": n_closed, "grasp": n_grasp, "lift": n_lift, "per_episode": grasps}}

    finetune_ok = None
    if args.checkpoint:
        finetune_ok, notes, problems, ckpt_dir = checkpoint_compat(args.checkpoint, meta, units, S.mean(0),
                                                                   args.checkpoint_fps)
        print(f"\ncheckpoint     {args.checkpoint} ({ckpt_dir})")
        for n in notes:
            print(f"  note: {n}")
        for p in problems:
            print(f"  INCOMPATIBLE: {p}")
        print(f"fine-tune      {'COMPATIBLE' if finetune_ok else 'NOT compatible'} with {args.checkpoint}")
        summary["checkpoint"] = {"id": args.checkpoint, "finetune_compatible": finetune_ok, "notes": notes,
                                 "problems": problems}
    print("from scratch   " + ("possible (6-dim state/action, cameras present)" if not errors else "NO"))

    if not args.no_frames:
        ep_ranges = {}
        idx = np.arange(len(E))
        for ep in states:
            where = idx[E == ep]
            ep_ranges[ep] = (int(where[0]), len(where))
        eps = sorted(states)
        picks = sorted({eps[int(round(i))] for i in np.linspace(0, len(eps) - 1, max(1, args.sample_episodes))})
        try:
            path = contact_sheet(ds, meta, grasps, ep_ranges, picks, out_dir / "samples.png")
            print(f"\nframes         decoded with {args.video_backend}; contact sheet {path}")
            if args.show:
                import cv2

                cv2.imshow(args.repo_id, cv2.imread(str(path)))
                cv2.waitKey(0)
        except Exception as e:  # decoder problems must not pass silently
            errors.append(f"video decoding failed with {args.video_backend}: {type(e).__name__}: {e}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary        {out_dir / 'summary.json'}")
    if errors:
        print("\nRESULT: FAIL\n  - " + "\n  - ".join(errors))
        sys.exit(1)
    if args.strict and finetune_ok is False:
        print("\nRESULT: FAIL (--strict: not fine-tune compatible with the checkpoint)")
        sys.exit(1)
    print("\nRESULT: PASS")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
