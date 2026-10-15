"""Collect ACT demonstrations with the vision_pick cube controller, no leader arm: place the cube, press Enter, the
arm picks and lifts it while both cameras, the joints and every command sent are recorded; keep or drop it.
Run from the repo root inside `conda activate lerobot` (needs a calibration: python -m local.vision_pick --mode calibrate):

    python local/collect_act.py --episodes 50     # until the dataset holds 50 saved episodes; resumes where it stopped
    python local/collect_act.py --step            # debug: Enter before every planned move, nothing recorded

Dataset: LeRobot v3.0 at $HF_LEROBOT_HOME/local/so101_redcube_lift (~/.cache/huggingface/lerobot/...), 30 fps, the
format of `lerobot-record --robot.use_degrees=true` with cameras world/wrist, as omiya239532/so101_act_cotrain takes:
  observation.state         measured joints: LeRobot degrees, gripper 0..100, names shoulder_pan.pos ... gripper.pos
  action                    the command actually sent after the safety checks: absolute joint targets, same units
  observation.images.world  third-person camera   } 480x640 RGB, cropped to 4:3 and resized exactly as
  observation.images.wrist  wrist camera          } local/run_act.py feeds the policy
One episode = start pose -> above -> open -> down -> close -> lift -> 1 s holding the cube up. Putting the cube back and
returning to the start pose are not recorded. Each saved episode is finalized on disk at once, so a crash loses at most
the episode being recorded. <dataset>/collect/ holds the start pose, every attempt (cube position, outcome, rate),
images of failures and split.json (45/5 train/validation for scripts/train_act.sh EPISODES_FILE).
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from local.vision_pick import hardware  # noqa: E402  (also puts local/ on sys.path)
from local.vision_pick.calibration import gate, load_calibration  # noqa: E402
from local.vision_pick.controller import Abort, Executor, is_via  # noqa: E402
from local.vision_pick.detection import DetectionError, detect_stable  # noqa: E402
from local.vision_pick.planner import gripper_bounds, plan_pick  # noqa: E402
from local.vision_pick.settings import arm_settings, load_settings, with_overrides  # noqa: E402

from action_format import JOINT_ORDER  # noqa: E402
from run_act_policy import crop_to_aspect  # noqa: E402  (the same image preprocessing as local/run_act.py)

TASK = "Pick up the red cube and lift it."
FPS = 30
IMAGE_HW = (480, 640)  # omiya239532/so101_act_cotrain's input size; local/run_act.py resizes to the checkpoint's
HOLD_S = 0.5  # recorded hold with the cube up, so the policy learns to stop there
NEAR_CM = 1.5  # warn when a new cube position is this close to a saved one
NAMES = [f"{n}.pos" for n in JOINT_ORDER]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
    **{f"observation.images.{cam}": {"dtype": "video", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channels"]}
       for cam in ("world", "wrist")},
}
PHASES = {"above": "Moving above the cube", "open": "Opening the gripper", "descend": "Approaching the cube",
          "close": "Closing the gripper", "lift": "Lifting"}


def to_policy_image(bgr):
    h, w = IMAGE_HW
    frame = crop_to_aspect(bgr, w / h)
    return cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


# ---- dataset -----------------------------------------------------------------------------------------------------

def open_dataset(repo_id, root):
    """Resume the dataset at root if it exists (never overwritten), else create it."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (root / "meta" / "info.json").is_file():
        if (root / "images").exists():  # frames of an episode that was being recorded when a session died
            shutil.rmtree(root / "images")
        ds = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4, video_backend="pyav")
        feats = ds.meta.features
        bad = [k for k, f in FEATURES.items() if k not in feats or tuple(feats[k]["shape"]) != f["shape"]]
        if bad or int(ds.meta.fps) != FPS:
            sys.exit(f"{root} holds a dataset with other features/fps ({bad or ds.meta.fps}); use another --repo-id")
        return ds
    if root.exists():
        sys.exit(f"{root} exists but is not a LeRobot dataset; move it away or use another --repo-id")
    return LeRobotDataset.create(repo_id, fps=FPS, features=FEATURES, root=root, robot_type="so101_follower",
                                 use_videos=True, image_writer_threads=4, video_backend="pyav")


def save_durably(ds, repo_id, root):
    """save_episode, then finalize (parquet footers and metadata hit the disk) and reopen for the next episode.
    Videos are encoded in this process: LeRobot's process pool for parallel encoding died on macOS here."""
    ds.save_episode(parallel_encoding=False)
    ds.finalize()
    return open_dataset(repo_id, root)


def drop_pending(ds):
    try:
        if ds.has_pending_frames():
            ds.clear_episode_buffer()
    except Exception:  # the buffer of a failed save is half torn down; finalize() still keeps the saved episodes
        pass


class Recorder:
    """Executor.on_step: one dataset frame per control step, with the state measured and the command sent at that
    step and the newest frame of each camera (a camera slower than 30 fps repeats frames rather than slowing the
    loop; one that stops for 0.5 s stops the episode)."""

    def __init__(self, ds, cams):
        self.ds, self.cams, self.times, self.busy = ds, cams, [], []

    def __call__(self, measured, sent):
        t0 = time.monotonic()
        wrist, third = self.cams.latest()
        self.ds.add_frame({"observation.state": np.asarray(measured, dtype=np.float32),
                           "action": np.asarray(sent, dtype=np.float32),
                           "observation.images.world": to_policy_image(third),
                           "observation.images.wrist": to_policy_image(wrist),
                           "task": TASK})
        self.times.append(time.monotonic())
        self.busy.append(self.times[-1] - t0)

    def timing(self):
        """Where a step's time goes: recording (cameras + dataset) vs the rest (robot I/O, control, sleep)."""
        step = 1000.0 / self.rate() if self.rate() else 0.0
        rec = 1000.0 * float(np.mean(self.busy)) if self.busy else 0.0
        return f"step {step:.0f} ms: recording {rec:.0f} ms, robot + control {step - rec:.0f} ms"

    def rate(self):
        return (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

    def worst_gap_s(self):
        return float(np.max(np.diff(self.times))) if len(self.times) > 1 else 0.0


# ---- collection log ----------------------------------------------------------------------------------------------

class CollectLog:
    def __init__(self, root, write=True):
        self.dir, self.write = root / "collect", write
        if write:
            self.dir.mkdir(parents=True, exist_ok=True)

    def start_pose(self):
        p = self.dir / "session.json"
        return json.loads(p.read_text())["start_pose"] if p.is_file() else None

    def save_start_pose(self, pose, arm, cal):
        if self.write:
            (self.dir / "session.json").write_text(json.dumps(
                {"start_pose": [round(v, 2) for v in pose], "arm": arm, "calibration_created_at": cal["created_at"],
                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")

    def add(self, **record):
        if self.write:
            with open(self.dir / "attempts.jsonl", "a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}, default=float) + "\n")

    def saved_positions(self):
        p = self.dir / "attempts.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()] if p.is_file() else []
        return [r["table_xy"] for r in rows if r.get("outcome") == "saved"]

    def image(self, name, bgr):
        if self.write:
            cv2.imwrite(str(self.dir / name), bgr)

    def write_split(self, n):
        """Every 10th episode (9, 19, ...; 5 of 50) for validation, spread over the session; the rest for training."""
        val = list(range(9, n, 10))[:5] or ([n - 1] if n > 1 else [])
        split = {"train": [i for i in range(n) if i not in val], "val": val}
        if self.write:
            (self.dir / "split.json").write_text(json.dumps(split) + "\n")
        return split


def coverage_lines(cal, positions, margin):
    """3 x 3 counts of saved cube positions over the calibrated area, in robot terms."""
    xy = np.asarray(cal["table_xy"])
    lo, hi = xy.min(0) - margin, xy.max(0) + margin
    counts = np.zeros((3, 3), int)
    for x, y in positions:
        i = int(np.clip((x - lo[0]) / (hi[0] - lo[0]) * 3, 0, 2))  # 0 = near the base
        j = int(np.clip((hi[1] - y) / (hi[1] - lo[1]) * 3, 0, 2))  # 0 = robot's left (+y)
        counts[i, j] += 1
    rows, names = ["far ", "mid ", "near"], ["left", "centre", "right"]
    lines = ["  saved cube positions (robot's view: left/right, far/near the base)",
             "         " + "".join(f"{n:>8s}" for n in names)]
    lines += [f"  {rows[2 - i]}   " + "".join(f"{counts[i, j]:8d}" for j in range(3)) for i in (2, 1, 0)]
    i, j = np.unravel_index(np.argmin(counts), counts.shape)
    lines.append(f"  fewest so far: {rows[2 - i].strip()}-{names[j]}")
    return lines


# ---- motion ------------------------------------------------------------------------------------------------------

def go_to_start(executor, start):
    here = executor.measured()
    if max(abs(a - b) for a, b in zip(here[:5], start[:5])) > 3.0:
        print("  moving to the start pose")
        executor.move_to(start, slow=True, label="start")


def run_pick(executor, plan, recorder, step_mode):
    """The planned pick, then HOLD_S holding the cube up. Records every control step when recorder is given."""
    executor.on_step = recorder
    try:
        phase = None
        for wp in plan.pick:
            name = "descend" if wp.name == "grasp" else wp.name.rstrip("0123456789")
            if step_mode and input(f"  next move: {wp.name}  [Enter]  [q] stop ").strip().lower() == "q":
                raise Abort("stopped at the step prompt")
            if name != phase:
                phase = name
                print(f"  {PHASES.get(name, name)}...")
            executor.move_to(wp.joints, wp.slow, wp.name, settle=not is_via(wp.name))
        executor.dwell(HOLD_S)
    finally:
        executor.on_step = None


def recover(executor, plan, start):
    """After an abort mid-pick: straight back up to the approach height, then to the start pose."""
    input("  The arm holds where it stopped. Clear anything in its way, then [Enter] = back up and to the start pose "
          "(Ctrl+C quits) ")
    above = plan.pick[0].joints[:5] + [executor.cmd[5]]
    executor.move_to(above, slow=True, label="recover")
    executor.move_to(start, slow=True, label="start")


def ask(prompt, choices):
    while True:
        key = input(prompt).strip().lower()
        if key in choices:
            return key


# ---- main --------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50, help="stop when the dataset holds this many saved episodes")
    ap.add_argument("--arm", default="home", help="arm/rig in local/vision_pick/settings.json (home, office)")
    ap.add_argument("--repo-id", default="local/so101_redcube_lift")
    ap.add_argument("--step", action="store_true", help="debug: Enter before every planned move; nothing recorded")
    args = ap.parse_args(argv)

    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = Path(HF_LEROBOT_HOME) / args.repo_id
    settings = with_overrides(load_settings(), motion={"fps": FPS})
    arm_cfg = arm_settings(settings, args.arm)
    cal = load_calibration(args.arm)
    if cal is None or cal.get("example"):
        sys.exit(f"no measured calibration for --arm {args.arm}: run python -m local.vision_pick --mode calibrate")

    recording = not args.step  # (not `if ds`: a LeRobotDataset with no frames yet is falsy, it has __len__)
    ds = open_dataset(args.repo_id, root) if recording else None
    log = CollectLog(root, write=recording)
    collected = ds.meta.total_episodes if recording else 0
    print(f"\nSO-101 ACT Data Collector\n=========================\ndataset {root}\n"
          f"Target: {args.episodes} successful demonstrations\nCollected: {collected} / {args.episodes}"
          + ("\nSTEP MODE: Enter before every move, nothing is recorded" if args.step else ""))
    if recording and collected >= args.episodes:
        ds.finalize()
        finish(log, root, collected, args.repo_id, arm_cfg)
        return

    cams = hardware.open_cameras(arm_cfg, threaded=True)
    rates = cams.frame_rates()
    print(f"cameras: world {rates['third']:.0f} fps, wrist {rates['wrist']:.0f} fps")
    if min(rates.values()) < 0.9 * FPS:
        print(f"WARNING: a camera runs below {FPS} fps (dim light lowers webcam fps): episodes will record slower")
    frame = hardware.read_third(cams)
    for reason in gate(cal, settings, (frame.shape[1], frame.shape[0]))[1]:
        print(f"WARNING calibration: {reason}")

    robot = hardware.open_arm(arm_cfg, torque="on")
    executor, start, stopped = None, None, False
    try:
        executor = Executor(robot, settings, cal, arm_cfg["gripper_floor"],
                            log=lambda m: None if m.startswith("  -> ") else print(m))
        start = log.start_pose()
        if start is None:
            start = executor.measured()
            start[5] = min(max(start[5], gripper_bounds(arm_cfg["gripper_floor"])[0]), 100.0)
            log.save_start_pose(start, args.arm, cal)
            print(f"start pose = the arm's pose now, kept for every episode and later sessions: "
                  f"{[round(v, 1) for v in start]}")
        else:
            print(f"start pose of this dataset: {[round(v, 1) for v in start]}")
        if input("Control mode: every Enter runs a whole pick. Type 'yes' to enable: ").strip() != "yes":
            print("not enabled")
            return
        go_to_start(executor, start)

        retry = False
        while collected < args.episodes:
            print(f"\nProgress: {collected} / {args.episodes}")
            if recording and not retry:
                print("\n".join(coverage_lines(cal, log.saved_positions(), settings["grasp"]["workspace_margin_m"])))
            print("Place the red cube within the calibrated area." if not retry else "Same position: retry.")
            if ask("[Enter] Start episode   [q] Quit\n>>> ", ("", "q")) == "q":
                break
            retry = False
            go_to_start(executor, start)
            try:
                det, _, _ = detect_stable(lambda: hardware.read_third(cams), settings["detection"], settings["checks"])
            except (DetectionError, RuntimeError) as e:
                print(f"no episode: {e}")
                continue
            plan = plan_pick((det.u, det.v), cal, settings, start, arm_cfg["gripper_floor"])
            x, y = plan.table_xy
            print(f"Cube detected at pixel ({det.u:.0f}, {det.v:.0f}) -> robot x {x * 100:.1f} y {y * 100:.1f} cm.")
            if not plan.ok:
                print("Not executable, nothing moved:\n  " + "\n  ".join(plan.problems))
                continue
            warnings = list(plan.warnings)
            near = [math.hypot(x - px, y - py) * 100 for px, py in log.saved_positions()]
            if near and min(near) < NEAR_CM:
                warnings.append(f"only {min(near):.1f} cm from a saved episode: move the cube somewhere new for variety")
            for w in warnings:
                print(f"WARNING: {w}")
            if warnings and ask("Run anyway? [y/N] ", ("y", "n", "")) != "y":
                continue
            print("Planner ready.")

            recorder = Recorder(ds, cams) if recording else None
            print(f"Recording episode {collected + 1}..." if recording else "Running (not recorded)...")
            attempt = {"pixel": [round(det.u, 1), round(det.v, 1)], "table_xy": [round(x, 4), round(y, 4)]}
            try:
                run_pick(executor, plan, recorder, args.step)
            except Abort as e:
                print(f"STOPPED: {e}")
                if recording:
                    ds.clear_episode_buffer()
                log.add(outcome="aborted", reason=str(e), **attempt)
                log.image(f"aborted_{time.strftime('%Y%m%d_%H%M%S')}.jpg", hardware.read_third(cams))
                recover(executor, plan, start)
                continue

            holding, grip = executor.holding_cube(plan.close_pct, cal["gripper_contact"])
            print("\nEpisode finished. " + ("The gripper stopped on the cube." if holding else
                                             "The gripper closed fully: probably MISSED." if holding is False else
                                             "Check by eye whether the cube is held."))
            if recorder:
                attempt.update(frames=len(recorder.times), fps=round(recorder.rate(), 1), gripper=round(grip, 1))
                print(f"{len(recorder.times)} frames ({len(recorder.times) / FPS:.1f} s of data) at {recorder.rate():.1f} fps "
                      f"(longest step {recorder.worst_gap_s() * 1000:.0f} ms; {recorder.timing()})")
                if recorder.rate() < 0.9 * FPS or recorder.worst_gap_s() > 3.0 / FPS:
                    print(f"WARNING: recorded slower than {FPS} fps or with a long gap; better to retry")
            if recording:
                choice = ask("[s] Save successful episode   [r] Discard and retry   [f] Mark failed   [q] Quit safely\n>>> ",
                             ("s", "r", "f", "q"))
            else:
                choice = ask("[Enter] continue   [q] Quit\n>>> ", ("", "q")) or "next"
            if choice == "s":
                print("Saving (encoding videos)...")
                ds = save_durably(ds, args.repo_id, root)
                collected = ds.meta.total_episodes
                log.add(outcome="saved", episode_index=collected - 1, **attempt)
                print(f"Episode {collected} saved! Progress: {collected} / {args.episodes}")
            elif recording:
                ds.clear_episode_buffer()
                log.add(outcome={"r": "discarded", "f": "failed", "q": "discarded"}[choice], **attempt)
                if choice == "f":
                    log.image(f"failed_{time.strftime('%Y%m%d_%H%M%S')}.jpg", hardware.read_third(cams))
            retry = choice == "r"
            print("Putting the cube back and returning to the start pose...")
            executor.run(plan.place)
            if choice == "q":
                break
            if not retry:
                print("Reposition the cube.")
    except (KeyboardInterrupt, Abort, RuntimeError) as e:
        stopped = True
        print(f"\nSTOPPED: {e or 'Ctrl+C'}")
        if executor:
            executor.hold()
        if recording:
            drop_pending(ds)
    finally:
        if recording:
            ds.finalize()
        if stopped and executor and executor.sent and not at_start(executor, start):
            try:
                input("The arm is holding its position. Support it, then press Enter to turn torque off ")
            except (EOFError, KeyboardInterrupt):
                pass
        hardware.close_arm(robot, disable_torque=True)
        cams.close()
    if recording:
        finish(log, root, collected, args.repo_id, arm_cfg)


def at_start(executor, start):
    try:
        return max(abs(a - b) for a, b in zip(executor.measured()[:5], start[:5])) <= 5.0
    except Exception:
        return False


def finish(log, root, collected, repo_id, arm_cfg):
    print(f"\n{collected} episodes in {root}")
    if not collected:
        return
    split = log.write_split(collected)
    print(f"split ({log.dir / 'split.json'}): validation episodes {split['val']}, training the other "
          f"{len(split['train'])}")
    start = log.start_pose() or []
    hopper = f"/scratch/e1583535/datasets/lerobot/{repo_id}"
    run = f"act_{repo_id.split('/')[-1]}_finetune"
    print(f"""
next (README "ACT demonstrations from vision_pick"):
  python scripts/check_act_dataset.py --repo-id {repo_id} --checkpoint omiya239532/so101_act_cotrain --strict
  rsync -av {root}/ <you>@<hopper login>:{hopper}/
  qsub -v DATASET_REPO_ID={repo_id},EPISODES_FILE={hopper}/collect/split.json,STEPS=10000 pbs/train_act.pbs
  python scripts/eval_act_offline.py --checkpoint outputs/{run}
  cd local && python run_act.py --checkpoint ../outputs/{run} --mode control --front-cam {arm_cfg['third_cam']} \\
      --wrist-cam {arm_cfg['wrist_cam']} --gripper-floor {arm_cfg['gripper_floor']} --start-pose={','.join(f'{v:g}' for v in start)}""")

if __name__ == "__main__":
    main()
