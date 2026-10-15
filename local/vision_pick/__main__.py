"""Pick up a red cube with the real SO-101: OpenCV detection + touch-calibrated homography + IK. Run from the repo
root inside `conda activate lerobot`:

    python -m local.vision_pick --mode detect      # camera + detection only; tune HSV sliders (s saves them)
    python -m local.vision_pick --mode calibrate   # torque OFF: touch the cube with the gripper at >= 5 spots
    python -m local.vision_pick --mode preview     # detect -> robot xy -> full plan + checks; never moves
    python -m local.vision_pick --mode control     # same, then type 'yes': grasp, lift, hold, put back

--arm home|office picks port, robot id, gripper floor and camera indices from settings.json (default home).
--camera N overrides the third-person camera index. --image FILE runs detect mode on a saved frame.
Every preview/control run leaves a folder in local/vision_pick/runs/ (log, images, plan, result).
"""

import argparse
import math

import cv2

from action_format import JOINT_ORDER

from . import hardware
from .calibration import (build_calibration, cal_path, draw_workspace, gate, inside_workspace, load_calibration,
                          pixel_to_table, save_calibration, table_to_pixel)
from .controller import Abort, Executor
from .detection import DetectionError, detect_red_cube, detect_stable, draw, red_mask
from .kinematics import tip
from .planner import format_plan, plan_pick
from .runlog import RunLog
from .settings import RUNS_DIR, arm_settings, load_settings, save_settings

ENTER_KEYS = (13, 10, 32)  # Enter / Return / Space
QUIT_KEYS = (ord("q"), 27)


def joints_of(robot):
    state = robot.get_state()
    return [float(state[n]) for n in JOINT_ORDER]


# ---- detect ----------------------------------------------------------------------------------------------------

def cmd_detect(args, settings, arm_cfg):
    det_cfg = settings["detection"]
    cal = load_calibration(args.arm)
    win = "vision_pick detect"
    cv2.namedWindow(win)
    sliders = [("hue_max", 30), ("sat_min", 255), ("val_min", 255), ("min_area_px", 5000)]
    for name, top in sliders:
        cv2.createTrackbar(name, win, int(det_cfg[name]), top, lambda _: None)
    cams = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            raise SystemExit(f"can't read {args.image}")

        def read():
            return still
    else:
        cams = hardware.open_cameras(arm_cfg, args.camera)

        def read():
            return hardware.read_third(cams)
    print("keys: m = mask view, s = save thresholds to settings.json, p = save the frame, q = quit. The robot never moves.")
    show_mask = False
    try:
        while True:
            frame = read()
            for name, _ in sliders:
                det_cfg[name] = cv2.getTrackbarPos(name, win)
            det = detect_red_cube(frame, det_cfg)
            view = cv2.cvtColor(red_mask(frame, det_cfg), cv2.COLOR_GRAY2BGR) if show_mask else frame.copy()
            lines = [f"cube pixel ({det.u:.0f}, {det.v:.0f})  area {det.area:.0f}px" if det else "no cube"]
            if cal and list(cal["image_size"]) == [frame.shape[1], frame.shape[0]]:
                draw_workspace(view, cal)
                if det:
                    x, y = pixel_to_table(cal, det.u, det.v)
                    inside = inside_workspace(cal, x, y, settings["grasp"]["workspace_margin_m"])
                    lines.append(f"robot x {x * 100:.1f} y {y * 100:.1f} cm  ({'inside' if inside else 'OUTSIDE'} "
                                 f"calibrated area)")
            elif cal is None:
                lines.append("not calibrated: no robot coordinates")
            cv2.imshow(win, draw(view, det, lines))
            key = cv2.waitKey(30) & 0xFF
            if key in QUIT_KEYS:
                break
            if key == ord("m"):
                show_mask = not show_mask
            elif key == ord("s"):
                save_settings(settings)
                print(f"saved thresholds { {n: det_cfg[n] for n, _ in sliders} } to settings.json")
            elif key == ord("p"):
                out = RUNS_DIR / "snapshots"
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"third_{len(list(out.glob('*.png'))):03d}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")
    finally:
        if cams:
            cams.close()
        cv2.destroyAllWindows()


# ---- calibrate -------------------------------------------------------------------------------------------------

def cmd_calibrate(args, settings, arm_cfg):
    det_cfg, checks = settings["detection"], settings["checks"]
    cams = hardware.open_cameras(arm_cfg, args.camera)
    robot = hardware.open_arm(arm_cfg, torque="off")
    third, _ = hardware.camera_indices(arm_cfg, args.camera)
    win = "vision_pick calibrate"
    spots = []
    print(f"Torque is OFF: the arm drops if you let go. Rest it on the table between spots.\n"
          f"Spread >= {checks['min_points']} spots (6-9 is better) over the area where the cube will be, including its "
          f"corners. Keys go to the camera window: SPACE/Enter = record, u = undo, f = finish, q = quit.")

    def live(lines, show_tip=False):
        while True:
            frame = hardware.read_third(cams)
            det = detect_red_cube(frame, det_cfg)
            extra = []
            if show_tip:
                joints = joints_of(robot)
                p, tilt = tip(joints)
                extra = [f"tips ({p[0] * 100:.1f}, {p[1] * 100:.1f}, {p[2] * 100:.1f}) cm, {tilt:.0f} deg off vertical, "
                         f"gripper {joints[5]:.0f}"]
            view = frame.copy()
            for s in spots:
                cv2.circle(view, (int(s["pixel"][0]), int(s["pixel"][1])), 5, (255, 200, 0), -1)
            cv2.imshow(win, draw(view, det, lines + extra))
            key = cv2.waitKey(30) & 0xFF
            if key in ENTER_KEYS:
                return "record", frame
            if key in QUIT_KEYS:
                return "quit", frame
            if key in (ord("u"), ord("f")):
                return chr(key), frame

    try:
        while True:
            n = len(spots) + 1
            key, frame = live([f"spot {n}: place the cube, keep the arm out of the camera's view of it",
                               f"SPACE record cube | u undo | f finish ({len(spots)}/{checks['min_points']}+) | q quit"])
            if key == "quit":
                print("quit: nothing saved")
                return
            if key == "u":
                if spots:
                    spots.pop()
                    print(f"removed spot {n - 1}")
                continue
            if key == "f":
                if len(spots) >= checks["min_points"]:
                    break
                print(f"need at least {checks['min_points']} spots, have {len(spots)}")
                continue
            try:
                det, frame, _ = detect_stable(lambda: hardware.read_third(cams), det_cfg, checks)
            except DetectionError as e:
                print(f"spot {n}: {e}")
                continue
            image_size = (frame.shape[1], frame.shape[0])
            while True:
                key, _ = live([f"spot {n}: cube at ({det.u:.0f}, {det.v:.0f}). Lower the OPEN gripper around it:",
                               "fingers vertical, cube centred between the jaws, tips half-way down",
                               "SPACE record pose | q skip this spot"], show_tip=True)
                if key != "record":
                    break
                joints = joints_of(robot)
                p, tilt = tip(joints)
                if tilt > 20:
                    print(f"spot {n}: fingers {tilt:.0f} deg off vertical; straighten them and record again")
                    continue
                spots.append({"pixel": (det.u, det.v), "joints": joints})
                print(f"spot {n}: pixel ({det.u:.0f}, {det.v:.0f}) -> fingertips ({p[0] * 100:.1f}, {p[1] * 100:.1f}, "
                      f"{p[2] * 100:.1f}) cm, {tilt:.0f} deg off vertical, wrist_roll {joints[4]:.0f}")
                break

        key, _ = live(["gripper check: put the cube between the jaws and squeeze", "the gripper closed on it by hand",
                       "SPACE record | q quit"], show_tip=True)
        if key != "record":
            print("quit: nothing saved")
            return
        contact = joints_of(robot)[5]
        key, _ = live([f"closed on the cube at {contact:.0f}. Now open the gripper fully by hand", "SPACE record | q quit"],
                      show_tip=True)
        if key != "record":
            print("quit: nothing saved")
            return
        opened = joints_of(robot)[5]
    finally:
        hardware.close_arm(robot, disable_torque=True)
        cams.close()
        cv2.destroyAllWindows()

    cal = build_calibration(arm=args.arm, robot_id=arm_cfg["robot_id"], camera_index=third, image_size=image_size,
                            cube_size_m=settings["cube"]["size_m"], spots=spots, gripper_open=opened,
                            gripper_contact=contact)
    path = save_calibration(cal, args.arm)
    print(f"\nsaved {path}")
    print(f"  pixel -> table map  {cal['map_model']} ({len(spots)} spots; a homography needs 8+)")
    print(f"  fit residual (cm)   {cal['fit_residual_cm']}")
    print(f"  leave-one-out (cm)  {cal['leave_one_out_cm']}   <- each spot predicted from the others")
    print(f"  fingertip z (cm)    {[round(z * 100, 1) for z in cal['tip_z']]}  -> grasp at {cal['grasp_z'] * 100:.1f} cm, "
          f"table at {cal['table_z'] * 100:.1f} cm (expected ~0)")
    print(f"  wrist_roll at grasp {cal['wrist_roll_grasp']:.1f} deg; gripper open {opened:.0f}, closed on cube {contact:.0f}")
    ok, reasons = gate(cal, settings)
    print("calibration looks good" if ok else
          "calibration WARNINGS (control still moves, but expect misses):\n  " + "\n  ".join(reasons))


# ---- preview / control -----------------------------------------------------------------------------------------

def annotate(frame, det, cal, plan):
    view = draw_workspace(frame.copy(), cal)
    u, v = table_to_pixel(cal, *plan.table_xy)  # where the grasp goes (with any grasp offsets)
    cv2.drawMarker(view, (int(round(u)), int(round(v))), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
    lines = [f"cube pixel ({det.u:.0f}, {det.v:.0f}) -> robot x {plan.table_xy[0] * 100:.1f} y {plan.table_xy[1] * 100:.1f} cm",
             f"grasp z {plan.grasp_z * 100:.1f} cm  " + ("plan OK" if plan.ok else "NOT EXECUTABLE")]
    lines += [p[:70] for p in plan.problems[:3]]
    return draw(view, det, lines, colour=(0, 255, 0) if plan.ok else (0, 0, 255))


def detect_and_plan(args, settings, arm_cfg, cal, cams, robot, log):
    det, frame, _ = detect_stable(lambda: hardware.read_third(cams), settings["detection"], settings["checks"])
    current = joints_of(robot)
    plan = plan_pick((det.u, det.v), cal, settings, current, arm_cfg["gripper_floor"])
    log(f"arm now {[round(v, 1) for v in current]}")
    log(format_plan(plan))
    image = annotate(frame, det, cal, plan)
    log.image("third.png", image)
    log.image("third_raw.png", frame)
    log.image("wrist.png", hardware.read_wrist(cams))
    log.json("plan.json", plan.to_dict())
    return det, plan, image


def show(image, title, wait_ms):
    cv2.imshow(title, image)
    cv2.waitKey(wait_ms)


def cmd_preview(args, settings, arm_cfg):
    cal = load_calibration(args.arm)
    if cal is None:
        raise SystemExit(f"no calibration at {cal_path(args.arm)}: run --mode calibrate first")
    log = RunLog("preview", args.arm)
    cams = hardware.open_cameras(arm_cfg, args.camera)
    robot = hardware.open_arm(arm_cfg, torque="untouched")
    try:
        frame = hardware.read_third(cams)
        for reason in gate(cal, settings, (frame.shape[1], frame.shape[0]))[1]:
            log(f"WARNING calibration: {reason}")
        _, plan, image = detect_and_plan(args, settings, arm_cfg, cal, cams, robot, log)
        log(f"PREVIEW: no motor commands were sent. Run folder: {log.dir}")
        print("press any key in the image window to close")
        show(image, "vision_pick preview", 0)
    except DetectionError as e:
        log(f"STOP: {e}")
    finally:
        hardware.close_arm(robot, disable_torque=False)
        cams.close()
        cv2.destroyAllWindows()


def cmd_control(args, settings, arm_cfg):
    cal = load_calibration(args.arm)
    if cal is None or cal.get("example"):
        raise SystemExit(f"no measured calibration at {cal_path(args.arm)}: run --mode calibrate first")
    checks = settings["checks"]
    log = RunLog("control", args.arm)
    cams = hardware.open_cameras(arm_cfg, args.camera)
    frame = hardware.read_third(cams)
    for reason in gate(cal, settings, (frame.shape[1], frame.shape[0]))[1]:
        log(f"WARNING calibration: {reason}")
    log("connecting: the servos are configured and torque goes ON (the arm holds where it is)")
    robot = hardware.open_arm(arm_cfg, torque="on")
    executor = None
    result = {"success": False, "stage": "plan"}
    try:
        det, plan, image = detect_and_plan(args, settings, arm_cfg, cal, cams, robot, log)
        if not plan.ok:
            raise Abort("plan not executable: " + "; ".join(plan.problems))
        show(image, "vision_pick control", 500)
        if input("The arm will move. Type 'yes' to grasp and lift (Ctrl+C stops at any time): ").strip() != "yes":
            raise Abort("not confirmed")
        # a fresh look right before moving: refuse a plan made from a stale view
        det2, _, _ = detect_stable(lambda: hardware.read_third(cams), settings["detection"], checks)
        shift = math.hypot(det2.u - det.u, det2.v - det.v)
        if shift > checks["max_shift_px"]:
            raise Abort(f"the cube moved {shift:.0f}px since planning: run again")
        executor = Executor(robot, settings, cal, arm_cfg["gripper_floor"], log=log)
        moved = max(abs(a - b) for a, b in zip(executor.cmd[:5], plan.place[-1].joints[:5]))
        if moved > 3.0:
            raise Abort(f"the arm moved {moved:.0f} deg since planning: run again")

        result["stage"] = "pick"
        executor.run(plan.pick, pause_after="open",
                     pause=lambda _: input("Above the cube with the gripper open. Enter = descend and grasp "
                                           "(Ctrl+C aborts) "))
        holding, grip = executor.holding_cube(plan.close_pct, cal["gripper_contact"])
        result.update(success=holding, gripper_after_lift=round(grip, 1), close_target=plan.close_pct, stage="hold")
        if holding is None:
            log(f"LIFTED; gripper at {grip:.0f}. Close target {plan.close_pct:.0f} is too near the cube's width "
                f"({cal['gripper_contact']:.0f}) to tell from the gripper: look at the arm (and set success by hand)")
        elif holding:
            log(f"LIFTED, holding the cube (gripper stopped at {grip:.0f}, cube contact {cal['gripper_contact']:.0f})")
        else:
            log(f"MISSED: the gripper closed to {grip:.0f} (cube contact {cal['gripper_contact']:.0f}), nothing "
                f"between the jaws")
        log.image("third_lifted.png", hardware.read_third(cams))
        log.image("wrist_lifted.png", hardware.read_wrist(cams))
        input("Holding. Enter = put the cube back down and return to the start pose ")
        result["stage"] = "place"
        executor.run(plan.place)
        result["stage"] = "done"
        log("done: back at the start pose")
    except (Abort, DetectionError) as e:
        result["reason"] = str(e)
        log(f"STOPPED: {e}")
    except KeyboardInterrupt:
        result["reason"] = "Ctrl+C"
        log("\nSTOPPED by Ctrl+C")
        if executor:
            executor.hold()
    finally:
        log.json("result.json", result)
        if executor and executor.sent and result["stage"] != "done":
            try:
                input("The arm is holding its position. Support it, then press Enter to turn torque off ")
            except (EOFError, KeyboardInterrupt):
                pass
        hardware.close_arm(robot, disable_torque=True)
        cams.close()
        cv2.destroyAllWindows()
        log(f"run folder: {log.dir}")


def main():
    ap = argparse.ArgumentParser(prog="python -m local.vision_pick", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["detect", "calibrate", "preview", "control"])
    ap.add_argument("--arm", default="home", help="arm/rig in settings.json (home, office)")
    ap.add_argument("--camera", type=int, default=None, help="third-person camera index (default: from settings.json)")
    ap.add_argument("--image", default=None, help="detect mode: a saved frame instead of the camera")
    args = ap.parse_args()
    settings = load_settings()
    arm_cfg = arm_settings(settings, args.arm)
    {"detect": cmd_detect, "calibrate": cmd_calibrate, "preview": cmd_preview, "control": cmd_control}[args.mode](
        args, settings, arm_cfg)


if __name__ == "__main__":
    main()
