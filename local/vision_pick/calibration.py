"""Camera -> table calibration by touch, stored as JSON in local/vision_pick/calibration/<arm>.json.

At each spot the cube's pixel comes from the fixed third-person camera, then the open gripper is lowered around
the cube by hand (torque off) and forward kinematics of that pose gives the cube's position in the robot base
frame. No tape measure and no camera intrinsics: >= 5 pairs fit a pixel -> table (x, y) map (affine below 8
spots, homography from 8), and the fingertip heights give the grasp height. Because the same cube is used, the
pixel is the blob centre the camera actually sees (top + side faces from an oblique view), so that offset is part
of the fit.

gate() lists what looks wrong: few spots, large leave-one-out error (each spot predicted from the others),
inconsistent grasp heights, a table height far from the base's z = 0, a failed gripper check.
"""

import datetime
import json
import math
from pathlib import Path

import cv2
import numpy as np

from cube_fallback import apply_h, fit_homography

from .kinematics import tip
from .settings import CALIBRATION_DIR

HOMOGRAPHY_MIN_SPOTS = 8  # fewer spots: affine. A homography has 8 unknowns and swings wildly on 5-7 clustered spots


def _fit_affine(pixels, xy):
    a = np.hstack([pixels, np.ones((len(pixels), 1))])
    m, *_ = np.linalg.lstsq(a, xy, rcond=None)
    return np.vstack([m.T, [0.0, 0.0, 1.0]])  # as a 3x3 so apply_h works for both models


def fit_table_map(pixels, xy):
    """pixel -> table (x, y). Returns (3x3 matrix, model name, residuals m, leave-one-out errors m)."""
    pixels, xy = np.asarray(pixels, dtype=float), np.asarray(xy, dtype=float)
    if len(pixels) >= HOMOGRAPHY_MIN_SPOTS:
        h, res, loo = fit_homography(pixels, xy)
        return np.asarray(h), "homography", res, loo
    h = _fit_affine(pixels, xy)
    res = np.linalg.norm(apply_h(h, pixels) - xy, axis=1)
    keep = [np.arange(len(pixels)) != i for i in range(len(pixels))]
    loo = np.array([np.linalg.norm(apply_h(_fit_affine(pixels[k], xy[k]), pixels[i:i + 1])[0] - xy[i])
                    for i, k in enumerate(keep)]) if len(pixels) > 3 else np.array([])
    return h, "affine", res, loo


def cal_path(arm: str) -> Path:
    return CALIBRATION_DIR / f"{arm}.json"


def load_calibration(arm: str, path: Path | None = None):
    p = Path(path) if path else cal_path(arm)
    return json.loads(p.read_text()) if p.is_file() else None


def save_calibration(cal: dict, arm: str, path: Path | None = None) -> Path:
    p = Path(path) if path else cal_path(arm)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2) + "\n")
    return p


def pixel_to_table(cal: dict, u: float, v: float):
    """Pixel -> (x, y) metres in the robot base frame on the table plane."""
    x, y = apply_h(cal["homography"], [[u, v]])[0]
    return float(x), float(y)


def table_to_pixel(cal: dict, x: float, y: float):
    """Inverse of pixel_to_table (for drawing robot targets on the image)."""
    u, v = apply_h(np.linalg.inv(np.asarray(cal["homography"])), [[x, y]])[0]
    return float(u), float(v)


def inside_workspace(cal: dict, x: float, y: float, margin: float) -> bool:
    """Inside the convex hull of the calibration spots, grown by margin (m)."""
    hull = cv2.convexHull(np.asarray(cal["table_xy"], dtype=np.float32))
    return cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -margin


def build_calibration(*, arm, robot_id, camera_index, image_size, cube_size_m, spots, gripper_open, gripper_contact):
    """spots: [{"pixel": (u, v), "joints": 6 arm-frame values at the grasp}]. Returns the calibration dict."""
    pixels = np.array([s["pixel"] for s in spots], dtype=float)
    tips, tilts = zip(*(tip(s["joints"]) for s in spots))
    tips = np.array(tips)
    h, model, res, loo = fit_table_map(pixels, tips[:, :2])
    grasp_z = float(np.median(tips[:, 2]))
    return {
        "arm": arm,
        "robot_id": robot_id,
        "camera_index": camera_index,
        "image_size": list(image_size),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cube_size_m": cube_size_m,
        "pixels": pixels.round(1).tolist(),
        "joints": [np.round(s["joints"], 2).tolist() for s in spots],
        "table_xy": tips[:, :2].round(4).tolist(),
        "tip_z": tips[:, 2].round(4).tolist(),
        "tilt_deg": np.round(tilts, 1).tolist(),
        "grasp_z": round(grasp_z, 4),
        "table_z": round(grasp_z - cube_size_m / 2, 4),
        "wrist_roll_grasp": round(float(np.median([s["joints"][4] for s in spots])), 2),
        "map_model": model,
        "homography": np.asarray(h).tolist(),  # the pixel -> table matrix (affine ones have last row 0 0 1)
        "fit_residual_cm": (res * 100).round(2).tolist(),
        "leave_one_out_cm": (loo * 100).round(2).tolist(),
        "gripper_open": round(float(gripper_open), 1),
        "gripper_contact": round(float(gripper_contact), 1),
    }


def gate(cal, settings, image_size=None):
    """(calibration looks good, reasons it does not). Control mode needs a measured calibration and prints these
    reasons as warnings, but still moves."""
    if cal is None:
        return False, ["no calibration: run --mode calibrate"]
    if cal.get("example"):
        return False, ["this is the example calibration, not a measured one: run --mode calibrate"]
    c = settings["checks"]
    reasons = []
    n = len(cal.get("pixels", []))
    if n < c["min_points"]:
        reasons.append(f"{n} calibration spots, need {c['min_points']}")
    loo = cal.get("leave_one_out_cm") or [math.inf]
    if max(loo) > c["max_loo_cm"]:
        reasons.append(f"leave-one-out error {max(loo):.1f} cm > {c['max_loo_cm']} cm: re-touch the worst spots or add more")
    spread = (max(cal["tip_z"]) - min(cal["tip_z"])) * 100 if cal.get("tip_z") else math.inf
    if spread > c["max_height_spread_cm"]:
        reasons.append(f"grasp heights spread {spread:.1f} cm > {c['max_height_spread_cm']} cm: joint calibration or "
                       f"touch heights inconsistent")
    if abs(cal.get("table_z", math.inf)) * 100 > c["max_table_z_cm"]:
        reasons.append(f"table at z = {cal.get('table_z', math.inf) * 100:.1f} cm from the touches, expected ~0 (the base "
                       f"sits on the table): joint calibration / cube size / touch height off")
    if abs(cal.get("cube_size_m", 0) - settings["cube"]["size_m"]) > 0.002:
        reasons.append(f"calibrated with a {cal.get('cube_size_m', 0) * 100:.1f} cm cube, settings say "
                       f"{settings['cube']['size_m'] * 100:.1f} cm")
    if cal.get("gripper_open", 0) - cal.get("gripper_contact", 0) < 10:
        reasons.append(f"gripper check failed: open {cal.get('gripper_open')} vs closed-on-cube "
                       f"{cal.get('gripper_contact')} (closing must lower the reading by > 10)")
    if image_size is not None and list(image_size) != list(cal.get("image_size", [])):
        reasons.append(f"camera gives {list(image_size)}, calibrated at {cal.get('image_size')}")
    return not reasons, reasons


def draw_workspace(frame, cal, colour=(255, 200, 0)):
    """Calibration spots and their hull, projected into the image."""
    pts = np.int32(cal["pixels"])
    hull = cv2.convexHull(pts)
    cv2.polylines(frame, [hull], True, colour, 1)
    for u, v in pts:
        cv2.circle(frame, (int(u), int(v)), 4, colour, -1)
    return frame
