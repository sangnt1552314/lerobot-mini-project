"""Red-cube detection in a BGR frame: HSV threshold -> contours -> size/shape filter -> centre.

Red wraps around hue 0 in OpenCV's 0..179 hue, so a pixel is red if hue <= hue_max or hue >= 180 - hue_max.
The orange wooden table at home sits at hue 7..14 with high saturation but is much darker than the cube
(V ~80..140 vs ~255), so the defaults are tight on hue and need a bright pixel. Tune with detect mode's sliders.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    u: float  # centroid, pixels
    v: float
    area: float  # contour area, pixels
    box: list  # minAreaRect corners (4, 2)
    bbox: tuple  # x, y, w, h
    rejected: list = field(default_factory=list)  # (bbox, reason) of the blobs that were filtered out


def red_mask(bgr, det_cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = det_cfg["hue_max"], det_cfg["sat_min"], det_cfg["val_min"]
    mask = cv2.inRange(hsv, (0, s, v), (h, 255, 255)) | cv2.inRange(hsv, (180 - h, s, v), (179, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def detect_red_cube(bgr, det_cfg):
    """Largest red blob that looks like a cube, or None. With nothing accepted, returns None; call
    candidates() to see why blobs were rejected."""
    accepted, rejected = candidates(bgr, det_cfg)
    if not accepted:
        return None
    best = max(accepted, key=lambda d: d.area)
    best.rejected = rejected
    return best


def candidates(bgr, det_cfg):
    """(accepted detections, [(bbox, reason)] rejected) for every red blob."""
    contours, _ = cv2.findContours(red_mask(bgr, det_cfg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted, rejected = [], []
    for c in contours:
        area = float(cv2.contourArea(c))
        bbox = cv2.boundingRect(c)
        if area < det_cfg["min_area_px"]:
            if area >= 20:  # don't report specks
                rejected.append((bbox, f"small {area:.0f}px"))
            continue
        if area > det_cfg["max_area_px"]:
            rejected.append((bbox, f"large {area:.0f}px"))
            continue
        (_, (w, h), _) = rect = cv2.minAreaRect(c)
        fill = area / max(w * h, 1.0)
        aspect = max(w, h) / max(min(w, h), 1.0)
        if fill < det_cfg["min_fill"]:
            rejected.append((bbox, f"ragged fill {fill:.2f}"))
            continue
        if aspect > det_cfg["max_aspect"]:
            rejected.append((bbox, f"long aspect {aspect:.1f}"))
            continue
        m = cv2.moments(c)
        accepted.append(Detection(u=m["m10"] / m["m00"], v=m["m01"] / m["m00"], area=area,
                                  box=cv2.boxPoints(rect).tolist(), bbox=bbox))
    return accepted, rejected


class DetectionError(RuntimeError):
    pass


def detect_stable(read_frame, det_cfg, checks, flush=5, clock=time.monotonic):
    """Detect on several FRESH frames and require them to agree. read_frame() -> BGR frame.

    OpenCV keeps a few old frames buffered, so the first `flush` reads are thrown away. Then the cube must be
    found in each of checks["stable_frames"] frames, within checks["max_jitter_px"] of their median, and two
    consecutive frames must not be byte-identical (a frozen camera returns the same buffer forever).
    Returns (Detection at the median centre, last frame, capture time). Raises DetectionError."""
    for _ in range(flush):
        read_frame()
    dets, prev, frame = [], None, None
    for _ in range(checks["stable_frames"]):
        frame = read_frame()
        if prev is not None and np.array_equal(frame, prev):
            raise DetectionError("camera returned the same frame twice: stale/frozen stream")
        prev = frame
        det = detect_red_cube(frame, det_cfg)
        if det is None:
            raise DetectionError(f"red cube not found in all {checks['stable_frames']} frames (check detect mode)")
        dets.append(det)
    us, vs = np.array([d.u for d in dets]), np.array([d.v for d in dets])
    mu, mv = float(np.median(us)), float(np.median(vs))
    jitter = float(np.max(np.hypot(us - mu, vs - mv)))
    if jitter > checks["max_jitter_px"]:
        raise DetectionError(f"detection unstable: centres spread {jitter:.1f}px (limit {checks['max_jitter_px']})")
    best = min(dets, key=lambda d: np.hypot(d.u - mu, d.v - mv))
    best.u, best.v = mu, mv
    return best, frame, clock()


def draw(frame, det, lines=(), colour=(0, 255, 0)):
    """Box, centre and text on a copy of frame. lines: extra text rows."""
    out = frame.copy()
    if det is not None:
        cv2.polylines(out, [np.int32(det.box)], True, colour, 2)
        cv2.drawMarker(out, (int(round(det.u)), int(round(det.v))), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        for (x, y, w, h), reason in det.rejected:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(out, reason, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for i, text in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
    return out
