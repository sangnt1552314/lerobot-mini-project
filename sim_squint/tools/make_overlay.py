"""Background image for squint's overlay ("greenscreen") from a real front-camera photo of the office3 desk.

Squint renders only the robot and the object and composites them over a real photo of the empty scene. The arm can't
leave the D435's view, so the photo is one with the arm folded (sim_mujoco/assets/office3/photo_front_folded.png) and
the arm is painted out: its dark pixels inside a box around it (+ a dilation) are inpainted from the desk around them.
Then centre-cropped to a square as deploy.py crops the real frames. The policy sees it at 16 x 16, so leftovers of the
arm hardly matter; a photo of the desk with the arm out of the way would be better (pass it with --photo --no-inpaint).

The wrist camera moves, so no single photo is its background: it gets a flat image of the desk's colour (the median
of the desk below the arm in the same photo), --wrist-out.

    python sim_squint/tools/make_overlay.py [--photo P] [--out sim_squint/assets/office3_front_overlay.png]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--photo", default=str(REPO / "sim_mujoco/assets/office3/photo_front_folded.png"))
ap.add_argument("--out", default=str(REPO / "sim_squint/assets/office3_front_overlay.png"))
ap.add_argument("--wrist-out", default=str(REPO / "sim_squint/assets/office3_wrist_overlay.png"))
ap.add_argument("--box", type=int, nargs=4, default=(330, 20, 475, 340), help="x0 y0 x1 y1 of the arm (640x480 px)")
ap.add_argument("--dark", type=int, default=110, help="pixels darker than this (max RGB) in the box are the arm")
ap.add_argument("--no-inpaint", action="store_true")
a = ap.parse_args()

img = cv2.imread(a.photo)
assert img.shape[:2] == (480, 640), img.shape
if not a.no_inpaint:
    x0, y0, x1, y1 = a.box
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = (img[y0:y1, x0:x1].max(axis=2) < a.dark).astype(np.uint8) * 255
    # the arm's white cables and bright highlights: close the gaps between the dark parts
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    img = cv2.inpaint(img, mask, 15, cv2.INPAINT_TELEA)
    cv2.imwrite(str(Path(a.out).with_name(Path(a.out).stem + "_mask.png")), mask)
h, w = img.shape[:2]
s = min(h, w)
img = img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
cv2.imwrite(a.out, img)
print("wrote", a.out, img.shape)
desk = img[s * 2 // 3 :].reshape(-1, 3)   # the bottom third of the square: desk only
cv2.imwrite(a.wrist_out, np.full((128, 128, 3), np.median(desk, axis=0), np.uint8))
print("wrote", a.wrist_out, "desk BGR", np.median(desk, axis=0))
