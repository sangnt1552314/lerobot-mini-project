"""Per-run debug folder: local/vision_pick/runs/<time>_<mode>_<arm>/ with log.txt, images and JSON."""

import datetime
import json

import cv2

from .settings import RUNS_DIR


class RunLog:
    def __init__(self, mode, arm, root=RUNS_DIR):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = root / f"{stamp}_{mode}_{arm}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        print(msg)
        with open(self.dir / "log.txt", "a") as f:  # appended as it happens, so a crash still leaves the log
            f.write(msg + "\n")

    def image(self, name, bgr):
        cv2.imwrite(str(self.dir / name), bgr)

    def json(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")
