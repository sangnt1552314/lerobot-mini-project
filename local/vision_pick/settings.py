"""Settings: local/vision_pick/settings.json (edit it; detect mode's `s` key writes the HSV thresholds back)."""

import copy
import json
from pathlib import Path

from . import PACKAGE_DIR

SETTINGS_PATH = PACKAGE_DIR / "settings.json"
CALIBRATION_DIR = PACKAGE_DIR / "calibration"
RUNS_DIR = PACKAGE_DIR / "runs"


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def save_settings(settings: dict, path: Path = SETTINGS_PATH) -> None:
    Path(path).write_text(json.dumps(settings, indent=2) + "\n")


def arm_settings(settings: dict, arm: str) -> dict:
    if arm not in settings["arms"]:
        raise SystemExit(f"unknown arm {arm!r}; settings.json has {sorted(settings['arms'])}")
    return settings["arms"][arm]


def with_overrides(settings: dict, **sections) -> dict:
    """Copy of settings with some keys replaced, e.g. with_overrides(s, motion={"fps": 1000}) (tests)."""
    out = copy.deepcopy(settings)
    for section, values in sections.items():
        out[section].update(values)
    return out
