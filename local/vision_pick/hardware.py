"""Opening the cameras (local/cameras.py) and the arm (local/robot.py) the way each mode needs them."""

import os
import threading
import time


class ThreadedCameras:
    """Both cameras of a local/cameras.py CameraManager read continuously in background threads, so frames are
    the newest ones instead of OpenCV's buffered, older ones. capture() returns (wrist, third) like CameraManager,
    waiting for a frame newer than the last one returned from each camera; a camera that stops delivering raises
    (stale stream)."""

    def __init__(self, cams, timeout_s=1.0):
        self.cams, self.timeout_s = cams, timeout_s
        self._latest, self._seen = {}, {"wrist": 0, "third": 0}
        self._cond, self._stop = threading.Condition(), False
        self._threads = [threading.Thread(target=self._read, args=(name, cap), daemon=True)
                         for name, cap in (("wrist", cams.wrist_cap), ("third", cams.third_cap))]
        for t in self._threads:
            t.start()

    def _read(self, name, cap):
        count = 0
        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            count += 1
            with self._cond:
                self._latest[name] = (count, frame, time.monotonic())
                self._cond.notify_all()

    def capture(self):
        deadline = time.monotonic() + self.timeout_s
        with self._cond:
            while not all(n in self._latest and self._latest[n][0] > self._seen[n] for n in self._seen):
                left = deadline - time.monotonic()
                if left <= 0:
                    stalled = [n for n in self._seen if n not in self._latest or self._latest[n][0] <= self._seen[n]]
                    raise RuntimeError(f"{' and '.join(stalled)} camera delivered no new frame in {self.timeout_s:g} s")
                self._cond.wait(left)
            for n in self._seen:
                self._seen[n] = self._latest[n][0]
            return self._latest["wrist"][1], self._latest["third"][1]

    def latest(self, max_age_s=0.5):
        """(wrist, third) newest frames without waiting: a camera slower than the control loop repeats its frame
        instead of slowing the loop down. Raises if a camera's newest frame is older than max_age_s (stale)."""
        now = time.monotonic()
        with self._cond:
            stale = [n for n in self._seen if n not in self._latest or now - self._latest[n][2] > max_age_s]
            if stale:
                raise RuntimeError(f"{' and '.join(stale)} camera: no frame in the last {max_age_s:g} s")
            return self._latest["wrist"][1], self._latest["third"][1]

    def frame_rates(self, seconds=1.0):
        """Measured frames per second of each camera."""
        with self._cond:
            start = {n: self._latest.get(n, (0,))[0] for n in self._seen}
        time.sleep(seconds)
        with self._cond:
            return {n: (self._latest.get(n, (0,))[0] - start[n]) / seconds for n in self._seen}

    def close(self):
        self._stop = True
        for t in self._threads:
            t.join(timeout=1.0)
        self.cams.close()


def camera_indices(arm_cfg, third_override=None):
    """(third, wrist) OpenCV indices. --camera N picks the third-person camera; the wrist camera is then the
    other configured index, so the two can be swapped from the command line."""
    third, wrist = arm_cfg["third_cam"], arm_cfg["wrist_cam"]
    if third_override is not None and third_override != third:
        third, wrist = third_override, (third if third_override == wrist else wrist)
    return third, wrist


def open_cameras(arm_cfg, third_override=None, warm_up_s=1.5, threaded=False):
    from cameras import CameraManager

    third, wrist = camera_indices(arm_cfg, third_override)
    cams = CameraManager(wrist_id=wrist, third_id=third)
    end = time.time() + warm_up_s  # auto-exposure settles (the RealSense's first ~1 s is almost white)
    while time.time() < end:
        cams.capture()
    return ThreadedCameras(cams) if threaded else cams


def read_third(cams):
    return cams.capture()[1]


def read_wrist(cams):
    return cams.capture()[0]


def open_arm(arm_cfg, torque):
    """torque: "untouched" (read joints only; the servos are not configured, torque stays as it is),
    "off" (read joints, torque off so the arm can be moved by hand) or "on" (LeRobot connect(): configures
    the servos and turns torque on, so the arm holds where it is)."""
    from robot import SO101Robot

    port = arm_cfg["port"]
    if not os.path.exists(port):
        raise SystemExit(f"arm {arm_cfg['robot_id']} not found at {port} (plugged in? ls /dev/tty.usbmodem*; "
                         f"then fix the port in local/vision_pick/settings.json)")
    robot = SO101Robot(port, arm_cfg["robot_id"])
    if torque == "on":
        robot.connect()
    else:
        robot.robot.bus.connect()
        if torque == "off":
            robot.robot.bus.disable_torque()
    return robot


def close_arm(robot, disable_torque):
    try:
        robot.robot.bus.disconnect(disable_torque)
    except Exception as e:  # e.g. one servo not answering the torque-off write: don't bury the real result
        print(f"warning: disconnecting the arm: {e}")
