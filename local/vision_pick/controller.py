"""Slow, checked execution of a Plan on anything with SO101Robot's interface (get_state() -> {joint: value},
send_action({joint: value})): the real arm via local/robot.py, or a mock in the tests.

Every command is ramped from the previous one at speed_deg_s (slow_speed_deg_s for slow waypoints), refused if
local/safety.py is_safe_action rejects it, clipped by clip_to_arm (ARM_LIMITS) and the arm's gripper floor,
and rate-limited by clamp_action as a backstop. If an arm joint lags its command by more than track_limit_deg
for 0.5 s (blocked, or hitting something) the arm is told to hold where it is and Abort is raised. Ctrl+C
(KeyboardInterrupt) is left to the caller, which holds the arm the same way.
"""

import re
import time

from action_format import JOINT_ORDER
from safety import clamp_action, clip_to_arm

from .planner import gripper_bounds, safety_check

ARM = range(5)  # joint indices without the gripper


class Abort(RuntimeError):
    pass


def is_via(name):
    """Numbered in-between points of the vertical column (descend1, lift2, ...): passed through without stopping,
    so the arm moves down and up in one continuous line."""
    return re.fullmatch(r"(descend|lift|lower|retreat)\d+", name) is not None


class Executor:
    def __init__(self, robot, settings, cal, gripper_floor, clock=time.monotonic, sleep=time.sleep, log=print):
        self.robot, self.cal, self.log = robot, cal, log
        self.m = settings["motion"]
        self.grip_lo, self.grip_hi = gripper_bounds(gripper_floor)
        self.clock, self.sleep = clock, sleep
        self.cmd = self.measured()
        self.sent = []  # every command sent, for the tests and the run log
        self.on_step = None  # on_step(measured, sent) after every command, e.g. to record a dataset frame

    def measured(self):
        state = self.robot.get_state()
        return [float(state[n]) for n in JOINT_ORDER]

    def hold(self):
        """Command the measured position: the servos stop pushing and hold there."""
        here = self.measured()
        here[5] = min(max(here[5], self.grip_lo), self.grip_hi)
        self.robot.send_action(dict(zip(JOINT_ORDER, here)))
        self.sent.append(here)
        self.cmd = here

    def _send(self, cmd, measured):
        safe, reason = safety_check(cmd, self.cal)
        if not safe:
            self.hold()
            raise Abort(f"refused by the local safety check: {reason}")
        out, _ = clip_to_arm(cmd)
        out[5] = min(max(out[5], self.grip_lo), self.grip_hi)
        # clamp_action (MAX_STEP_DEG per step) as a backstop on the arm joints; the gripper may legitimately sit
        # far from its target while squeezing the cube, so it is left out of that scaling
        current = dict(zip(JOINT_ORDER, measured))
        current["gripper"] = out[5]
        out, _ = clamp_action(current, out)
        self.robot.send_action(dict(zip(JOINT_ORDER, out)))
        self.sent.append(out)
        if self.on_step:
            self.on_step(measured, out)

    def move_to(self, target, slow=False, label="", settle=True):
        """Ramp the command to target (6 arm-frame values) and wait for the arm to get there (settle=False: return as
        soon as the command is there, for points the arm only passes through)."""
        safe, reason = safety_check(target, self.cal)  # (IK targets were also checked against the joint ranges)
        if not safe:
            raise Abort(f"{label}: target refused before moving: {reason}")
        dt = 1.0 / self.m["fps"]
        arm_step = (self.m["slow_speed_deg_s"] if slow else self.m["speed_deg_s"]) * dt
        grip_step = self.m["gripper_speed_pct_s"] * dt
        gripper_moves = abs(target[5] - self.cmd[5]) > 1e-6
        lag_since = arrived = None
        while True:
            t0 = self.clock()
            measured = self.measured()
            delta = [target[i] - self.cmd[i] for i in ARM]
            biggest = max(abs(d) for d in delta)
            scale = min(1.0, arm_step / biggest) if biggest > 0 else 1.0
            cmd = [self.cmd[i] + delta[i] * scale for i in ARM]
            g = target[5] - self.cmd[5]
            cmd.append(self.cmd[5] + max(-grip_step, min(grip_step, g)))
            self._send(cmd, measured)
            self.cmd = cmd

            lag = max(abs(cmd[i] - measured[i]) for i in ARM)
            if lag > self.m["track_limit_deg"]:
                lag_since = lag_since if lag_since is not None else t0
                if t0 - lag_since > 0.5:
                    self.hold()
                    worst = max(ARM, key=lambda i: abs(cmd[i] - measured[i]))
                    raise Abort(f"{label}: {JOINT_ORDER[worst]} is {abs(cmd[worst] - measured[worst]):.0f} deg behind "
                                f"its command (blocked?)")
            else:
                lag_since = None

            if all(abs(cmd[i] - target[i]) < 1e-9 for i in range(6)):
                if not settle:
                    return measured
                arrived = arrived if arrived is not None else t0
                waited = t0 - arrived
                close = all(abs(measured[i] - target[i]) <= self.m["arrive_tol_deg"] for i in ARM)
                # a gripper move gets 0.4 s to finish (or to stall on the cube); the arm gets up to settle_s to sag in
                if (close and (not gripper_moves or waited >= 0.4)) or waited >= self.m["settle_s"]:
                    if not close:
                        err = max(abs(measured[i] - target[i]) for i in ARM)
                        self.log(f"  {label}: arm settled {err:.1f} deg from target (gravity sag)")
                    return measured
            self.sleep(max(0.0, dt - (self.clock() - t0)))

    def dwell(self, seconds):
        """Keep commanding the current target for a while (each step goes through _send, so it is recorded)."""
        dt = 1.0 / self.m["fps"]
        for _ in range(int(round(seconds * self.m["fps"]))):
            t0 = self.clock()
            self._send(list(self.cmd), self.measured())
            self.sleep(max(0.0, dt - (self.clock() - t0)))

    def run(self, waypoints, pause_after=None, pause=None, on_waypoint=None):
        """Move through waypoints in order. pause(name) is called after the waypoint named pause_after (e.g. to
        ask for confirmation); on_waypoint(name, measured) after each one."""
        for wp in waypoints:
            self.log(f"  -> {wp.name}")
            measured = self.move_to(wp.joints, wp.slow, wp.name, settle=not is_via(wp.name))
            if on_waypoint:
                on_waypoint(wp.name, measured)
            if pause and wp.name == pause_after:
                pause(wp.name)

    def holding_cube(self, close_pct, contact_pct):
        """After closing: (holding, gripper reading). The jaws stop on the cube near the calibrated contact reading;
        closing past the midpoint toward the close target means nothing is between them. None when the two are
        under 6 apart (gripper floor right at the cube's width): the reading can't tell."""
        grip = self.measured()[5]
        if contact_pct - close_pct < 6:
            return None, grip
        return grip > (contact_pct + close_pct) / 2, grip
