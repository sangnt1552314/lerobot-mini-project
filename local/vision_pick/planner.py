"""Grasp-and-lift plan for a cube at a table (x, y): joint-space waypoints from IK, all checked before anything moves.

Pick:  above (approach height) -> open -> straight down in step_m steps -> grasp -> close -> straight up -> lift.
Place: back down -> release -> above -> the pose the arm started from.
Fixed top-down grasp: fingers vertical, wrist_roll = the median roll of the calibration touches.
The gripper only opens once the arm is above the cube, so it never swings open next to the table.
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from safety import ARM_LIMITS, is_safe_action

from .calibration import inside_workspace, pixel_to_table
from .kinematics import ik, in_joint_range, joints_to_rad, rad_to_joints, tip


@dataclass
class Waypoint:
    name: str
    joints: list  # 6 arm-frame values: degrees, gripper %
    tip: list  # fingertip x, y, z (m)
    tilt: float  # fingers off vertical (deg)
    slow: bool = False


@dataclass
class Plan:
    pixel: tuple
    table_xy: tuple
    grasp_z: float
    open_pct: float
    close_pct: float
    pick: list = field(default_factory=list)
    place: list = field(default_factory=list)
    problems: list = field(default_factory=list)  # any problem: control mode refuses to move
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problems

    def to_dict(self):
        return asdict(self)


def gripper_bounds(gripper_floor):
    """The gripper's allowed range: local/safety.py ARM_LIMITS, raised to the arm's floor."""
    lo, hi = ARM_LIMITS.get("gripper", (-math.inf, math.inf))
    return max(gripper_floor, lo), hi


def gripper_targets(cal, settings, gripper_floor):
    """(open, close) gripper targets: close = squeeze_pct past the calibrated contact with the cube."""
    lo, hi = gripper_bounds(gripper_floor)
    open_pct = min(max(settings["grasp"]["open_pct"], lo), hi)
    close_pct = min(max(cal["gripper_contact"] - settings["grasp"]["squeeze_pct"], lo), hi)
    return float(open_pct), float(close_pct)


def safety_check(joints, cal):
    """local/safety.py is_safe_action, with wrist_roll taken relative to the calibrated grasp roll (its
    SANITY_RANGE was written for a wrist reading ~0 in the grasp orientation)."""
    relative = list(joints)
    relative[4] -= cal["wrist_roll_grasp"]
    return is_safe_action(None, relative)


def _heights(top, bottom, step):
    n = max(1, math.ceil((top - bottom) / step - 1e-9))
    return list(np.linspace(top, bottom, n + 1))


def plan_pick(pixel, cal, settings, current, gripper_floor):
    """pixel: cube centre (u, v). current: the arm's 6 arm-frame joints now. Returns a Plan."""
    g, c = settings["grasp"], settings["checks"]
    x, y = pixel_to_table(cal, *pixel)
    x, y = x + g["grasp_offset_x_m"], y + g["grasp_offset_y_m"]
    z_grasp = cal["grasp_z"] + g["grasp_offset_z_m"]
    z_above, z_lift = z_grasp + g["approach_height_m"], z_grasp + g["lift_height_m"]
    open_pct, close_pct = gripper_targets(cal, settings, gripper_floor)
    plan = Plan(pixel=tuple(round(p, 1) for p in pixel), table_xy=(round(x, 4), round(y, 4)), grasp_z=round(z_grasp, 4),
                open_pct=open_pct, close_pct=close_pct)

    if not inside_workspace(cal, x, y, g["workspace_margin_m"]):
        plan.warnings.append(f"cube at ({x * 100:.1f}, {y * 100:.1f}) cm is outside the calibrated area "
                             f"(+{g['workspace_margin_m'] * 100:.0f} cm): the position is extrapolated")
    if close_pct >= open_pct - 5:
        plan.problems.append(f"gripper close target {close_pct:.0f} is not below open {open_pct:.0f}")
    if cal["gripper_contact"] - close_pct < 3:
        plan.warnings.append(f"the jaws meet the cube at {cal['gripper_contact']:.0f} but may only close to "
                             f"{close_pct:.0f} (gripper floor): almost no squeeze, the cube may slip")
    if z_grasp - cal["table_z"] < 0.005:
        plan.problems.append(f"grasp height {z_grasp * 100:.1f} cm is within 5 mm of the table: check grasp_offset_z_m")
    safe, reason = safety_check(current, cal)
    if not safe:
        plan.problems.append(f"the arm's current pose fails the local safety check: {reason}")

    roll = math.radians(cal["wrist_roll_grasp"])
    taught = np.array(cal["table_xy"])
    nearest = int(np.argmin(np.hypot(taught[:, 0] - x, taught[:, 1] - y)))
    seeds = [joints_to_rad(cal["joints"][nearest])[:4], joints_to_rad(current)[:4], [math.atan2(y, x), 0.0, 0.5, 1.2]]

    def solve(name, z, gripper, slow, tilt_tol):
        nonlocal seeds
        q, pos_err, _ = ik(np.array([x, y, z]), seeds, roll, down_weight=0.1 if tilt_tol < 15 else 0.01)
        seeds = [q[:4]] + seeds
        joints = rad_to_joints(q, gripper)
        p, tilt = tip(joints)
        if pos_err > c["ik_pos_tol_m"]:
            plan.problems.append(f"{name}: unreachable at z {z * 100:.1f} cm (IK misses by {pos_err * 1000:.0f} mm)")
        if tilt > tilt_tol:
            plan.problems.append(f"{name}: fingers {tilt:.0f} deg off vertical (limit {tilt_tol})")
        if not in_joint_range(joints):
            plan.problems.append(f"{name}: outside the joint ranges")
        safe, reason = safety_check(joints, cal)
        if not safe:
            plan.problems.append(f"{name}: local safety check: {reason}")
        return Waypoint(name, [round(v, 2) for v in joints], [round(float(v), 4) for v in p], round(tilt, 1), slow)

    lo, hi = gripper_bounds(gripper_floor)
    grip_now = min(max(current[5], lo), hi)  # unchanged on the way over, but inside the gripper limits
    above = solve("above", z_above, grip_now, False, c["travel_tilt_tol_deg"])
    plan.pick.append(above)
    plan.pick.append(Waypoint("open", above.joints[:5] + [open_pct], above.tip, above.tilt, True))
    descend = [solve(f"descend{i + 1}", z, open_pct, True, c["grasp_tilt_tol_deg"])
               for i, z in enumerate(_heights(z_above, z_grasp, g["step_m"])[1:-1])]
    plan.pick += descend
    grasp = solve("grasp", z_grasp, open_pct, True, c["grasp_tilt_tol_deg"])
    plan.pick.append(grasp)
    plan.pick.append(Waypoint("close", grasp.joints[:5] + [close_pct], grasp.tip, grasp.tilt, True))
    lift_heights = _heights(z_lift, z_grasp, g["step_m"])[::-1][1:]
    for i, z in enumerate(lift_heights):
        name = "lift" if i == len(lift_heights) - 1 else f"lift{i + 1}"
        plan.pick.append(solve(name, z, close_pct, True, c["travel_tilt_tol_deg"]))

    # place: the same column back down, release, back up it with the jaws open, then to where the arm started
    lifts = plan.pick[-len(lift_heights):-1]
    p_now, tilt_now = tip(current)
    plan.place = (
        [Waypoint(f"lower{i + 1}", w.joints, w.tip, w.tilt, True) for i, w in enumerate(reversed(lifts))]
        + [Waypoint("set_down", grasp.joints[:5] + [close_pct], grasp.tip, grasp.tilt, True),
           Waypoint("release", grasp.joints[:5] + [open_pct], grasp.tip, grasp.tilt, True)]
        + [Waypoint(f"retreat{i + 1}", w.joints, w.tip, w.tilt, True) for i, w in enumerate(reversed(descend))]
        + [Waypoint("retreat", above.joints[:5] + [open_pct], above.tip, above.tilt, True),
           Waypoint("start", [round(v, 2) for v in current[:5]] + [grip_now], [round(float(v), 4) for v in p_now],
                    round(tilt_now, 1))]
    )
    _check_path(plan, cal, settings, current)
    return plan


def _check_path(plan, cal, settings, current, samples=12):
    """Fingertip along the joint-space interpolation between consecutive waypoints: never lower than both ends
    and the clearance plane (cube top + min_table_clearance_m above the table), and the vertical column stays
    within 1 cm of the cube's x, y. Also catches IK branch flips (a joint jumping > 30 deg between neighbours)."""
    g = settings["grasp"]
    plane = cal["table_z"] + cal["cube_size_m"] + g["min_table_clearance_m"]
    x, y = plan.table_xy
    seq = [Waypoint("current", list(current), [], 0.0)] + plan.pick + plan.place
    column = ("open", "descend", "grasp", "close", "lift", "lower", "set_down", "release", "retreat")
    for a, b in zip(seq, seq[1:]):
        za, zb = tip(a.joints)[0][2], tip(b.joints)[0][2]
        floor = min(za, zb, plane) - 0.003
        in_column = a.name.startswith(column) and b.name.startswith(column)
        if in_column and max(abs(p - q) for p, q in zip(a.joints[:5], b.joints[:5])) > 30:
            plan.problems.append(f"{a.name} -> {b.name}: a joint jumps > 30 deg (IK changed branch)")
        for t in np.linspace(0, 1, samples)[1:-1]:
            p, _ = tip([pa + t * (pb - pa) for pa, pb in zip(a.joints, b.joints)])
            if p[2] < floor:
                plan.problems.append(f"{a.name} -> {b.name}: fingertips dip to {p[2] * 100:.1f} cm "
                                     f"(floor {floor * 100:.1f} cm)")
                break
            if in_column and math.hypot(p[0] - x, p[1] - y) > 0.01:
                plan.warnings.append(f"{a.name} -> {b.name}: path drifts {math.hypot(p[0] - x, p[1] - y) * 100:.1f} cm "
                                     f"sideways from the cube")
                break


def format_plan(plan):
    rows = [f"cube pixel ({plan.pixel[0]:.0f}, {plan.pixel[1]:.0f}) -> table ({plan.table_xy[0] * 100:.1f}, "
            f"{plan.table_xy[1] * 100:.1f}) cm, grasp height {plan.grasp_z * 100:.1f} cm; gripper open "
            f"{plan.open_pct:.0f} / close {plan.close_pct:.0f}",
            f"  {'waypoint':10s} {'pan':>7s} {'lift':>7s} {'elbow':>7s} {'w_flex':>7s} {'w_roll':>7s} {'grip':>6s}"
            f"   tip x/y/z (cm)     tilt"]
    for section, wps in (("pick", plan.pick), ("place", plan.place)):
        rows.append(f"  -- {section}")
        for w in wps:
            j = w.joints
            rows.append(f"  {w.name:10s} {j[0]:7.1f} {j[1]:7.1f} {j[2]:7.1f} {j[3]:7.1f} {j[4]:7.1f} {j[5]:6.1f}"
                        f"   {w.tip[0] * 100:5.1f} {w.tip[1] * 100:5.1f} {w.tip[2] * 100:5.1f}   {w.tilt:4.0f}")
    rows += [f"  WARNING: {w}" for w in plan.warnings]
    rows += [f"  PROBLEM: {p}" for p in plan.problems]
    rows.append("  plan OK" if plan.ok else "  NOT EXECUTABLE")
    return "\n".join(rows)
