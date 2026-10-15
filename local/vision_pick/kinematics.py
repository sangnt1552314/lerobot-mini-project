"""SO-101 kinematics, reused from local/cube_fallback.py: body offsets copied from MuJoCo Menagerie
robotstudio_so101 @ 0059d433 (tests/test_kinematics.py checks FK against that MJCF in MuJoCo). The fingertip
point is the model's "gripperframe" site, between the fingertips.

Joint convention (as local/run_act_policy.py): MJCF radians = radians(LeRobot degrees), use_degrees=True.
The base frame's z = 0 is the bottom of the base (lowest base-mesh vertex at -2.4 mm), i.e. the table.
"""

import math

import numpy as np

from cube_fallback import DOWN, RANGE_RAD, fk, ik  # noqa: F401  (re-exported)


def joints_to_rad(joints):
    """6 arm-frame values (deg..., gripper %) or 5 -> 5 MJCF radians."""
    return np.radians(np.asarray(joints[:5], dtype=float))


def rad_to_joints(q, gripper):
    return [math.degrees(float(v)) for v in q[:5]] + [float(gripper)]


def tilt_deg(rot):
    """Angle between the fingers (site x axis) and straight down."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[:, 0] @ DOWN)))))


def tip(joints):
    """Fingertip position (3,) in metres and the fingers' tilt from vertical (deg) for arm-frame joints."""
    p, r = fk(joints_to_rad(joints))
    return p, tilt_deg(r)


def in_joint_range(joints):
    q = joints_to_rad(joints)
    return all(lo - 1e-6 <= v <= hi + 1e-6 for v, (lo, hi) in zip(q, RANGE_RAD))
