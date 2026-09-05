"""Stage 3: sanity-check MolmoAct2's action chunk before ever trusting it
enough to move the SO101. server/policy.py already converts the model's raw
output back to SO101 arm-frame degrees, so `action` here should be directly
comparable to robot_state - this is only a structural safety net, not a
frame/unit conversion.
"""

import math

# Same order as local/robot.py's SO101Robot.get_state() dict keys.
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def validate_action(action):
    """Structural check only: right length, all values finite numbers.
    Returns (is_valid, reason)."""
    if not isinstance(action, (list, tuple)) or len(action) != len(JOINT_ORDER):
        return False, f"expected {len(JOINT_ORDER)} values, got {action!r}"

    for v in action:
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            return False, f"non-finite value in action: {action!r}"

    return True, "ok"
