"""Stage 5: minimal local safety layer, checked before any action would ever
be executed. Does NOT execute anything itself - that's Stage 6.
"""

from action_format import JOINT_ORDER, validate_action

# Max allowed change per single step (degrees for arm joints, percent for
# gripper) between the robot's current state and a predicted target. Tune
# these down before ever enabling real execution.
MAX_STEP = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 15.0,
    "elbow_flex": 15.0,
    "wrist_flex": 10.0,
    "wrist_roll": 10.0,
    "gripper": 20.0,
}


def is_safe_action(current_state, action, *, expected_observation_id=None, response_observation_id=None):
    """current_state: dict from robot.get_state(). action: list of floats in
    JOINT_ORDER, arm frame (as returned by server/policy.py). Returns
    (is_safe, reason)."""

    if expected_observation_id is not None and response_observation_id is not None:
        if expected_observation_id != response_observation_id:
            return False, (
                f"stale response: expected observation_id={expected_observation_id}, "
                f"got {response_observation_id}"
            )

    valid, reason = validate_action(action)
    if not valid:
        return False, reason

    for name, target in zip(JOINT_ORDER, action):
        current = current_state[name]
        delta = abs(target - current)
        limit = MAX_STEP[name]
        if delta > limit:
            return False, (
                f"{name} delta {delta:.1f} exceeds max step {limit} "
                f"(current={current:.1f}, target={target:.1f})"
            )

    return True, "ok"
