"""Stage 5: minimal local safety layer, checked before any action is executed.

Two separate mechanisms, deliberately kept apart:

  is_safe_action() is the hard gate - stale responses, malformed chunks and
  targets outside the arm's physical range are refused outright.

  clamp_action() is the rate limiter - MolmoAct2 returns absolute joint
  targets, and the first target in a chunk is normally well away from where
  the arm currently is. Rather than refusing to move at all, each joint is
  moved at most MAX_STEP toward its target and converges over several steps.
"""

import math

from action_format import JOINT_ORDER, validate_action

# Max allowed change for any single joint in one step (degrees for arm joints,
# percent for gripper). Actions are scaled down to this, not rejected by it.
# Matches --max-step-deg in the reference implementation, which uses one cap
# for every joint so the scaling stays uniform.
MAX_STEP_DEG = 15.0

# Absolute sanity bounds on a target, arm frame. These are the model's own
# action q01..q99 (converted from norm_stats.json) padded by ~50 deg - wide
# enough that every normal prediction passes, tight enough to catch a
# frame-conversion blow-up. A target outside this is refused outright.
SANITY_RANGE = {
    "shoulder_pan": (-90.0, 100.0),
    "shoulder_lift": (-145.0, 95.0),
    "elbow_flex": (-105.0, 135.0),
    "wrist_flex": (-45.0, 145.0),
    "wrist_roll": (-115.0, 95.0),
    "gripper": (-35.0, 125.0),
}

# What this particular arm can physically reach, arm frame. The model is
# trained on arms whose gripper closes to ~0 (its action q01 is -0.3), but this
# follower stalls below ~15 and trips a servo overload that drops motor 6 off
# the bus. A target here is legitimate, just unreachable, so it is clipped
# rather than refused. Recalibrating the gripper's closed stop to 0 would let
# this entry be removed. Joints absent from this dict are left alone.
ARM_LIMITS = {
    "gripper": (15.0, 100.0),
}


def clip_to_arm(action):
    """Clips an otherwise-valid target into what the arm can physically do.

    Returns (clipped_action, notes) - notes is empty when nothing was clipped."""
    clipped = []
    notes = []

    for name, target in zip(JOINT_ORDER, action):
        low, high = ARM_LIMITS.get(name, (float("-inf"), float("inf")))
        bounded = min(max(target, low), high)
        if bounded != target:
            notes.append(f"{name} {target:.1f}->{bounded:.1f} (arm limit)")
        clipped.append(bounded)

    return clipped, notes


def is_safe_action(current_state, action, *, expected_observation_id=None, response_observation_id=None):
    """current_state: dict from robot.get_state(). action: list of floats in
    JOINT_ORDER, arm frame (as returned by server/policy.py). Returns
    (is_safe, reason).

    A target further than MAX_STEP_DEG away is NOT unsafe - clamp_action()
    handles that, and one the arm simply can't reach is clip_to_arm()'s job.
    This only refuses what neither of those can make safe."""

    if expected_observation_id is not None and response_observation_id is not None:
        if expected_observation_id != response_observation_id:
            return False, (
                f"stale response: expected observation_id={expected_observation_id}, "
                f"got {response_observation_id}"
            )

    valid, reason = validate_action(action)
    if not valid:
        return False, reason

    violations = []
    for name, target in zip(JOINT_ORDER, action):
        low, high = SANITY_RANGE[name]
        if not low <= target <= high:
            violations.append(f"{name} target {target:.1f} outside [{low}, {high}]")

    if violations:
        # Report every offending joint: one joint out of range is a bad
        # prediction, all six point at the arm<->model frame conversion.
        return False, "; ".join(violations)

    return True, "ok"


def clamp_action(current_state, action):
    """Rate-limits an absolute joint target so no joint moves more than
    MAX_STEP_DEG in one step. Call only on an action is_safe_action() accepted.

    The whole delta vector is scaled by one factor rather than each joint being
    clamped separately: per-joint clamping shortens only the fastest joints and
    so bends the motion away from the direction the model asked for. This
    mirrors clip_action() in the reference implementation.

    Returns (clamped_action, notes) - notes is empty when nothing was scaled."""
    deltas = [target - current_state[name] for name, target in zip(JOINT_ORDER, action)]
    biggest = max(abs(d) for d in deltas)

    if biggest <= MAX_STEP_DEG or biggest == 0.0:
        return list(action), []

    scale = MAX_STEP_DEG / biggest
    clamped = [current_state[name] + d * scale for name, d in zip(JOINT_ORDER, deltas)]
    notes = [f"scaled {scale:.2f}x (largest delta {biggest:.1f} > {MAX_STEP_DEG})"]

    return clamped, notes
