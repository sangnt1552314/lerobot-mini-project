"""Warns when the arm is in a pose MolmoAct2 was never trained on.

MolmoAct2-SO100_101 normalizes state with norm_mode=q01_q99, so any joint
outside the q01..q99 band of its training data saturates at +-1 and the model
is effectively guessing. In practice it responds by predicting the current
pose back - a chunk that asks the arm to hold still, which looks exactly like
a broken client.

Numbers below are the model's own state_stats from
https://huggingface.co/allenai/MolmoAct2-SO100_101/blob/main/norm_stats.json
converted from the model frame to the SO101 arm frame with the same offsets
and signs as server/policy.py.
"""

from action_format import JOINT_ORDER

# (q01, q99) per joint, arm frame. Outside this the model is extrapolating.
TRAINING_RANGE = {
    "shoulder_pan": (-41.9, 48.3),
    "shoulder_lift": (-95.3, 46.3),
    "elbow_flex": (-51.6, 83.1),
    "wrist_flex": (5.7, 91.8),
    "wrist_roll": (-63.4, 42.9),
    "gripper": (0.9, 44.1),
}

# (q10, q90) per joint, arm frame - where the training data actually lives.
COMFORT_RANGE = {
    "shoulder_pan": (-24.9, 31.5),
    "shoulder_lift": (-90.9, 23.7),
    "elbow_flex": (-21.8, 78.6),
    "wrist_flex": (27.1, 81.6),
    "wrist_roll": (-39.5, 15.9),
    "gripper": (1.6, 31.9),
}

# Median training pose (q50), arm frame. A good place to start an episode.
READY_POSE = {
    "shoulder_pan": 3.1,
    "shoulder_lift": -33.2,
    "elbow_flex": 34.4,
    "wrist_flex": 57.9,
    "wrist_roll": -11.0,
    "gripper": 9.2,
}


def check_pose(state):
    """state: dict from robot.get_state(). Returns a list of warning strings,
    empty when every joint sits inside the training distribution."""
    warnings = []

    for name in JOINT_ORDER:
        value = state[name]
        low, high = TRAINING_RANGE[name]
        if value < low:
            warnings.append(f"{name}={value:.1f} is {low - value:.1f} below q01 ({low})")
        elif value > high:
            warnings.append(f"{name}={value:.1f} is {value - high:.1f} above q99 ({high})")

    return warnings


def report(state):
    """Prints the pose check and returns True when the pose looks usable."""
    warnings = check_pose(state)

    if not warnings:
        print("pose check: OK - inside the model's training distribution")
        return True

    print("pose check: OUT OF DISTRIBUTION -", "; ".join(warnings))
    print("  MolmoAct2 tends to predict 'hold still' from poses like this.")
    print(f"  Suggested start pose: { {k: round(v, 1) for k, v in READY_POSE.items()} }")
    return False
