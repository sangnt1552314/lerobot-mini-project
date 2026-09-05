"""Thin wrapper around the real SO101 follower arm.

Mirrors the connect/get_observation/send_action/disconnect pattern used in
scripts/move_follower.py. send_action() is only ever called from Stage 6's
single-action control-mode path in client.py - never from preview mode.
"""

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


class SO101Robot:
    def __init__(self, port, robot_id):
        # use_degrees=True: MolmoAct2-SO100_101 expects the old (v2.1) degree
        # convention, not LeRobot's newer normalized -100..100 range. See
        # server/policy.py for the matching arm<->model frame conversion.
        config = SOFollowerRobotConfig(port=port, id=robot_id, use_degrees=True)
        self.robot = SOFollower(config)

    def connect(self):
        self.robot.connect(calibrate=False)

    def get_state(self):
        """Returns the current joint positions as a JSON-serializable dict."""
        obs = self.robot.get_observation()
        return {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}

    def send_action(self, action):
        """action: {joint_name: value} dict, same keys as get_state()."""
        self.robot.send_action({f"{k}.pos": v for k, v in action.items()})

    def disconnect(self):
        self.robot.disconnect()
