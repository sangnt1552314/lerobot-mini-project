"""Thin wrapper around the real SO101 follower arm (read-only in this stage).

Mirrors the connect/get_observation/disconnect pattern used in
scripts/move_follower.py. Does NOT send any actions to the motors.
"""

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


class SO101Robot:
    def __init__(self, port, robot_id):
        config = SOFollowerRobotConfig(port=port, id=robot_id)
        self.robot = SOFollower(config)

    def connect(self):
        self.robot.connect(calibrate=False)

    def get_state(self):
        """Returns the current joint positions as a JSON-serializable dict."""
        obs = self.robot.get_observation()
        return {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}

    def disconnect(self):
        self.robot.disconnect()
