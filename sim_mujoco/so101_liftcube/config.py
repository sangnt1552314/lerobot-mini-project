"""All tunables of the MuJoCo SO-101 LiftCube task in one place (the counterpart of
sim/configs/lift_cube_dualcam.yaml for the Isaac version).

Distances in metres, angles in radians unless the name says _deg. World frame: robot base at the
origin on the table surface (z = 0), +x points away from the robot (towards the front camera).
"""

from dataclasses import dataclass, field

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "Lift the red cube."
ROBOT_TYPE = "so101_follower"


@dataclass
class CameraConfig:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy_deg: float = 55.0


@dataclass
class LiftCubeConfig:
    # timing: 30 Hz control = one dataset frame per env step; 6 physics substeps of 1/180 s
    fps: int = 30
    substeps: int = 6
    episode_length_s: float = 12.0          # eval time-out; demos end earlier (expert finishes)

    # images (same size as the Isaac dataset)
    width: int = 640
    height: int = 480
    front: CameraConfig = field(default_factory=lambda: CameraConfig(pos=(0.60, 0.0, 0.42), lookat=(0.17, 0.0, 0.02)))
    camera_jitter_pos: float = 0.01         # +- uniform per episode, front camera only
    camera_jitter_deg: float = 1.5

    # cube (red, 3 cm) and its spawn region in front of the robot
    cube_half: float = 0.015
    cube_mass: float = 0.03
    cube_friction: float = 1.0
    cube_x: tuple[float, float] = (0.17, 0.25)
    cube_y: tuple[float, float] = (-0.07, 0.07)
    cube_yaw_deg: float = 30.0              # +- uniform

    # success: cube centre this high above the table (Isaac task: 0.20 m above the robot base)
    success_height: float = 0.10
    success_hold_steps: int = 5             # consecutive steps above the height

    # robot start pose (rad) and per-joint +- uniform noise
    home_qpos: tuple[float, ...] = (0.0, -1.6, 0.55, 1.55, 1.5708, 0.0)  # fingertips ~(0.15, 0, 0.18)
    home_noise: float = 0.05

    # scripted expert (see expert.py). Phase durations in seconds.
    hover_height: float = 0.07              # fingertips above the cube centre before descending
    grasp_depth: float = 0.007              # fingertips this far below the cube centre when grasping
    cube_in_site_z: float = -0.004          # cube centre on the jaw axis in the gripper site frame (fixed jaw at -0.02)
    lift_height: float = 0.13               # fingertip height at the end of the lift (cube centre ~0.14)
    gripper_open: float = 0.9               # rad (~8 cm tip gap)
    gripper_closed: float = -0.15           # rad (target; the cube stops the jaw earlier)
    t_approach: float = 1.6
    t_descend: float = 0.9
    t_close: float = 0.6
    t_lift: float = 1.4
    t_hold: float = 0.6
    timing_noise: float = 0.15              # +- fraction, per phase, per episode
