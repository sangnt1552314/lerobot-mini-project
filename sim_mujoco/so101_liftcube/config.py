"""All tunables of the MuJoCo SO-101 LiftCube task in one place (the counterpart of
sim/configs/lift_cube_dualcam.yaml for the Isaac version).

Distances in metres, angles in radians unless the name says _deg. World frame: robot base at the
origin on the table surface (z = 0), +x points away from the robot (towards the front camera).
"""

from dataclasses import dataclass, field, replace

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "Lift the red cube."
ROBOT_TYPE = "so101_follower"


@dataclass
class CameraConfig:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy_deg: float = 55.0


@dataclass
class Appearance:
    """How the scene looks and how much that is randomized per episode. Defaults = the original sim
    (yellow robot, checker table, fixed lights), so existing datasets stay reproducible."""

    robot_rgba: tuple[float, float, float, float] | None = None   # repaint the yellow printed parts
    table_texture: str | None = None        # PNG under sim_mujoco/assets/; None = checker
    floor_texture: str | None = None        # what the cameras see beyond the table edge
    table_center: tuple[float, float] = (0.0, 0.0)
    table_half: tuple[float, float] = (1.0, 1.0)
    texture_repeat: tuple[float, float] = (8.0, 8.0)
    headlight_ambient: float | None = None  # None = 0.35; higher = brighter, softer shadows
    key_light_diffuse: float | None = None  # None = 0.6
    # per-episode randomization (0 = off)
    tint_jitter: float = 0.0                 # +- multiplicative per RGB channel: table, floor, robot
    light_jitter: float = 0.0                # +- fraction of light intensity
    light_dir_jitter: float = 0.0            # +- per component of each light's direction
    front_fovy_jitter_deg: float = 0.0
    wrist_fovy_range: tuple[float, float] | None = None   # None = model's 61 deg


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

    appearance: Appearance = field(default_factory=Appearance)


# Gripper 15 on the real arm (LeRobot 0..100) -> MJCF rad. The real follower stalls below ~15
# (local/safety.py ARM_LIMITS), so the "real" expert never asks for less.
_GRIPPER_LO, _GRIPPER_HI = -0.174533, 1.7453292
REAL_GRIPPER_FLOOR_RAD = _GRIPPER_LO + 0.15 * (_GRIPPER_HI - _GRIPPER_LO)


def make_config(profile: str = "default") -> LiftCubeConfig:
    """'default': the original sim. 'real': matched to the physical setup (2026-10-12):
    start pose read from the real arm, front camera fitted to the real photo, the robot painted
    white, table/floor textured from the real front camera, and randomized lighting, colours and
    camera pose to cover what was not matched exactly."""
    cfg = LiftCubeConfig()
    if profile == "default":
        return cfg
    if profile != "real":
        raise ValueError(f"unknown profile {profile!r}; use 'default' or 'real'")
    return replace(
        cfg,
        # real arm, hand-placed start pose, read with local/run_act_policy.py's mapping:
        # arm frame deg [7.9, 6.5, 11.3, 87.7, 93.0], gripper 79 (open)
        home_qpos=(0.1381, 0.1135, 0.1979, 1.5305, 1.6226, 1.3421),
        home_noise=0.08,
        # rough fit of the real front webcam (sim render overlaid on the real photo). The real robot
        # looks larger, so the camera is probably closer: replace with a tape-measured pose when
        # available; the jitter below covers the uncertainty meanwhile.
        front=CameraConfig(pos=(0.36, -0.02, 0.50), lookat=(0.08, 0.02, 0.05), fovy_deg=55.0),
        camera_jitter_pos=0.06,
        camera_jitter_deg=5.0,
        # the real cube sits further out than the original spawn region
        cube_x=(0.17, 0.30),
        cube_y=(-0.10, 0.10),
        gripper_closed=REAL_GRIPPER_FLOOR_RAD,
        appearance=Appearance(
            robot_rgba=(0.93, 0.93, 0.91, 1.0),
            table_texture="real/table_wood.png",
            floor_texture="real/curtain.png",
            table_center=(0.15, 0.0),
            table_half=(0.40, 0.45),
            texture_repeat=(3.0, 3.0),
            tint_jitter=0.15,
            light_jitter=0.4,
            light_dir_jitter=0.3,
            front_fovy_jitter_deg=8.0,
            wrist_fovy_range=(65.0, 90.0),   # real wrist lens is wide-angle
            headlight_ambient=0.55,
            key_light_diffuse=0.35,
        ),
    )
