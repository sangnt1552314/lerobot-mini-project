"""All tunables of the MuJoCo SO-101 Lift task in one place (the counterpart of
sim/configs/lift_cube_dualcam.yaml for the Isaac version).

A config is a physical setup (profile: start pose, cameras, table, lighting) combined with an object
to lift: make_config(profile, obj). Each combination is its own dataset and its own policy.

Distances in metres, angles in radians unless the name says _deg. World frame: robot base at the
origin on the table surface (z = 0), +x points away from the robot (towards the front camera),
+y to the robot's left.
"""

from dataclasses import dataclass, field, replace

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ROBOT_TYPE = "so101_follower"


@dataclass
class CameraConfig:
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy_deg: float = 55.0
    roll_deg: float = 0.0                   # about the viewing axis (a webcam on a clip is rarely level)


@dataclass
class ObjectConfig:
    """The object to lift, spawned upright on the table at a random x, y and yaw."""

    name: str = "cube"
    shape: str = "box"                      # "box", or "cylinder" (upright; any jaw direction grips it)
    half_width: float = 0.015               # box: half side; cylinder: radius
    half_height: float = 0.015
    base_half_width: float = 0.0            # cylinder: radius of a flange at its foot (0 = none)
    base_half_height: float = 0.0
    mass: float = 0.03
    friction: float = 1.0
    rgba: tuple[float, float, float, float] = (0.85, 0.08, 0.08, 1.0)
    # grasp geometry for the scripted expert
    grasp_depth: float = 0.007              # fingertips this far below the object centre when grasping
    in_site_z: float = -0.004               # object centre on the jaw axis in the gripper site frame (fixed jaw at -0.02)
    task: str = "Lift the red cube."        # dataset task string


@dataclass
class Wall:
    """Visual-only vertical wall: the plane x = offset (axis "x") or y = offset (axis "y"), facing the robot."""

    axis: str
    offset: float
    rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)


@dataclass
class Appearance:
    """How the scene looks and how much that is randomized per episode. Defaults = the original sim
    (yellow robot, checker table, fixed lights), so existing datasets stay reproducible."""

    robot_rgba: tuple[float, float, float, float] | None = None   # repaint the yellow printed parts
    table_texture: str | None = None        # PNG under sim_mujoco/assets/; None = checker
    floor_texture: str | None = None        # what the cameras see beyond the table edge
    floor_rgba: tuple[float, float, float, float] | None = None   # plain floor instead (no floor_texture)
    floor_z: float = -0.03                  # visual only; a desk's real floor is ~0.75 m down
    walls: tuple[Wall, ...] = ()
    table_center: tuple[float, float] = (0.0, 0.0)
    table_half: tuple[float, float] = (1.0, 1.0)
    texture_repeat: tuple[float, float] = (8.0, 8.0)
    headlight_ambient: float | None = None  # None = 0.35; higher = brighter, softer shadows
    key_light_diffuse: float | None = None  # None = 0.6
    key_light_dir: tuple[float, float, float] | None = None   # None = (-0.3, 0.3, -1.0); keep walls out of its path
    # colour of each real camera on the rendered image: saturation (1 = unchanged, >1 more saturated), then
    # RGB gains (the colour cast), e.g. {"front": 1.7} and {"front": (0.8, 1.0, 0.6)}
    camera_gain: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    camera_saturation: dict[str, float] = field(default_factory=dict)
    object_rgba: tuple[float, float, float, float] | None = None   # the object's colour under this setup's light
    # per-episode randomization (0 = off)
    tint_jitter: float = 0.0                 # +- multiplicative per RGB channel: table, floor, walls, robot
    light_jitter: float = 0.0                # +- fraction of light intensity
    light_dir_jitter: float = 0.0            # +- per component of each light's direction
    front_fovy_jitter_deg: float = 0.0
    wrist_fovy_range: tuple[float, float] | None = None   # None = model's 61 deg
    camera_gain_jitter: float = 0.0          # +- per RGB channel of each camera's gain (white balance)


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

    # the object and its spawn region in front of the robot
    obj: ObjectConfig = field(default_factory=ObjectConfig)
    object_x: tuple[float, float] = (0.17, 0.25)
    object_y: tuple[float, float] = (-0.07, 0.07)
    object_yaw_deg: float = 30.0            # +- uniform

    # success: object centre this high above the table (Isaac task: 0.20 m above the robot base)
    success_height: float = 0.10
    success_hold_steps: int = 5             # consecutive steps above the height

    # robot start pose (rad) and per-joint +- uniform noise
    home_qpos: tuple[float, ...] = (0.0, -1.6, 0.55, 1.55, 1.5708, 0.0)  # fingertips ~(0.15, 0, 0.18)
    home_noise: float = 0.05

    # scripted expert (see expert.py). Phase durations in seconds.
    hover_height: float = 0.07              # fingertips above the object centre before descending
    lift_height: float = 0.13               # fingertip height at the end of the lift (object centre ~0.14)
    gripper_open: float = 0.9               # rad (~8 cm tip gap)
    gripper_closed: float = -0.15           # rad (target; the object stops the jaw earlier)
    t_approach: float = 1.6
    t_descend: float = 0.9
    t_close: float = 0.6
    t_lift: float = 1.4
    t_hold: float = 0.6
    timing_noise: float = 0.15              # +- fraction, per phase, per episode
    # off in the default profile (keeps its trajectories identical); see make_config("real")
    approach_clearance: float = 0.0         # approach to hover_height + this, then lower to hover (extra phase)
    t_lower: float = 0.5                    # duration of that extra lowering phase
    collision_aware_grasp: bool = False     # try the other grasp directions if the descent self-collides

    appearance: Appearance = field(default_factory=Appearance)


# Gripper 15 on the real arm (LeRobot 0..100) -> MJCF rad. The real follower stalls below ~15
# (local/safety.py ARM_LIMITS), so the experts of the real setups never ask for less.
_GRIPPER_LO, _GRIPPER_HI = -0.174533, 1.7453292


def gripper_rad(units: float) -> float:
    """Gripper command in LeRobot units (0 = closed .. 100) -> MJCF rad."""
    return _GRIPPER_LO + units / 100 * (_GRIPPER_HI - _GRIPPER_LO)


REAL_GRIPPER_FLOOR_RAD = gripper_rad(15)

OBJECTS = {
    "cube": ObjectConfig(),
    # small red part of the office setup: a ~1.45 cm tube, 2.8 cm tall, on a ~2.1 cm flange. Estimated
    # from the photos; replace with calliper values. The jaws close on the tube just above the flange,
    # centred so the open fixed jaw clears the flange on the way down (1 mm).
    "cylinder": ObjectConfig(
        name="cylinder", shape="cylinder", half_width=0.0072, half_height=0.014, base_half_width=0.0105,
        base_half_height=0.002, mass=0.004, grasp_depth=0.005, in_site_z=-0.02 + 0.0105 + 0.001,
        task="Lift the red cylinder.",
    ),
}


def _real(cfg: LiftCubeConfig) -> LiftCubeConfig:
    """Home setup (2026-10-12): start pose read from the real arm, front camera fitted to the real photo,
    the robot painted white, table/floor textured from the real front camera, and randomized
    lighting, colours and camera pose to cover what was not matched exactly."""
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
        object_x=(0.17, 0.30),
        object_y=(-0.10, 0.10),
        gripper_closed=REAL_GRIPPER_FLOOR_RAD,
        # the start pose is low and forward (fingertips ~3.5 cm above the table), so a direct joint-space
        # approach can sweep the gripper into cubes at the far edge; approach from higher instead.
        approach_clearance=0.06,
        # near the base some wrist rolls put the wrist camera mount into the shoulder link
        collision_aware_grasp=True,
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


def _office(cfg: LiftCubeConfig) -> LiftCubeConfig:
    """Office setup (2026-10-14): black robot near the back edge of a light wood desk, pale wall on its right,
    front webcam on the front-left looking down. Fitted to assets/office/photo_front.png + photo_wrist.png
    (both views at once: sim segmentation vs colour labels of the photos; check with
    tools/overlay_photo.py); no tape measurements yet, so the camera jitter is generous."""
    return replace(
        cfg,
        # the folded pose of the photos, arm frame deg [23.3, -86.8, 82.4, 67.4, 90.4], gripper ~38 (fitted,
        # not read from the arm: replace with local/capture_setup.py values)
        home_qpos=(0.4070, -1.5152, 1.4383, 1.1761, 1.5780, 0.5520),
        home_noise=0.06,
        front=CameraConfig(pos=(0.290, 0.123, 0.414), lookat=(-0.028, -0.247, -0.186), fovy_deg=54.6, roll_deg=-5.1),
        camera_jitter_pos=0.04,
        camera_jitter_deg=4.0,
        # around where the photo has it (x 0.25, y 0.00); the wall is at y = -0.185
        object_x=(0.17, 0.30),
        object_y=(-0.07, 0.10),
        # this arm's gripper calibration puts the closed stop at ~0 (not ~15 like the home arm). The jaws
        # touch the 1.45 cm tube at ~9: closing to 8 pinches it (10 lets it go; lower only wedges it in
        # the tips), and keeps the real servo's stall small (run_act_policy.py --gripper-floor 8)
        gripper_closed=gripper_rad(8),
        approach_clearance=0.06,
        collision_aware_grasp=True,
        appearance=Appearance(
            robot_rgba=(0.05, 0.05, 0.05, 1.0),
            table_texture="office/table.png",
            floor_rgba=(0.12, 0.13, 0.12, 1.0),
            floor_z=-0.75,
            walls=(Wall("y", -0.185, (0.58, 0.64, 0.70, 1.0)),),
            table_center=(0.452, 0.307),         # back edge 9.6 cm behind the base, meets the wall
            table_half=(0.548, 0.493),
            texture_repeat=(6.0, 6.0),
            key_light_dir=(-0.3, -0.3, -1.0),     # from the room side, so the wall casts no shadow on the desk
            camera_gain={"front": (0.66, 0.77, 0.43)},   # this webcam's green cast; the wrist camera is neutral
            tint_jitter=0.12,
            light_jitter=0.35,
            light_dir_jitter=0.3,
            front_fovy_jitter_deg=5.0,
            wrist_fovy_range=(75.0, 92.0),        # fitted 85
            camera_gain_jitter=0.12,
            headlight_ambient=0.55,
            key_light_diffuse=0.35,
        ),
    )


def _home(cfg: LiftCubeConfig) -> LiftCubeConfig:
    """Home setup (2026-10-14): white robot near the back of a glossy reddish-brown wood table, front webcam in
    front of it, left of centre, looking down at it. Front camera fitted to assets/home/photo_front*.png at two
    poses with the joints read from the arm (folded rest + the start pose below); the wrist camera is a
    wide-angle lens with barrel distortion the pinhole sim camera can't copy, so its fovy is randomized widely.
    Same start pose and expert settings as "real" (the earlier, roughly fitted model of this home setup)."""
    return replace(
        cfg,
        # the start pose of the home policies (arm frame deg [7.9, 6.5, 11.3, 87.7, 93.0], gripper 79)
        home_qpos=(0.1381, 0.1135, 0.1979, 1.5305, 1.6226, 1.3421),
        home_noise=0.08,
        front=CameraConfig(pos=(0.370, 0.099, 0.486), lookat=(0.100, -0.021, -0.174), fovy_deg=42.3, roll_deg=9.0),
        camera_jitter_pos=0.04,
        camera_jitter_deg=4.0,
        # around where the photos have it (x 0.16, y 0.03)
        object_x=(0.15, 0.28),
        object_y=(-0.10, 0.10),
        gripper_closed=REAL_GRIPPER_FLOOR_RAD,
        approach_clearance=0.06,
        collision_aware_grasp=True,
        appearance=Appearance(
            robot_rgba=(0.93, 0.93, 0.91, 1.0),
            table_texture="home/table.png",
            floor_rgba=(0.85, 0.85, 0.85, 1.0),
            floor_z=-0.45,
            table_center=(0.4075, 0.085),        # x -0.085..0.90, y -0.45..0.62 (back edge 8.5 cm behind the base)
            table_half=(0.4925, 0.535),
            texture_repeat=(6.0, 6.0),
            tint_jitter=0.10,
            light_jitter=0.35,
            light_dir_jitter=0.3,
            front_fovy_jitter_deg=5.0,
            wrist_fovy_range=(85.0, 115.0),       # wide-angle real lens: the cube's size fits ~115, the jaw's ~85
            # measured on the start-pose photos: the front webcam is darker and much more saturated (the table
            # turns red-orange, the robot stays grey), the wrist camera a little brighter and paler
            camera_gain={"front": (0.62, 0.598, 0.665), "wrist": (1.101, 1.106, 1.112)},
            camera_saturation={"front": 1.69, "wrist": 1.1},
            object_rgba=(0.88, 0.30, 0.25, 1.0),  # the cube looks pinkish red under these lights
            camera_gain_jitter=0.06,              # the saturated front colours make larger casts look wrong
            headlight_ambient=0.55,
            key_light_diffuse=0.35,
        ),
    )


SETUPS = {"default": lambda cfg: cfg, "real": _real, "office": _office, "home": _home}
PROFILES = tuple(SETUPS)                    # the scripts' --profile choices


def make_config(profile: str = "default", obj: str = "cube") -> LiftCubeConfig:
    """profile = physical setup ('default': the original sim), obj = object to lift (OBJECTS)."""
    if profile not in SETUPS:
        raise ValueError(f"unknown profile {profile!r}; use one of {PROFILES}")
    if obj not in OBJECTS:
        raise ValueError(f"unknown object {obj!r}; use one of {tuple(OBJECTS)}")
    return SETUPS[profile](replace(LiftCubeConfig(), obj=replace(OBJECTS[obj])))
