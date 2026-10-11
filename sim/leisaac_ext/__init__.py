"""Out-of-tree LeIsaac tasks for this repository.

Importing this package registers the Gym ids below. It only imports gymnasium, so it is
safe to import before Isaac Sim starts; the env config itself is loaded lazily by
isaaclab_tasks.utils.parse_env_cfg after the AppLauncher is up.
"""

from . import lift_cube_dualcam  # noqa: F401
from .lift_cube_dualcam import TASK_ID

__all__ = ["TASK_ID"]
