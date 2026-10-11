import gymnasium as gym

TASK_ID = "LeIsaac-SO101-LiftCube-DualCam-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LiftCubeDualCamEnvCfg",
    },
)
