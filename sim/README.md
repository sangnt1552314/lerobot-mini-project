# ACT training on the NUS HPC

Shared training side of the MuJoCo pipeline ([`sim_mujoco/`](../sim_mujoco/README.md)); the Isaac Sim
pipeline that used to live here was removed (it needed an RTX GPU, which the HPC does not have).

| | |
|---|---|
| [`setup_hpc_train.sh`](setup_hpc_train.sh) | one-time venv: LeRobot 0.4.2 on the Singularity PyTorch image (`qsub pbs/setup_act_env.pbs`); pins in `requirements_hpc_train.txt` / `constraints_hpc_train.txt` |
| [`hpc_env.sh`](hpc_env.sh) | run a command in that environment (container + venv + caches + MuJoCo EGL) |
| [`train_act_hpc.sh`](train_act_hpc.sh) | `lerobot-train` ACT on one H200 (resume, W&B, checkpoint pruning); used by `pbs/mujoco_pipeline.pbs` |
| [`leisaac_ext/act_policy.py`](leisaac_ext/act_policy.py) | `LocalACTPolicy`: runs a checkpoint in-process (sim eval and the real robot) |
| [`tools/`](tools) | `inspect_dataset.py` (dataset check), `monitor_usage.py` (GPU/CPU usage CSV), and the H200 plumbing test / benchmark of `pbs/test_act_h200.pbs` on synthetic data |
