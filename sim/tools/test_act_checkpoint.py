"""Load an ACT checkpoint the way sim/scripts/eval_act.py does and run it on a dummy observation.

    python sim/tools/test_act_checkpoint.py outputs/act_liftcube_h200/checkpoints/last/pretrained_model

Checks the files eval needs are present, loads them with LocalACTPolicy (sim/leisaac_ext/act_policy.py,
no Isaac imports), feeds state (6,) float32 + front/wrist (480, 640, 3) uint8, and checks that a finite
(6,) action comes back. Says nothing about policy quality.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    files = sorted(p.name for p in ckpt.iterdir())
    print(f"checkpoint {ckpt.resolve()}\nfiles      {files}")
    required = ["config.json", "model.safetensors"]
    missing = [f for f in required if f not in files]
    if not any(f.startswith("policy_preprocessor") and f.endswith(".json") for f in files):
        missing.append("policy_preprocessor*.json")
    if not any(f.startswith("policy_postprocessor") and f.endswith(".json") for f in files):
        missing.append("policy_postprocessor*.json")
    if missing:
        sys.exit(f"FAIL: missing {missing}")

    from leisaac_ext.act_policy import LocalACTPolicy  # same import as sim/scripts/eval_act.py

    policy = LocalACTPolicy(ckpt, device=args.device)
    print(f"cameras    {policy.camera_keys}\nstate_dim  {policy.state_dim}")
    rng = np.random.default_rng(0)
    obs = {"observation.state": rng.uniform(-50, 50, 6).astype(np.float32)}
    for cam in policy.camera_keys:
        obs[f"observation.images.{cam}"] = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    policy.reset()
    action = policy(obs, task="Lift the red cube.", robot_type="so101_follower")
    print(f"action     shape {action.shape} dtype {action.dtype} values {np.round(action, 3)}")
    if action.shape != (6,) or not np.all(np.isfinite(action)):
        sys.exit("FAIL: expected a finite (6,) action")
    print("RESULT: OK")


if __name__ == "__main__":
    main()
