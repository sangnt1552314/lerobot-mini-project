import os
import time

from dotenv import load_dotenv
from lerobot.motors.feetech import FeetechMotorsBus

load_dotenv()

ARMS = {
    "LEADER": os.environ["LEADER_PORT"],
    "FOLLOWER": os.environ["FOLLOWER_PORT"],
}

TRIALS = 20

for name, port in ARMS.items():
    print(f"\n=== {name} ===")
    print(f"Port: {port}")

    bus = FeetechMotorsBus(port=port, motors={})

    try:
        bus.connect(handshake=False)
        bus.set_baudrate(1_000_000)

        for motor_id in range(1, 7):
            success = 0

            for _ in range(TRIALS):
                model = bus.ping(motor_id, num_retry=1)
                if model == 777:
                    success += 1
                time.sleep(0.01)

            status = "OK" if success == TRIALS else "PROBLEM"
            print(
                f"ID {motor_id}: "
                f"{success}/{TRIALS} "
                f"[{status}]"
            )

    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass