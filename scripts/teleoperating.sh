#bash

set -a
source "$(dirname "$0")/.env"
set +a

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=my_follower \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=my_leader \
    --display_data=false