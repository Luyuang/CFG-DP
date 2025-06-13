#!/bin/bash

# Parse arguments
while getopts "b:" opt; do
  case $opt in
    b) BAG_PATH="$OPTARG" ;;
    *) echo "Usage: $0 [-b bag_file_path]" >&2
       exit 1 ;;
  esac
done

# Default bag path if not specified
BAG_PATH=${BAG_PATH:-"/home/leju-ali/hx/kuavo/r_go.bag"}

SCRIPT_DIR=$(dirname "$0")

cleanup() {
    echo "Cleaning up processes..."
    pkill -P $$
    exit 0
}

trap cleanup SIGINT SIGTERM

# Change arm mode
rosservice call /arm_traj_change_mode "control_mode: 2" &

# Move from zero to start position
python "${SCRIPT_DIR}/tools/msg_forwarding_server.py" &
###########################
# you should change the -p to the position you want(in arm_joint_deg_poses.json)
###########################
python "${SCRIPT_DIR}/tools/arm_joint_deg_interpolator.py" -p r_arm_go_pos
rosbag play "${BAG_PATH}" --topics /kuavo_arm_traj --start=2 --duration=10
python "${SCRIPT_DIR}/tools/arm_joint_deg_interpolator.py" -p start

# Record test data
rosbag record -o 'Eval' \
    /kuavo_arm_traj_rad \
    /sensors_data_raw \
    /zedm/zed_node/right/image_rect_color/compressed \
    /joint_cmd \
    /cam_r/color/image_raw/compressed &

# Prepare environment
python "${SCRIPT_DIR}/tools/dex_hand_fake_state.py" &

# Run modeling
python "${SCRIPT_DIR}/eval.py"

wait
