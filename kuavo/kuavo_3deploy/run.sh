#!/bin/bash

# Get script directory
SCRIPT_DIR=$(dirname "$0")

# Function to kill child processes on exit
cleanup() {
    echo "Cleaning up processes..."
    pkill -P $$  # Kill all child processes
    exit 0
}

# Trap signals for proper cleanup
trap cleanup SIGINT SIGTERM

# Run processes with terminal output
echo "Starting processes..."
python "${SCRIPT_DIR}/tools/dex_hand_fake_state.py" &
FAKE_STATE_PID=$!

python "${SCRIPT_DIR}/tools/msg_forwarding_server.py" &
MSG_SERVER_PID=$!

rosbag record \
    /joint_cmd \
    /sensors_data_raw \
    /leju_claw_command \
    /leju_claw_state \
    /control_robot_hand_position \
    /control_robot_hand_position_state \
    /zedm/zed_node/right/image_rect_color/compressed \
    /kuavo_arm_traj \
    /kuavo_arm_traj_rad &
ROSBAG_PID=$!

python "${SCRIPT_DIR}/eval.py"

# Wait for processes to complete
wait $FAKE_STATE_PID $MSG_SERVER_PID $ROSBAG_PID
