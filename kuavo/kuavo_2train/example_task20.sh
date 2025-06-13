#!/bin/bash
# Example script for Task20_conveyor_pick

# This script demonstrates how to use the convert_and_train.sh script
# with the specific parameters for Task20_conveyor_pick

# Path to the raw rosbag directory
RAW_DIR="/home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag"

# Run the conversion and training
./kuavo/convert_and_train.sh \
  --task_name Task20_conveyor_pick \
  --raw_dir "$RAW_DIR" \
  --version v0 \
  --num_processes 2 \
  --port 29503 \
  --policy_type act

# The above command will:
# 1. Convert the rosbag data to lerobot format
#    Output: /home/leju-ali/hx/kuavo/Task20_conveyor_pick/v0/lerobot
# 2. Train a model using the converted data
#    Output: /home/leju-ali/hx/kuavo/Task20_conveyor_pick/v0/train_lerobot

# To run only the conversion step:
# ./kuavo/convert_and_train.sh \
#   --task_name Task20_conveyor_pick \
#   --raw_dir "$RAW_DIR" \
#   --version v0 \
#   --skip_training

# To run only the training step (assuming data is already converted):
# ./kuavo/convert_and_train.sh \
#   --task_name Task20_conveyor_pick \
#   --raw_dir "$RAW_DIR" \
#   --version v0 \
#   --num_processes 2 \
#   --port 29503 \
#   --policy_type act \
#   --skip_conversion
