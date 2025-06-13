#!/bin/bash
source ~/robot_ws/bin/activate
source ~/kuavo_il/kuavo/kuavo_ws/devel/setup.bash
rosservice call /arm_traj_change_mode "control_mode: 2" 

python3 /home/leju_kuavo/kuavo_il/kuavo/kuavo_3deploy/tools/dex_hand_fake_state.py

rosbag play /home/leju_kuavo/Data/rosbag/rosbag_out/go_target_pose/go_target_pose_20250422_211103_0.bag --topics /kuavo_arm_traj --start=0 --duration=12
python3 ~/kuavo_il/kuavo/kuavo_3deploy/eval.py

