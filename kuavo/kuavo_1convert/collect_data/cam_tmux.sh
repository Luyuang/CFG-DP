#!/bin/bash

# 创建一个新的 tmux 会话
tmux new-session -d -s camera_session

# 启动第一个相机（在第一个窗格中）
tmux send-keys -t camera_session "source ~/kuavo_ws/devel/setup.bash" C-m
tmux send-keys -t camera_session "roslaunch realsense2_camera rs_camera.launch camera:=cam_h serial_no:=310222077896" C-m

# 水平分割窗口，启动第二个相机
tmux split-window -h -t camera_session
tmux send-keys -t camera_session "source ~/kuavo_ws/devel/setup.bash" C-m
tmux send-keys -t camera_session "roslaunch realsense2_camera rs_camera.launch camera:=cam_l serial_no:=218722270143" C-m

# 垂直分割窗口，启动第三个相机
tmux split-window -v -t camera_session
tmux send-keys -t camera_session "source ~/kuavo_ws/devel/setup.bash" C-m
tmux send-keys -t camera_session "roslaunch realsense2_camera rs_camera.launch camera:=cam_r serial_no:=218622271266" C-m

# 创建一个新的窗口，用于图像压缩节点
tmux new-window -t camera_session
tmux send-keys -t camera_session:1 "source ~/compressed_ws/devel/setup.bash" C-m
tmux send-keys -t camera_session:1 "rosrun image_transport republish raw in:=/cam_h/color/image_raw compressed out:=/cam_h/color/" C-m

# 水平分割窗口，启动第二个图像压缩节点
tmux split-window -h -t camera_session:1
tmux send-keys -t camera_session:1 "source ~/compressed_ws/devel/setup.bash" C-m
tmux send-keys -t camera_session:1 "rosrun image_transport republish raw in:=/cam_l/color/image_raw compressed out:=/cam_l/color/" C-m

# 垂直分割窗口，启动第三个图像压缩节点
tmux split-window -v -t camera_session:1
tmux send-keys -t camera_session:1 "source ~/compressed_ws/devel/setup.bash" C-m
tmux send-keys -t camera_session:1 "rosrun image_transport republish raw in:=/cam_r/color/image_raw compressed out:=/cam_r/color/" C-m

# 附加到 tmux 会话
tmux attach-session -t camera_session