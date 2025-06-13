#!/bin/bash

# 启动第一个相机
source ~/kuavo_ws/devel/setup.bash
roslaunch realsense2_camera rs_camera.launch camera:=cam_h serial_no:=310222077896 &

# 等待一段时间，确保第一个相机启动完成
sleep 5

# 启动第二个相机
source ~/kuavo_ws/devel/setup.bash
roslaunch realsense2_camera rs_camera.launch camera:=cam_l serial_no:=218722270143 &

# 等待一段时间，确保第二个相机启动完成
sleep 5

# 启动第三个相机
source ~/kuavo_ws/devel/setup.bash
roslaunch realsense2_camera rs_camera.launch camera:=cam_r serial_no:=218622271266 &

# 等待一段时间，确保第三个相机启动完成
sleep 5

# 启动图像压缩节点
source ~/compressed_ws/devel/setup.bash
rosrun image_transport republish raw in:=/cam_h/color/image_raw compressed out:=/cam_h/color/ &
rosrun image_transport republish raw in:=/cam_l/color/image_raw compressed out:=/cam_l/color/ &
rosrun image_transport republish raw in:=/cam_r/color/image_raw compressed out:=/cam_r/color/ &

# 等待所有后台进程完成
wait

