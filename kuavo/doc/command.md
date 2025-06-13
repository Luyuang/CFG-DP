# 常用指令
## arm control
'''bash
rostopic pub /kuavo_arm_traj sensor_msgs/JointState "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
name: ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5', 'arm_joint_6', 'arm_joint_7', 'arm_joint_8', 'arm_joint_9', 'arm_joint_10', 'arm_joint_11', 'arm_joint_12', 'arm_joint_13', 'arm_joint_14']
position: [-9.849275097595948, 25.222922208483315, -13.44202991848696, -86.66281227708686, -54.83745165154792, -12.633218924390484, -11.147216795054188, 2.811977048081159, -9.90131013167694, 5.988853622402726, -33.41956556880875, 30.359264771786457, 15.21255154261246, -10.163522174546088, ]
velocity: []
effort: []"
'''

## head control
'''
rostopic pub /robot_head_motion_data kuavo_msgs/robotHeadMotionData  "joint_data: [0.0, 35.0]"
'''

## hand control
'''
rostopic pub /control_robot_hand_position kuavo_msgs/robotHandPosition "{left_hand_position:[0, 0, 0, 0, 0, 0], right_hand_position:[0, 0, 0, 0, 0, 0]}"
'''