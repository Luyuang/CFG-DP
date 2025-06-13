#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import JointState
import numpy as np
import time
import subprocess
from kuavo_msgs.msg import lejuClawState, sensorsData, robotHandPosition, armTargetPoses    # type: ignore
import os
import json
import logging
from termcolor import colored
from datetime import datetime
from typing import List, Optional
import argparse

# 常量定义
NUM_JOINTS = 14  # 机械臂关节数量
INTERPOLATION_STEPS = 30  # 默认插值步数
JOINT_POSITION_TOLERANCE = 5.0  # 关节位置容差(度)

# 运行 rostopic pub 命令，发送一个初始消息
cmd = [
    "rostopic", "pub", "/control_robot_hand_position", "kuavo_msgs/robotHandPosition",
    "{left_hand_position:[100, 100, 0, 0, 0, 0], right_hand_position:[0, 100, 0, 0, 0, 0]}"
]
subprocess.Popen(cmd)
print('rostopic pub /control_robot_hand_position...')

def publish_joint_states(q_now: List[float]) -> None:
    """
    发布当前关节状态到指定话题。
    
    Args:
        q_now: 当前关节位置列表，长度应为NUM_JOINTS
        
    Raises:
        ValueError: 如果关节位置数量不正确
    """
    if len(q_now) != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joint positions, got {len(q_now)}")
        
    pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    rospy.sleep(0.1)

    msg = JointState()
    msg.name = [f"arm_joint_{i}" for i in range(1, NUM_JOINTS+1)]
    msg.header.stamp = rospy.Time.now()
    msg.position = np.array(q_now, dtype=np.float64)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    positions_str = ", ".join([f"{x:.2f}" for x in msg.position])
    rospy.loginfo(colored(
        f"[{timestamp}] 发布关节位置: [{positions_str}]", 
        "yellow", 
        attrs=["bold"]
    ))
    pub.publish(msg)


def interpolate_joint_positions(
    q0: List[float], 
    q1: List[float], 
    steps: int = INTERPOLATION_STEPS
) -> List[List[float]]:
    """
    生成从 q0 到 q1 的平滑插值轨迹。
    
    Args:
        q0: 初始关节位置列表
        q1: 目标关节位置列表
        steps: 插值步数，默认为INTERPOLATION_STEPS
        
    Returns:
        包含插值位置的列表，每个元素是一个长度为NUM_JOINTS的列表
        
    Raises:
        ValueError: 如果输入关节位置数量不正确
    """
    if len(q0) != NUM_JOINTS or len(q1) != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joint positions")
        
    return [
        [
            q0[j] + i / float(steps) * (q1[j] - q0[j])
            for j in range(NUM_JOINTS)
        ]
        for i in range(steps)
    ]

def load_joint_positions(file_path: str, pos_name: str) -> List[float]:
    """从JSON文件加载指定位置的关节角度
    
    Args:
        file_path: JSON文件路径
        pos_name: 位置名称
        
    Returns:
        关节角度列表
        
    Raises:
        FileNotFoundError: 如果文件不存在
        KeyError: 如果位置名称不存在
    """
    try:
        with open(file_path, "r") as f:
            poses = json.load(f)
        return poses[pos_name]
    except FileNotFoundError:
        rospy.logerr(f"关节位置文件未找到: {file_path}")
        raise
    except KeyError:
        rospy.logerr(f"未找到位置名称: {pos_name}")
        raise

def main() -> None:
    """主函数，执行关节轨迹插值和发布"""
    parser = argparse.ArgumentParser(
        description="机械臂关节轨迹插值程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p", "--pos_name", 
        type=str, 
        required=True,
        help="目标位置名称(必须存在于arm_joint_deg_poses.json中)"
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=30,
        help="插值步数(默认: 30)"
    )
    args = parser.parse_args()
    
    rospy.init_node('sim_traj', anonymous=True)
    
    # 获取当前关节位置
    try:
        msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=5.0)
        q0 = [np.rad2deg(float(q)) for q in msg.joint_data.joint_q[12:26]]
        rospy.loginfo(f"当前关节位置: {[f'{x:.2f}' for x in q0]}")
    except rospy.ROSException as e:
        rospy.logerr(f"获取关节位置失败: {str(e)}")
        raise

    try:
        # 加载目标位置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pos_file = os.path.join(current_dir, "arm_joint_deg_poses.json")
        q1 = load_joint_positions(pos_file, args.pos_name)
        rospy.loginfo(f"开始移动到位置: {args.pos_name}")
        
        # 生成插值轨迹
        q_list = interpolate_joint_positions(q0, q1, args.steps)
        
        # 验证初始位置差异
        q_diff = np.array(q_list[2]) - np.array(q0)
        # print(colored(
        #     f"初始关节位置差异: {q_diff}", 
        #     "yellow", 
        #     attrs=["bold"]
        # ))
        if not np.all(np.abs(q_diff) < JOINT_POSITION_TOLERANCE):
            rospy.logwarn(
                f"初始关节位置差异超过{JOINT_POSITION_TOLERANCE}度: {q_diff}"
            )
            exit()
            
        print(colored('Running... ', 'green', attrs=['bold']))
        # 发布轨迹
        for q in q_list:
            publish_joint_states(q)
            time.sleep(0.01)
            
        rospy.loginfo("轨迹发布完成")
        
    except Exception as e:
        rospy.logerr(f"轨迹生成或发布失败: {str(e)}")
        raise


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
