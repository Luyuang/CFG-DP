#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest, controlLejuClawResponse

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('leju_claw_client_node')
    
    # 创建请求对象
    req = controlLejuClawRequest()
    req.data.name = ['left_claw', 'right_claw']
    req.data.position = [100, 0]
    req.data.velocity = [50, 100]
    req.data.effort = [1.0, 1.0]
    
    #确保服务启动
    rospy.wait_for_service('/control_robot_leju_claw')
    #调用服务并获取响应
    control_leju_claw = rospy.ServiceProxy('/control_robot_leju_claw', controlLejuClaw)
    res = control_leju_claw(req)