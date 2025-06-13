#!/usr/bin/env python
import rospy
from kuavo_msgs.msg import robotHandPosition
from std_msgs.msg import Header  # 用于设置时间戳
import subprocess


# 初始化ROS节点
rospy.init_node('fake_robot_hands_state_pub')

# 创建Publisher
pub = rospy.Publisher('/control_robot_hand_position_state', robotHandPosition, queue_size=10)

# 存储最后接收到的消息
last_msg = None
active = False  # 标志当前控制话题是否活跃

# 订阅控制话题的回调函数
def callback(msg):
    global last_msg, active
    last_msg = msg  # 更新最新的消息
    active = True   # 标记话题活跃
    # rospy.loginfo(f"Received a new message: {msg}")
    pub.publish(msg)  # 立即转发

# 订阅控制话题
sub = rospy.Subscriber('control_robot_hand_position', robotHandPosition, callback)

# 运行 rostopic pub 命令，发送一个初始消息
cmd = [
    "rostopic", "pub", "/control_robot_hand_position", "kuavo_msgs/robotHandPosition",
    "{left_hand_position:[0, 100, 0, 0, 0, 0], right_hand_position:[0, 100, 0, 0, 0, 0]}"
]
subprocess.Popen(cmd)

# =================================================Deployed code=================================================
# 定期检查是否仍然活跃，并发布最后一条消息
def timer_callback(event):
    global active, last_msg
    if not active and last_msg is not None:
        new_msg = robotHandPosition()  # 创建新消息
        new_msg.left_hand_position = last_msg.left_hand_position[:]  # 复制左手数据
        new_msg.right_hand_position = last_msg.right_hand_position[:]  # 复制右手数据
        new_msg.header = Header(stamp=rospy.Time.now())  # 设置当前时间戳
        pub.publish(new_msg)  # 重新发布消息
        # rospy.loginfo(f"Republished the last message: {new_msg}")
    active = False  # 置为非活跃状态，等待新的消息到来

# 创建定时器，每 10ms 触发一次
timer = rospy.Timer(rospy.Duration(0.01), timer_callback)

# 关闭回调
def shutdown_hook():
    rospy.loginfo("Shutting down the node...")

rospy.on_shutdown(shutdown_hook)
# =================================================Deployed code=================================================

# 保持节点运行
rospy.spin()
