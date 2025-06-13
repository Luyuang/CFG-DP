#!/usr/bin/env python3
# pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
# pip install roslz4 --extra-index-url https://rospypi.github.io/simple/
import numpy as np
import cv2
import rosbag
from pprint import pprint
import os
import glob
from collections import defaultdict 
import yaml
import time

# ================ 机器人关节信息定义 ================

DEFAULT_LEG_JOINT_NAMES=[
    "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
    "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll",
]
DEFAULT_ARM_JOINT_NAMES = [
    "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",
    "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",
]
DEFAULT_HEAD_JOINT_NAMES = [
    "head_yaw", "head_pitch"
]
DEFAULT_DEXHAND_JOINT_NAMES = [
    "left_qiangnao_1", "left_qiangnao_2","left_qiangnao_3","left_qiangnao_4","left_qiangnao_5","left_qiangnao_6",
    "right_qiangnao_1", "right_qiangnao_2","right_qiangnao_3","right_qiangnao_4","right_qiangnao_5","right_qiangnao_6",
]
DEFAULT_LEJUCLAW_JOINT_NAMES = [
    "left_claw", "right_claw",
]

DEFAULT_JOINT_NAMES_LIST = DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES

DEFAULT_JOINT_NAMES = {
    "full_joint_names": DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES,
    "leg_joint_names": DEFAULT_LEG_JOINT_NAMES,
    "arm_joint_names": DEFAULT_ARM_JOINT_NAMES,
    "head_joint_names": DEFAULT_HEAD_JOINT_NAMES,
}

# ================ 数据转换信息定义 ================
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'lerobot_dataset.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DEFAULT_CAMERA_NAMES = config['default_camera_names']   # ['wrist_cam_l', 'wrist_cam_r', 'head_cam_l', 'head_cam_r']
TRAIN_HZ = config['train_hz']   # 30
MAIN_TIMELINE_FPS = config['main_timeline_fps'] # 30
SAMPLE_DROP = config['sample_drop'] # 10
CONTROL_HAND_SIDE = config['control_hand_side'] # 'left' or 'right' or 'both'

if CONTROL_HAND_SIDE == 'left':
    SLICE_ROBOT = [(12, 19),(19, 19)]
elif CONTROL_HAND_SIDE == 'right':
    SLICE_ROBOT = [(12, 12),(19, 26)]
elif CONTROL_HAND_SIDE == 'both':
    SLICE_ROBOT = [(12, 19),(19, 26)]
    
SLICE_DEX = config['dex_slice'] #0-6: 左手关节 6-12: 右手关节
SLICE_CLAW = config['claw_slice'] #0-2: 左手夹爪 2-4: 右手夹爪

IS_BINARY = config['is_binary'] 
DELTA_ACTION = config['delta_action']
RELATIVE_START = config['relative_start']

RESIZE_W = config['resize']['width']
RESIZE_H = config['resize']['height']

ONLY_HALF_UP_BODY = config['only_half_up_body'] # 是否只使用上半身数据
USE_LEJU_CLAW = config['use_leju_claw'] # 是否使用乐聚夹爪
USE_QIANGNAO = config['use_qiangnao'] # 是否使用强脑灵巧手

# ================ 数据处理函数定义 ==================
class KuavoMsgProcesser:
    """
    Kuavo 话题处理函数
    """
    @staticmethod
    def process_color_image(msg):
        """
        Process the color image.
        Args:
            msg (sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage): The color image message.
        Returns:
             Dict:
                - data(np.ndarray): Image data with shape (height, width, 3).
                - "timestamp" (float): The timestamp of the image.
        """
        if hasattr(msg, 'encoding'):
            if msg.encoding != 'rgb8':
                # Handle different encodings here if necessary
                raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected 'rgb8'.")

            # Convert the ROS Image message to a numpy array
            img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

            # If the image is in 'bgr8' format, convert it to 'rgb8'
            if msg.encoding == 'bgr8':
                cv_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            else:
                cv_img = img_arr
        else:
            # 处理 CompressedImage
            img_arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("Failed to decode compressed image")
            # 色域转换由BGR->RGB
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img=cv2.resize(cv_img,(RESIZE_W,RESIZE_H)) ### ATT: resize the image to 640x480(w * h)
        return {"data": cv_img, "timestamp": msg.header.stamp.to_sec()}


    @staticmethod
    def process_joint_state(msg):
        """
            Args:
                msg (kuavo_msgs/sensorsData): The joint state message.
            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        joint_q = msg.joint_data.joint_q
        return {"data": joint_q, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd(msg):
        """
            Args:
                msg (kuavo_msgs/jointCmd): The joint state message.

            Returns:
                Dict:
                    - data(np.ndarray): The joint state data with shape (28,).
                    - "timestamp" (float): The timestamp of the joint state.
        """
        # radian
        return {"data": msg.joint_q, "timestamp": msg.header.stamp.to_sec()}
    @staticmethod
    def process_claw_state(msg):
        """
            Args:
                msg (kuavo_sdk/lejuClawState): The claw state message.
            Returns:
                Dict:
                    - data(np.ndarray): The claw state data with shape (2,).
                    - "state" (float): The state of the claws state.
        """
        state= msg.data.position
        return { "data": state, "timestamp": msg.header.stamp.to_sec() }

    @staticmethod
    def process_claw_cmd(msg):
        position= msg.data.position
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_qiangnao_state(msg):
        state= list(msg.left_hand_position)
        state.extend(list(msg.right_hand_position))
        return { "data": state, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_qiangnao_cmd(msg):
        position= list(msg.left_hand_position)
        position.extend(list(msg.right_hand_position))
        return { "data": position, "timestamp": msg.header.stamp.to_sec() }
    
    @staticmethod
    def process_sensors_data_raw_extract_imu(msg):
        imu_data = msg.imu_data
        gyro = imu_data.gyro
        acc = imu_data.acc
        free_acc = imu_data.free_acc
        quat = imu_data.quat

        # 将数据合并为一个NumPy数组
        imu = np.array([gyro.x, gyro.y, gyro.z,
                        acc.x, acc.y, acc.z,
                        free_acc.x, free_acc.y, free_acc.z,
                        quat.x, quat.y, quat.z, quat.w])

        return {"data": imu, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_sensors_data_raw_extract_arm(msg):
        """
        Processes raw joint state data from a given message by extracting the portion relevant to the arm.

        Parameters:
            msg: The input message containing joint state information.

        Returns:
            dict: A dictionary with processed joint state data. The 'data' field is sliced to include only indices 12 through 25.

        Notes:
            This function uses KuavoMsgProcesser.process_joint_state to initially process the input message and then extracts the specific range of data for further use.
        """
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:26]
        return res

    @staticmethod
    def process_sensors_data_raw_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_state(msg)
        res["data"] = res["data"][12:]
        return res

    @staticmethod
    def process_joint_cmd_extract_arm_head(msg):
        res = KuavoMsgProcesser.process_joint_cmd(msg)
        res["data"] = res["data"][12:]
        return res

    @staticmethod
    def process_depth_image(msg):
        """
        Process the depth image.

        Args:
            msg (sensor_msgs/Image): The depth image message.

        Returns:
            Dict:
                - data(np.ndarray): Depth image data with shape (height, width).
                - "timestamp" (float): The timestamp of the image.
        """
        # Check if the image encoding is '16UC1' which is a common encoding for depth images
        if msg.encoding != '16UC1':
            raise ValueError(f"Unsupported encoding: {msg.encoding}. Expected '16UC1'.")

        # Convert the ROS Image message to a numpy array
        img_arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        # The depth image is already in '16UC1' format, so no conversion is needed
        return {"data": img_arr, "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_step_from_seq(msg):
        """
        从消息的 header.seq 字段提取当前步数（帧序号）。
        Args:
            msg: ROS 消息，包含 header.seq 字段。
        Returns:
            Dict:
                - data (int): 当前步数（从零开始）。
                - timestamp (float): 消息的时间戳。
        """
        # 提取帧序号
        raw_step = msg.header.seq

        # 预处理：将帧序号从零开始
        # 假设初始帧序号为 `initial_seq`，需要在外部逻辑中动态设置
        if not hasattr(KuavoMsgProcesser, "_initial_seq"):
            KuavoMsgProcesser._initial_seq = raw_step  # 初始化起始帧序号

        step = raw_step - KuavoMsgProcesser._initial_seq

        return {"data": step, "timestamp": msg.header.stamp.to_sec()}

class KuavoRosbagReader:
    def __init__(self):
        self._msg_processer = KuavoMsgProcesser()
        self._topic_process_map = {
            "observation.state": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_joint_state,
            },
            "action": {
                "topic": "/joint_cmd",
                "msg_process_fn": self._msg_processer.process_joint_cmd,
            },
            "observation.imu": {
                "topic": "/sensors_data_raw",
                "msg_process_fn": self._msg_processer.process_sensors_data_raw_extract_imu,
            },
            "observation.claw": {
                # 末端数据二指夹爪的位置状态信息
                "topic": "/leju_claw_state",
                "msg_process_fn": self._msg_processer.process_claw_state,
            },
            "action.claw": {
                # 末端数据二指夹爪的运动信息位置
                "topic": "/leju_claw_command",
                "msg_process_fn": self._msg_processer.process_claw_cmd,
            },
            "observation.qiangnao": {
                "topic": "/control_robot_hand_position_state",
                "msg_process_fn": self._msg_processer.process_qiangnao_state,
            },
            "action.qiangnao": {
                "topic": "/control_robot_hand_position",
                "msg_process_fn": self._msg_processer.process_qiangnao_cmd,
            },
            "observation.step": {
                "topic": "/control_robot_hand_position",
                "msg_process_fn": self._msg_processer.process_step_from_seq,
            },
        }
        for camera in DEFAULT_CAMERA_NAMES:
            # observation.images.{camera}.depth  => color images
            if 'wrist' in camera:
                self._topic_process_map[f"{camera}"] = {
                    "topic": f"/{camera[-5:]}/color/image_raw/compressed",   # "/{camera}/color/compressed", 新刷的20.04orin镜像可以直接发布压缩图像，不用额外的压缩节点
                    "msg_process_fn": self._msg_processer.process_color_image,
                }
            elif 'head_cam_l' in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/zedm/zed_node/left/image_rect_color/compressed",
                "msg_process_fn": self._msg_processer.process_color_image,
            }
            elif 'head_cam_r' in camera:
                self._topic_process_map[f"{camera}"] = {
                "topic": f"/zedm/zed_node/right/image_rect_color/compressed",
                "msg_process_fn": self._msg_processer.process_color_image,
            }
                
            # observation.images.{camera}.depth => depth images
            # self._topic_process_map[f"{camera}.depth"] = {
            #     "topic": f"/{camera}/depth/image_rect_raw",
            #     "msg_process_fn": self._msg_processer.process_depth_image,
            # }

    def load_raw_rosbag(self, bag_file: str):
        bag = rosbag.Bag(bag_file)      
        return bag
    
    def print_bag_info(self, bag: rosbag.Bag):
        pprint(bag.get_type_and_topic_info().topics)
     
    def process_rosbag(self, bag_file: str):
        """
        Process the rosbag file and return the processed data.

        Args:
            bag_file (str): The path to the rosbag file.

        Returns:
            Dict: The processed data.
        """
        bag = self.load_raw_rosbag(bag_file)
        data = {}

        for key, topic_info in self._topic_process_map.items():
            topic = topic_info["topic"]
            msg_process_fn = topic_info["msg_process_fn"]
            data[key] = []

            for _, msg, t in bag.read_messages(topics=topic):
                msg_data = msg_process_fn(msg)
                # 如果没有 header.stamp或者时间戳是远古时间不合要求，使用bag的时间戳 
                correct_timestamp = t.to_sec() 
                msg_data["timestamp"] = correct_timestamp
                data[key].append(msg_data)

        data_aligned = self.align_frame_data(data)
        return data_aligned
    
    def align_frame_data(self, data: dict):
        aligned_data = defaultdict(list)
        main_timeline = max(
            DEFAULT_CAMERA_NAMES, 
            key=lambda cam_k: len(data.get(cam_k, []))
        )
        jump = MAIN_TIMELINE_FPS // TRAIN_HZ
        main_img_timestamps = [t['timestamp'] for t in data[main_timeline]][SAMPLE_DROP:-SAMPLE_DROP][::jump]
        min_end = min([data[k][-1]['timestamp'] for k in data.keys() if len(data[k]) > 0])
        main_img_timestamps = [t for t in main_img_timestamps if t < min_end]

        for stamp in main_img_timestamps:
            stamp_sec = stamp
            for key, v in data.items():
                if len(v) > 0:
                    if key == "observation.step":
                        # 对齐步数
                        aligned_data[key].append({"data": len(aligned_data[main_timeline]), "timestamp": stamp_sec})
                    else:
                        this_obs_time_seq = [this_frame['timestamp'] for this_frame in v]
                        time_array = np.array([t for t in this_obs_time_seq])
                        idx = np.argmin(np.abs(time_array - stamp_sec))
                        aligned_data[key].append(v[idx])
                else:
                    aligned_data[key] = []
        return aligned_data
    
    def list_bag_files(self, bag_dir: str):
        bag_files = glob.glob(os.path.join(bag_dir, '*.bag'))
        bag_files.sort()
        return bag_files
    
    def process_rosbag_dir(self, bag_dir: str):
        all_data = []
        # 按照文件名排序，获取 bag 文件列表
        bag_files = self.list_bag_files(bag_dir)
        episode_id = 0
        for bf in bag_files:
            print(f"Processing bag file: {bf}")
            episode_data = self.process_rosbag(bf)
            all_data.append(episode_data)
        
        return all_data
    
    

if __name__ == '__main__':
    bag_file = '/home/leju-ali/hx/kuavo/Task16_nls/rosbag/nls_0_20250408_174422.bag'
    reader = KuavoRosbagReader()
    data = reader.process_rosbag(bag_file)

    print("输出每个时间戳对应的所有数据：\n")
    for i in range(len(next(iter(data.values()), []))):  # 遍历第一个话题的帧数
        for key in sorted(data.keys()):
            if len(data[key]) > i:
                timestamp = data[key][i].get("timestamp", "N/A")
                frame_data = data[key][i].get("data", "N/A")
                print(f"Timestamp: {timestamp}")
                print(f"  {key}: {frame_data}")
        print("-" * 40)

