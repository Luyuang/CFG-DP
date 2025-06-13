import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import math

from kuavo_utils.pcd_util import  optimized_extract_xyzrgb_from_pointcloud2

from kuavo_utils.pcd_util import farthest_point_sampling, preprocess_point_cloud


class ProcessUtil:
    @staticmethod
    def process_compressed_image(msg, resize=None):
        """
        Args:
            msg: CompressedImage
                The compressed image message.
            resize: tuple (width, height), optional
                Resize the image to this size. If None, the original size is used.
        
        Returns:
            Dict:
                - "data" (np.array): Image data with shape (height, width, 3).
                - "timestamp" (float): The timestamp of the image in seconds.
        """
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(cv_img, (resize[0], resize[1])) if resize else cv_img
        return {"data": resized_img, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_record_arm_hand_pose(msg):
        """
        args:
            msg: recordArmHandPose
        return:
            left_right_eef: np.array (12,)
        """
        left_right_eef = np.concatenate([
            np.concatenate((
                np.array(pose.pos_xyz), 
                R.from_quat(pose.quat_xyzw).as_euler('xyz')
            ))
            for pose in [msg.left_pose, msg.right_pose]
        ])
        return {"data": left_right_eef, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint(msg, is_cmd: bool = True):
        """Process joint data to extract joint angles"""
        if is_cmd:
            joint = [i * math.pi / 180 for i in list(msg.position)]
        else:
            joint = msg.q
        return {"data": joint, "timestamp": msg.header.stamp.to_sec()}
    


    @staticmethod
    def process_hand_data(msg, is_cmd: bool = True, is_binary: bool = False, eef_type: str = 'dex'):
        """Process Dexterous hand data to extract hand state"""
        if eef_type == "dex":
            hand_data = [float(num) for num in msg.left_hand_position + msg.right_hand_position]
            if is_binary:
                hand_data = [
                    1 if figure > 50 else 0 for figure in hand_data 
                ]
        elif eef_type == "leju_claw":
            hand_data = msg.data.position   
            if is_binary:
                hand_data = [
                    1 if hand_data[0] > 50 else 0,
                    1 if hand_data[1] > 50 else 0
                ]
        return {"data": hand_data, "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_pcd(msg):
        # import visualizer
        """Process point cloud data to extract point cloud"""
        pcd = optimized_extract_xyzrgb_from_pointcloud2(msg)
        # xyz = pcd[:, :3]  # 提取前 3 列 (x, y, z)
        # rgb = pcd[:, 3:]  # 提取后 3 列 (r, g, b)
        # visualizer.visualize_pointcloud(pcd)
        
        # pcd = preprocess_point_cloud(pcd)
        # visualizer.visualize_pointcloud(pcd)
        return {"data": pcd, "timestamp": msg.header.stamp.to_sec()}
    
    def process_raw_image(msg, resize=None):
        """
        Args:
            msg: sensor_msgs/Image
                The raw image message.
            resize: tuple (width, height), optional
                Resize the image to this size. If None, the original size is used.
        
        Returns:
            Dict:
                - "data" (np.array): Image data with shape (height, width, 3).
                - "timestamp" (float): The timestamp of the image in seconds.
        """
        from cv_bridge import CvBridge
        # 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 图像
        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"Failed to convert image: {e}")
            return None

        # 将 BGR 图像转换为 RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # 如果需要调整大小
        resized_img = cv2.resize(cv_img, (resize[0], resize[1])) if resize else cv_img

        return {"data": resized_img, "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_jointCmd(msg):
        """Process joint data to extract joint angles"""
        joint = msg.joint_q[12:26]
        return {"data": joint, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_sensorsData(msg):
        """Process joint data to extract joint angles"""
        joint = msg.joint_data.joint_q[12:26]
        return {"data": joint, "timestamp": msg.header.stamp.to_sec()}
    
    @staticmethod
    def process_leju_claw(msg, is_cmd: bool = False, is_binary: bool = False, HAND_CLOSE_STATE:str = None):
        """Process leju_claw_state hand data to extract hand state"""
        if is_cmd:
            dex_hand = msg.data.position
        else:
            dex_hand = msg.data.position
        if is_binary:
            dex_hand = [
                1 if dex_hand[0] == HAND_CLOSE_STATE[0] else 0,
                1 if dex_hand[1] == HAND_CLOSE_STATE[1] else 0
            ]
        return {"data": dex_hand, "timestamp": msg.header.stamp.to_sec()}