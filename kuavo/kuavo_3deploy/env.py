from typing import Optional
from pathlib import Path
import numpy as np
import time
import math
from sensor_msgs.msg import JointState

# =============================================================================
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

import cv2
from collections import deque
# from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Union, Dict, Callable
from tqdm import tqdm  
import matplotlib.pyplot as plt
import signal
import sys
from sensor_msgs.msg import Image

from kuavo_msgs.msg import lejuClawState, sensorsData, robotHandPosition, armTargetPoses    # type: ignore
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest, controlLejuClawResponse
# =============================================================================

from deploy_config import *

class ObsBuffer:
    def __init__(self, 
                 obs_key_map: Optional[Dict[str, Dict[str, str]]] = None
                 ) -> None:
        
        # self.img_buffer_size = img_buffer_size
        # self.robot_state_buffer_size = robot_state_buffer_size
        self.obs_key_map = obs_key_map if obs_key_map is not None else DEFAULT_OBS_KEY_MAP
        
        # TODO: replace `self.obs_key_map["low_dim"][key]["frequency"]` with `self.obs_buffer_size[key]`
        self.obs_buffer_size = {key: self.obs_key_map["img"][key]["frequency"] for key in self.obs_key_map["img"]}
        self.obs_buffer_size.update({key: self.obs_key_map["low_dim"][key]["frequency"] for key in self.obs_key_map["low_dim"]})
        
        self.obs_buffer_data = {key: {"data": deque(maxlen=self.obs_key_map["img"][key]["frequency"]),"timestamp": deque(maxlen=self.obs_key_map["img"][key]["frequency"]),} \
                                for key in self.obs_key_map["img"]}
        self.obs_buffer_data.update({key: {"data": deque(maxlen=self.obs_key_map["low_dim"][key]["frequency"]),"timestamp": deque(maxlen=self.obs_key_map["low_dim"][key]["frequency"]),} \
                                    for key in self.obs_key_map["low_dim"]})

        self.callback_key_map = {
            CompressedImage: self.compressedImage_callback,
            sensorsData: self.sensorsData_callback,
            robotHandPosition: self.robotHandPosition_callback,
            # recordArmHandPose: self.recordArmHandPose_callback,
            # JointState: self.joint_callback,
            # robotArmQVVD: self.robotArmQVVD_callback,
            # robot_hand_eff: self.robot_hand_eff_callback,
            # robotHandPosition: self.robotHandPosition_callback,
            # Image: self.Image_callback,
            lejuClawState: self.lejuClawState_callback, # 添加


        }
        self.suber_dict = {}
        self.setup_subscribers()
    
    def create_callback(self, callback, topic_key, handle = dict):
        return lambda msg: callback(msg, topic_key, handle)
    def setup_subscribers(self):
        for obs_cls, topics in self.obs_key_map.items():
            for topic_key, topic_info in topics.items():
                topic_name = topic_info["topic"]
                msg_type = topic_info["msg_type"]
                handle =topic_info.get("handle", {})    
                callback = self.callback_key_map.get(msg_type)
                if callback:
                    self.suber_dict[topic_key] = rospy.Subscriber(
                        topic_name, msg_type, self.create_callback(callback, topic_key, handle)
                    )
                    print(f"Subscribed to {topic_name} with callback {callback.__name__}")
                else:
                    print(f"No callback found for message type {msg_type}")
    
    # Subscribe to the ROS topics
    def compressedImage_callback(self, msg: CompressedImage, key: str, handle = dict):
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img = cv2.resize(cv_img, resize_wh) if resize_wh else cv_img
        self.obs_buffer_data[key]["data"].append(cv_img)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())

    def Image_callback(self, msg: Image, key: str, handle = dict):
        resize_wh  = handle.get("params", {}).get("resize_wh", None)
        from cv_bridge import CvBridge
        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"Failed to convert image: {e}")
            return None
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv_img = cv2.resize(cv_img, resize_wh) if resize_wh else cv_img
        self.obs_buffer_data[key]["data"].append(cv_img)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())

    def sensorsData_callback(self, msg: sensorsData, key: str, handle = dict):
        # Float64Array ()
        joint = [float(i) for i in msg.joint_data.joint_q[:]]
        slice_value = handle.get("params", {}).get("slice", None)  
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    
    def lejuClawState_callback(self, msg: lejuClawState, key: str, handle = dict):
        # Float64Array ()
        joint = msg.data.position
        slice_value = handle.get("params", {}).get("slice", None)  
        joint = [x / 100 for slc in slice_value for x in joint[slc[0]:slc[1]]] # 注意缩放
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
            
    def robotHandPosition_callback(self, msg: robotHandPosition, key: str, handle = dict):
        joint = [float(num) for num in msg.left_hand_position + msg.right_hand_position]
        # joint = [1 if figure > 30 else 0 for figure in joint]
        joint = [figure / 100 for figure in joint]
        slice_value = handle.get("params", {}).get("slice", None)
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
        
    '''
    #####################
    ### kuavo3.5  
    #####################
    # def recordArmHandPose_callback(self, msg: recordArmHandPose, key: str):
    #     # Float64Array ()
    #     xyz = np.array(msg.left_pose.pos_xyz)
    #     xyzw = np.array(msg.left_pose.quat_xyzw)
    #     rotation = R.from_quat(xyzw)
    #     euler_angles = rotation.as_euler("xyz")
    #     xyzrpy = np.concatenate((xyz, euler_angles))
    #     self.obs_buffer_data[key]["data"].append(xyzrpy)
    #     self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
        
    # def joint_callback(self, msg: JointState, key: str):
    #     # Float64Array ()
    #     joint = (msg.position)
    #     if key == "cmd_joint":
    #         # convert from degree to rad
    #         joint = [i * math.pi / 180 for i in joint]
    #     self.obs_buffer_data[key]["data"].append(joint)
    #     self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    
    # def robotArmQVVD_callback(self, msg: robotArmQVVD, key: str):
    #     # Float64Array ()
    #     joint = msg.q
    #     self.obs_buffer_data[key]["data"].append(joint)
    #     self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    
    # def robot_hand_eff_callback(self, msg: robot_hand_eff, key: str):
    #     # Float32Array (12)
    #     joint = msg.data
    #     self.obs_buffer_data[key]["data"].append(joint)
    #     self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    
    # def robotHandPosition_callback(self, msg: robotHandPosition, key: str):
    #     # Uint8Array (6) + Uint8Array (6)
    #     joint = msg.left_hand_position + msg.right_hand_position
    #     joint= [float(i) for i in joint]
    #     self.obs_buffer_data[key]["data"].append(joint)
    #     self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    '''   
        
    def obs_buffer_is_ready(self, just_img: bool = True):
        if not just_img:
            return all([len(self.obs_buffer_data[key]["data"]) == self.obs_key_map["img"][key]["frequency"] for key in DEFAULT_OBS_KEY_MAP["img"]]) and \
                all([len(self.obs_buffer_data[key]["data"]) == self.obs_key_map["low_dim"][key]["frequency"] for key in DEFAULT_OBS_KEY_MAP["low_dim"]])
        else:
            return all([len(self.obs_buffer_data[key]["data"]) == self.obs_key_map["img"][key]["frequency"] for key in DEFAULT_OBS_KEY_MAP["img"]])

    def stop_subscribers(self):
        for key, suber in self.suber_dict.items():
            suber.unregister()

    def get_lastest_k_img(self, k_image: dict) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'color': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        out = {}
        for i, key in enumerate(self.obs_key_map["img"]):
            k = k_image[key]
            out[i] = {
                "color": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                "timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
            }
        return out

    def get_latest_k_robotstate(self, k_robot: dict) -> dict:
        """
        Return order T,D
        {
            0: {
                'data': (T,D),
                'robot_receive_timestamp': (T,)
            },
            1: ...
        }
        """
        out = {}
        for i, key in enumerate(self.obs_key_map["low_dim"]):
            k = k_robot[key]
            out[key] = {
                "data": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                "robot_receive_timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
            }
        return out
    
    def wait_buffer_ready(self, just_img: bool = False):
        progress_bars = {}
        position = 0
        for key in self.obs_key_map["img"]:
            progress_bars[key] = tqdm(total=self.obs_key_map["img"][key]["frequency"], desc=f"Filling {key}", position=position, leave=True)
            position += 1
        if not just_img:
            for key in self.obs_key_map["low_dim"]:
                progress_bars[key] = tqdm(total=self.obs_key_map["low_dim"][key]["frequency"], desc=f"Filling {key}", position=position, leave=True)
                position += 1


        while not self.obs_buffer_is_ready(just_img):
            for key in self.obs_key_map["img"]:
                current_len = len(self.obs_buffer_data[key]["data"])
                progress_bars[key].n = current_len
                progress_bars[key].refresh()
            
            if not just_img:
                for key in self.obs_key_map["low_dim"]:
                    current_len = len(self.obs_buffer_data[key]["data"])
                    progress_bars[key].n = current_len
                    progress_bars[key].refresh()

        print("All buffers are ready!")
        time.sleep(0.5)
        
class TargetPublisher:
    def __init__(self):
        self.target_pub = rospy.Publisher(DEFAULT_ACT_KEY_MAP["target_left_eef_pose"],Float32MultiArray,queue_size=10)
        self.strict_pub = rospy.Publisher('kuavo_arm_target_poses', armTargetPoses, queue_size=10)
        self.joint_pub = rospy.Publisher(DEFAULT_ACT_KEY_MAP["target_joint"], JointState, queue_size=10)
        self.qiangnao_pub = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=10)
        # self.lejuclaw_pub = rospy.Publisher('/control_robot_leju_claw', controlLejuClaw, queue_size=10)       
        self.pub_cnt = 0
        self.last_pose = np.zeros(6)
        self.l_shift_queue = deque(maxlen=12)
        self.r_shift_queue = deque(maxlen=10)
        
    def publish_target_pose(self, pose: np.ndarray, cur_state: Optional[np.ndarray] = None):
        msg = Float32MultiArray()
        msg.data = pose.tolist()
        self.joint_pub.publish(msg)
        rospy.loginfo("Publishing target pose: %s", msg.data)
        time.sleep(0.1)

    def publish_target_joint(self, joint: np.ndarray, cur_state: Optional[np.ndarray] = None):
        arm_min = [-180, -20, -135, -100, -135, -45, -45, -180, -180, -180, -180, -180, -45, -45]
        arm_max = [30, 135, 135, 100, 135, 45, 45, 180, 180, 180, 180, 180, 45, 45]
        joint = [max(min_val, min(val, max_val)) for val, min_val, max_val in zip(joint, arm_min, arm_max)]
        
        is_strict_duration = True
        if is_strict_duration:  #似乎只有人形有接收接口
            msg = armTargetPoses()
            msg.times = [0.1]  # 时间列表
            msg.values = joint
            self.strict_pub.publish(msg)
            time.sleep(0.1) # 修改
        else:
            msg = JointState()
            msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
            msg.header.stamp = rospy.Time.now() 
            msg.position = joint
            time.sleep(0.1) # 修改
            self.joint_pub.publish(msg)
        return msg
    
    def publish_target_eef_tool(self, left_eef_tool_pose: List, right_eef_tool_pose: List, eef_type: str): # 删除固定运行"dex_hand"
        if eef_type == "dex_hand":
            msg = robotHandPosition()
            close_threshold = 0.5   # 判断预测为合的阈值
            shift_threshold = int(len(self.l_shift_queue) * 0.5)    # 判断真正合的粘滞数量
            if left_eef_tool_pose[0] > close_threshold:
                self.l_shift_queue.append(1)
            else:
                self.l_shift_queue.append(0)

            if sum(self.l_shift_queue) > shift_threshold:
                left_eef_tool_pose = [100] * 6
            else:
                left_eef_tool_pose = [0, 100, 0, 0, 0, 0]

            if right_eef_tool_pose[0] > close_threshold:
                self.r_shift_queue.append(1)
            else:
                self.r_shift_queue.append(0)
            if sum(self.r_shift_queue) > shift_threshold:
                right_eef_tool_pose = [100] * 6
            else:
                right_eef_tool_pose = [0, 100, 0, 0, 0, 0]  
            msg.left_hand_position =  [max(0, int(angle)) for angle in left_eef_tool_pose] # 左手位置
            msg.right_hand_position =  [max(0, int(angle)) for angle in right_eef_tool_pose]  # 右手位置
            msg.header.stamp = rospy.Time.now() 
            log_env.info(f"Publishing target hand joint:\n {msg.left_hand_position}, {msg.right_hand_position}")
            self.qiangnao_pub.publish(msg)
        elif eef_type == 'leju_claw':
            req = controlLejuClawRequest()
            req.data.name = ['left_claw', 'right_claw']
            req.data.position = [left_eef_tool_pose*100, right_eef_tool_pose*100] ###ATT:  注意扩充,如果数据处理处以100
            req.data.velocity = [50, 50]
            req.data.effort = [1.0, 1.0]
            rospy.wait_for_service('/control_robot_leju_claw') 
            control_leju_claw = rospy.ServiceProxy('/control_robot_leju_claw', controlLejuClaw)
            res = control_leju_claw(req)
        else:
            raise ValueError("'EEF_TYPE' must be 'dex_hand' or 'leju_claw'")
        


    #####################
    ### control kuavo3.5 hand joint
    #####################
    # def control_hand(self, left_hand_position: List[float], right_hand_position: List[float]):
    #     hand_positions = controlEndHandRequest()
    #     hand_positions.left_hand_position = left_hand_position
    #     hand_positions.right_hand_position = right_hand_position
    #     try:
    #         rospy.wait_for_service('/control_end_hand')
    #         control_end_hand = rospy.ServiceProxy('/control_end_hand', controlEndHand)
    #         resp = control_end_hand(hand_positions)
    #         if resp.result:
    #             # rospy.loginfo("Gripper control successful")
    #             pass
    #         else:
    #             rospy.logwarn("Gripper control failed")
    #         return resp.result
    #     except rospy.ROSException as e:
    #         rospy.logerr("Service call failed: %s" % e)
    #         return False
    #     except KeyboardInterrupt:
    #         rospy.loginfo("Service call interrupted, shutting down.")
    #         return False

class KuavoEnv:
    def __init__(self,
                frequency:int = 10, 
                n_obs_steps:int = 2, 
                
                video_capture_fps=30,
                robot_publish_rate=100,
                
                # img_buffer_size = 30,
                # robot_state_buffer_size = 100,
                obs_key_map: Optional[Dict[str, Dict[str, str]]] = None,
                video_capture_resolution=(640, 480), # (W,H)
                resize_resolution=(384, 384), # (W,H)
                output_dir: str = "output",
                eef_type: str = None,
                ) -> None:
        assert frequency <= video_capture_fps
        output_dir = Path(output_dir)
        assert output_dir.parent.is_dir()
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.video_capture_fps = video_capture_fps
        self.robot_publish_rate = robot_publish_rate
        # self.img_buffer_size = img_buffer_size
        # self.robot_state_buffer_size = robot_state_buffer_size
        self.video_capture_resolution = video_capture_resolution
        self.resize_resolution = resize_resolution
        
        self.obs_key_map = obs_key_map if obs_key_map is not None else DEFAULT_OBS_KEY_MAP
        
        # self.hand_close_state, self.hand_open_state = HAND_CLOSE_STATE, HAND_OPEN_STATE

        self.obs_buffer = ObsBuffer(obs_key_map=self.obs_key_map)
        self.eef_type = eef_type if eef_type is not None else EEF_TYPE

        self.target_publisher = TargetPublisher()
        self.l_shift_queue = deque(maxlen=10)
        self.r_shift_queue = deque(maxlen=10)
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.obs_buffer.obs_buffer_is_ready()

    def start(self, wait=True):
        print(self.is_ready)

    def stop(self):
        self.obs_buffer.stop_subscribers()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_fake_obs(self):
        '''
        img:(T,H,W,C)
        '''
        import datetime
        first_timestamp = datetime.datetime.now().timestamp()
        second_timestamp = first_timestamp + 0.1
        return {
            "image": np.random.rand(2, 480, 640, 3),
            "agent_pos": np.random.rand(2, 7),
            "timestamp": np.array([first_timestamp, second_timestamp])
        }
        
    # ========= async env API ===========
    def get_obs(self, just_img: bool=False) -> dict:
        # TODO: 把n_obs_step添加到get_obs的参数里
        "observation dict"
        assert self.is_ready
        
        ############################ 
        # get img data according to obs_key_map["img"]
        ############################ 
        # k_image = {
        #     key: min(self.obs_key_map["img"][key]["frequency"], math.ceil((self.n_obs_steps + 3) * (self.obs_key_map["img"][key]["frequency"] / self.frequency))) for key in self.obs_key_map["img"]
        # }
        k_image = {
            key: self.obs_key_map["img"][key]["frequency"] for key in self.obs_key_map["img"]
        }
        self.last_realsense_data = self.obs_buffer.get_lastest_k_img(k_image)

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.min(
            [x["timestamp"][-1] for x in self.last_realsense_data.values()]
        )
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        camera_obs = dict()
        camera_obs_timestamps = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in obs_align_timestamps:
                this_idx = np.argmin(np.abs(this_timestamps - t))
                # is_before_idxs = np.nonzero(this_timestamps < t)[0]
                # this_idx = 0
                # if len(is_before_idxs) > 0:
                #     this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f"img0{camera_idx+1}"] = value["color"][this_idxs]
            camera_obs_timestamps[f"img0{camera_idx+1}"] = this_timestamps[this_idxs]

        ############################ 
        # get low_dim data according to obs_key_map["low_dim"]
        ############################ 
        if not just_img:
            # k_robot = {
            #     key: min(self.obs_key_map["low_dim"][key]["frequency"], math.ceil((self.n_obs_steps + 3) * (self.obs_key_map["low_dim"][key]["frequency"] / self.frequency))) for key in self.obs_key_map["low_dim"]
            # }
            k_robot = {
                key: self.obs_key_map["low_dim"][key]["frequency"] for key in self.obs_key_map["low_dim"]
            }
            last_robot_data = self.obs_buffer.get_latest_k_robotstate(k_robot)
            # align robot obs timestamps
            robot_obs = dict()
            robot_obs_timestamps = dict()
            for robot_state_name, robot_state_data in last_robot_data.items():
                if robot_state_name in self.obs_key_map["low_dim"]:
                    this_timestamps = robot_state_data['robot_receive_timestamp']
                    this_idxs = list()
                    for t in obs_align_timestamps:
                        this_idx = np.argmin(np.abs(this_timestamps - t))
                        # is_before_idxs = np.nonzero(this_timestamps < t)[0]
                        # this_idx = 0
                        # if len(is_before_idxs) > 0:
                        #     this_idx = is_before_idxs[-1]
                        this_idxs.append(this_idx)
                    robot_obs[f"ROBOT_{robot_state_name}"] = robot_state_data['data'][this_idxs]
                    robot_obs_timestamps[f"ROBOT_{robot_state_name}"] = this_timestamps[this_idxs]

    
        ############################ 
        # process raw data to standard obs
        '''
        obs_data = {
            "img01": (T,H,W,C),
            "img02": (T,H,W,C),
            "img...": (T,H,W,C),
            "agent_pos": (T,D),
            "timestamp": (T,)
        }
        '''
        ############################ 
        obs_data = dict(camera_obs)
        if not just_img:
            robot_final_obs = dict()
            half_arm_joint_len = robot_obs["ROBOT_state_joint"].shape[1] // 2
            half_hand_joint_len = robot_obs["ROBOT_state_dex_hand"].shape[1] // 2 if self.eef_type == "dex_hand" else robot_obs["ROBOT_state_gripper"].shape[1]
            eef_tool_state = robot_obs["ROBOT_state_dex_hand"] if self.eef_type == "dex_hand" else robot_obs["ROBOT_state_leju_claw"]
            assert self.n_obs_steps == robot_obs["ROBOT_state_joint"].shape[0] and self.n_obs_steps == eef_tool_state.shape[0]
            
            robot_final_obs["agent_pos"] = np.concatenate((robot_obs["ROBOT_state_joint"][:,:half_arm_joint_len], eef_tool_state[:,:half_hand_joint_len],\
                                                           robot_obs["ROBOT_state_joint"][:,half_arm_joint_len:], eef_tool_state[:,half_hand_joint_len:] ), axis=-1)
            obs_data.update(robot_final_obs)
        else:
            robot_obs = None
            robot_obs_timestamps = None
        obs_data["timestamp"] = obs_align_timestamps
        
        return obs_data, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps

    
    
    def check_predict(self, actions: np.ndarray, current_state: np.ndarray):
        actions = np.array(actions) if not isinstance(actions, np.ndarray) else actions
        current_state = np.array(current_state) if not isinstance(current_state, np.ndarray) else current_state
            
        dualArm_state = np.hstack(np.rad2deg((current_state[:,:7], current_state[:,8:15])))
        dualArm_action = np.hstack(np.rad2deg((actions[:,:7], actions[:,8:15])))
        
        mean_joint_action = np.round((np.mean(dualArm_action, axis=0)), 1)
        mean_joint_state = np.round((np.mean(dualArm_state, axis=0)), 1)
        mean_dualArm_joint_diff = np.round(np.abs(mean_joint_action - mean_joint_state), 1)
        log_env.info(f"\nMean of Current {len(current_state)} steps of states and Submitted {len(actions)} steps of actions: \n {', '.join(map(str, mean_joint_state))}, \n {', '.join(map(str, mean_joint_action))}")
        log_env.warning(f"Joint diff cur_s and predict_a: \n{', '.join(map(str, mean_dualArm_joint_diff))}")
        # assert np.all(mean_dualArm_joint_diff < 30), "The joint diff between current state and predicted action is too large"
        # exit()
        
        sequence_state_and_action = np.vstack((dualArm_state, dualArm_action))
        # round_sequence_state_and_action = np.round(sequence_state_and_action, 1)
        # log_env.warning(f"Sequence of Current {len(current_state)} steps of states and Submitted {len(actions)} steps of actions: \n {round_sequence_state_and_action}")
        
        diffs = np.diff(sequence_state_and_action, axis=0)
        key_joint_diff = np.vstack((diffs[:,0:1]))
        
        # log_env.warning(f"Diff of Current {len(current_state)} steps of states and Submitted {len(actions)} steps of actions: \n {np.round(diffs, 1)}")
        # log_env.warning(f"Key Joint diff cur_s and predict_a: \n{key_joint_diff}")

        # TODO: check the joint diff between current state and predicted action
        # assert np.all(np.abs(diffs) < 30), "The joint diff between current state and predicted action is too large"
        # assert np.all(np.abs(key_joint_diff) < 15), "The joint diff between current state and predicted action is too large"
        # can_move = np.all(mean_dualArm_joint_diff < 30) and \
        #             np.all(np.abs(diffs) < 30) and  \
        #             np.all(np.abs(key_joint_diff) < 30)
        log_env.info(f"diff joint check passed.")
        return True
    
    def exec_fake_actions(
        self,
        actions: np.ndarray,
        current_state: Optional[np.ndarray] = None,
        which_arm: Optional[str] = None,
        eef_type: Optional[str] = None,
        ):
        if which_arm == 'right':
            actions = np.hstack((np.zeros((actions.shape[0], actions.shape[1])), actions))
            current_state = np.hstack((np.zeros((current_state.shape[0], current_state.shape[1])), current_state))
        elif which_arm == 'left':
            actions = np.hstack((actions, np.zeros((actions.shape[0], actions.shape[1]))))
            current_state = np.hstack((current_state, np.zeros((current_state.shape[0], current_state.shape[1]))))
        elif which_arm == 'both':
            pass
        else:
            raise ValueError("which_arm must be 'right', 'left' or 'both'.")
        # check the joint diff between current state and predicted action
            
        if current_state is not None:
            # assert self.check_predict(actions, current_state), "当前状态和预测动作的关节差值过大"
            if self.check_predict(actions, current_state):
                log_env.info("The joint diff between current state and predicted action is ok")
            else:
                log_env.warning("The joint diff between current state and predicted action is too large")
                return
        else:
            log_env.error("current_state is None, can't check predict")
        for i in range(9):
            log_env.info(f"{i} seconds passed")
            time.sleep(0.1)
        return
    
    def exec_actions(
        self,
        actions: np.ndarray,
        current_state: Optional[np.ndarray] = None,
        which_arm: Optional[str] = None,
        eef_type: Optional[str] = None,
    ):  
        if which_arm == 'right':
            actions = np.hstack((np.zeros((actions.shape[0], actions.shape[1])), actions))
            current_state = np.hstack((np.zeros((current_state.shape[0], current_state.shape[1])), current_state))
        elif which_arm == 'left':
            actions = np.hstack((actions, np.zeros((actions.shape[0], actions.shape[1]))))
            current_state = np.hstack((current_state, np.zeros((current_state.shape[0], current_state.shape[1]))))
        elif which_arm == 'both':
            pass
        else:
            raise ValueError("which_arm must be 'right', 'left' or 'both'.")
        
        # check the joint diff between current state and predicted action
        if current_state is not None:
            # assert self.check_predict(actions, current_state), "当前状态和预测动作的关节差值过大"
            if self.check_predict(actions, current_state):
                log_env.info("The joint diff between current state and predicted action is ok")
            else:
                log_env.error("The joint diff between current state and predicted action is too large")
                return
        else:
            log_env.error("current_state is None, can't check predict")
            
        actions = np.array(actions) if not isinstance(actions, np.ndarray) else actions

        ###trick: 如果模型总是预测偏高或者低，可以在这里进行修正
        # actions[:, -3] = actions[:, -3] + 0.1 # test抓顺丰袋
        actions[:, -3] = actions[:, -3] + 0.01  # for Task21_convoyer_pick_cup(手腕向内补偿旋转)
        # actions[:, 4] = actions[:, 4] + 0.12
        
        if eef_type == 'dex_hand':
            assert actions.shape[1] == 16 , "双臂操作训练只取了大拇指第一个关节， 这里在8:13和21:26之间插入其他5*2个手指关节" 
            
            pad_value = 0
            for i in range(8, 13):
                actions = np.insert(actions, i, pad_value, axis=1)
            for i in range(21, 26):
                actions = np.insert(actions, i, pad_value, axis=1)

            for i in range(len(actions)):
                left_arm_joint = actions[i, :7]
                left_hand_joint = actions[i, 7:13]
                right_arm_joint = actions[i, 13:20]
                right_hand_joint = actions[i, 20:26]
            
        elif eef_type == 'leju_claw':
            for i in range(len(actions)):
                left_arm_joint = actions[i, :7]
                left_hand_joint = actions[i, 7:8]
                right_arm_joint = actions[i, 8:15]
                right_hand_joint = actions[i, 15:16]
            

        for i in range(len(actions)):
                
            #####################
            ### control arm joint
            #####################
            arm_joint = np.concatenate((left_arm_joint, right_arm_joint))
            arm_joint = np.rad2deg(arm_joint)
            armJoint_msg = self.target_publisher.publish_target_joint(joint = arm_joint)
            # round_armJoint = ", ".join(map(str, np.round(armJoint_msg.values, 1)))
            # log_env.info(f"Publishing target arm joint:\n {round_armJoint}")
            
            #####################
            ### control gripping
            ##################### 
            self.target_publisher.publish_target_eef_tool(left_eef_tool_pose = left_hand_joint, right_eef_tool_pose = right_hand_joint, eef_type = eef_type)


                
            #####################
            ### control kuavo3.5 hand joint
            #####################
            # if new_actions[i, -1] > 0.5:
            #     self.target_publisher.control_hand(left_hand_position=list(map(int, self.hand_close_state[1:-1].split(", ")))[:6], right_hand_position=[0, 0, 0, 0, 0, 0])
            # else:
            #     self.target_publisher.control_hand(left_hand_position=list(map(int, self.hand_open_state[1:-1].split(", ")))[:6], right_hand_position=[0, 0, 0, 0, 0, 0])
            # time.sleep(1)

    def record_video(self, output_video_path, width=640, height=480, fps=30, hstack=True, stop_event=None,title="IL"):
        obs, _, _, _, _ = self.get_obs(just_img=True)
        cam_num = 0
        for k in obs.keys():
            if "img" in k:
                cam_num += 1
        # fourcc = cv2.VideoWriter_fourcc(*'H264')  # 使用 H264 编码器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use MJPG if H264 is not working
        if hstack:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (cam_num * width, height))
        else:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height * cam_num))
            
        while not stop_event.is_set():
            obs, _, _, _, _ = self.get_obs(just_img=True)
            imgs = []
            for k, v in obs.items():
                if "img" in k:
                    img = obs[k][-1]
                    img = cv2.resize(img, (width, height))
                    img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    imgs.append(img01_bgr)
                    
            concatenated_img = np.hstack(imgs) 
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(concatenated_img, title, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
            out.write(concatenated_img)
            time.sleep(1/fps)
        out.release()



import logging
import colorama

colorama.init(autoreset=True)

# 颜色映射
COLORS = {
    "DEBUG": colorama.Fore.CYAN,
    "INFO": colorama.Fore.GREEN,
    "WARNING": colorama.Fore.YELLOW,
    "ERROR": colorama.Fore.RED,
    "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT,
}

# 自定义日志格式
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, "")
        return f"{color}{super().format(record)}{colorama.Style.RESET_ALL}"

# 创建分类 Logger
log_model = logging.getLogger("model")  # 网络日志
log_env = logging.getLogger("robot")  # 机器人日志

# 统一配置 Handler
def setup_logger(logger, level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(f"%(asctime)s - [%(name)s] - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)

# 配置不同 Logger
setup_logger(log_model, logging.INFO)
setup_logger(log_env, logging.DEBUG)




if __name__ == "__main__":
    
    try:
        import threading, signal, sys
        rospy.init_node("env_teset")
        task_name = "env_test"
        with KuavoEnv(
        frequency=10,
        n_obs_steps=2,
        video_capture_fps=30,
        robot_publish_rate=500,
        img_buffer_size=30,
        robot_state_buffer_size=100,
        video_capture_resolution=(640, 480),
        ) as env:
            is_just_img = False
            ## ========= prepare obs ==========
            print("waiting for the obs buffer to be ready ......")
            env.obs_buffer.wait_buffer_ready(is_just_img)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            import os
            current_file_path = os.path.abspath(__file__)

            test_rec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "test_rec")
            os.makedirs(test_rec_path, exist_ok=True)
            
            output_video_path = os.path.join(test_rec_path, f'./{current_time}_{task_name}_x1.mp4')
            stop_event = threading.Event()  
            record_title = f"{current_time}: Task Name: {task_name} at {current_time}"
            video_thread = threading.Thread(target=env.record_video, args=(output_video_path, 640, 480, 30, True,stop_event,record_title))
            video_thread.start()
            
            import imageio
            writer = imageio.get_writer(output_video_path, fps=10, codec="libx264")
            
            
            def handle_exit_signal(signum, frame, stop_event):
                print(f"Signal received, saving video to {output_video_path} and cleaning up...")
                stop_event.set()  # 停止视频录制
                cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
                sys.exit(0)  # 退出程序
            
            # 注册信号处理器
            signal.signal(signal.SIGINT, lambda sig, frame: handle_exit_signal(sig, frame, stop_event))
            signal.signal(signal.SIGQUIT, lambda sig, frame: handle_exit_signal(sig, frame, stop_event))
            while True:
                # ========= human control loop ==========
                print("skip Human in control!")
                
                # ========== policy control loop ==============
                try:
                    import matplotlib.pyplot as plt
                    while True:
                        # =============
                        # get obs
                        # =============
                        obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs(is_just_img)
                        print('##################################################################')
                        print(f"{camera_obs_timestamps['img01'][0]:.10f}, {camera_obs_timestamps['img01'][1]:.10f}, \n"
                            f"{camera_obs_timestamps['img02'][0]:.10f}, {camera_obs_timestamps['img02'][1]:.10f}, \n"
                            # f"{camera_obs_timestamps['img03'][0]:.10f}, {camera_obs_timestamps['img03'][1]:.10f}, \n"
                            # f"{camera_obs_timestamps['img04'][0]:.10f}, {camera_obs_timestamps['img04'][1]:.10f}\n"
                        )
                        if not is_just_img:
                            print(f"{robot_obs_timestamps['ROBOT_state_joint'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_joint'][1]:.10f}, \n"
                                f"{robot_obs_timestamps['ROBOT_state_gripper'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_gripper'][1]:.10f}\n"
                                )
                        print('##################################################################')
                        import numpy as np
                        
                        # =============
                        # show sense img
                        # =============
                        imgs = []
                        for k, v in obs.items():
                            if "img" in k:
                                img = obs[k][-1]
                                img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                imgs.append(img01_bgr)
                                
                        concatenated_img = np.hstack(imgs)  # 横向拼接图像
                        if os.getenv("DISPLAY") is not None:
                            try:
                                cv2.imshow('Image Stream', concatenated_img)
                            except Exception as e:
                                pass
                        else:
                            print("No display found, skipping image display.")
                        from PIL import Image, ImageDraw, ImageFont
                        concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(concatenated_img)
                        draw = ImageDraw.Draw(pil_img)
                        font = ImageFont.load_default()  # 使用默认字体，可以选择其他字体
                        title = record_title
                        draw.text((50, 50), title, font=font, fill="green")
                        
                        # 转回Numpy数组
                        concatenated_img_with_text = np.array(pil_img)
                        
                        # 写入视频
                        writer.append_data(concatenated_img_with_text)
                        

                        # 处理按键事件，按'q'退出
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.close()
                    exit(0)
                print("Stopped.")

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down node...")
        rospy.signal_shutdown("Manual shutdown")
        
