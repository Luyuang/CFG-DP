import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import math
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
from typing import List, Tuple, Dict
from collections import defaultdict
import importlib

def _adjust_single_arm_pose(pose_data: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Adjust pose angles to avoid 2π jumps for a single arm.
    
    Args:
        pose_data: Input pose data array of shape (N, M)
        start_idx: Start column index for RPY values
        end_idx: End column index for RPY values
        
    Returns:
        Adjusted pose data with continuous angles
    """
    pose = pose_data[:, start_idx:end_idx].copy()
    pose[:, 3:6] = np.unwrap(pose[:, 3:6], axis=0)  # Unwrap RPY angles
    return pose

def adjust_pose_rpy(lr_pose: np.ndarray) -> np.ndarray:
    """Adjust left and right arm poses to avoid 2π jumps in RPY angles.
    
    Args:
        lr_pose: Combined pose data array of shape (N, 12)
        
    Returns:
        Adjusted pose data with continuous angles
    """
    lr_pose = np.asarray(lr_pose)
    left_pose = _adjust_single_arm_pose(lr_pose, 0, 6)
    right_pose = _adjust_single_arm_pose(lr_pose, 6, 12)
    return np.concatenate([left_pose, right_pose], axis=1)

class RosbagReader:
    def __init__(self, 
                 bag_path: str,
                 cfg: Dict,
                 save_plt_folder: str= None, 
                 save_lastPic_folder: str = None, 
                 raw_video_folder: str = None,
                 ):
        self.bag_path = bag_path
        self.save_plt_folder = save_plt_folder
        self.save_lastPic_folder = save_lastPic_folder
        self.raw_video_folder = raw_video_folder
        
        self.base_name = os.path.splitext(os.path.basename(bag_path))[0]
        self.video_frames = {}
        self.config = cfg
        self.map = self._generate_processor_map()
     
    def _get_processor(self, topic):
        # 从 config 中获取对应的话题处理函数
        if topic in self.map:
            return self.map[topic]
        else:
            raise ValueError(f"No processor defined for topic: {topic}")
        
    def _generate_processor_map(self):
        processor_map = {}
        # 遍历 "img" 和 "low_dim" 配置，创建映射
        for key in ["img", "low_dim"]:
            state = self.config["DEFAULT_OBS_KEY_MAP"][key]
            for sensor, sub_value in state.items():
                topic = sub_value['topic']
                processor_str = sub_value['handle']['processor']
                params = sub_value['handle'].get('params', {})  # 获取参数
                
                processor_func = self._import_processor(processor_str)
                processor_map[topic] = self._create_processor_lambda(processor_func, params)
                
        return processor_map

    def _import_processor(self, processor_str, *args, **kwargs):
        """
        根据函数路径字符串动态导入函数
        """
        parts = processor_str.split('.')
        module_path = '.'.join(parts[:-2])
        class_name = parts[-2]
        method_name = parts[-1]
        module = importlib.import_module(module_path)
        class_ = getattr(module, class_name)
        method = getattr(class_, method_name)
        return method

    def _create_processor_lambda(self, processor_func, params):
        """
        创建一个闭包处理函数，避免 lambda 问题
        """
        def processor_lambda(msg):
            return processor_func(msg, **params)
        return processor_lambda
        
    def _collect_bag_data(self) -> Dict:
        data = defaultdict(list)
        topics = {
            **{v['topic']: k for k, v in self.config["DEFAULT_OBS_KEY_MAP"]['img'].items()},
            **{v['topic']: k for k, v in self.config["DEFAULT_OBS_KEY_MAP"]['low_dim'].items()}
        }
        with rosbag.Bag(self.bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=list(topics.keys())):
                key = topics[topic]
                processor = self._get_processor(topic)
                ret_dict = processor(msg)
                correct_timestamp = t.to_sec()  # 如果没有 header.stamp，使用bag的时间戳
                # 由于偶尔系统时间不对导致header.stamp是远古时间，所以需要添加检查header.stamp是否是合理的时间戳          
                if hasattr(msg, 'header') and hasattr(msg.header, 'stamp') and abs(msg.header.stamp.to_sec() - correct_timestamp) < 10:
                    correct_timestamp = msg.header.stamp.to_sec()    
                data[f"{key}_time_stamp"].append(correct_timestamp)
                data[key].append(ret_dict["data"])
        
        if self.raw_video_folder:
            for img in self.config["DEFAULT_OBS_KEY_MAP"]['img'].keys():
                self.video_frames[img] = data[img][self.config["SAMPLE_DROP"]:-self.config["SAMPLE_DROP"]]
                os.makedirs(self.raw_video_folder / self.base_name, exist_ok=True)
                self._save_video(output_video=self.raw_video_folder / self.base_name / f"{img}.mp4", img_array=self.video_frames[img])

        if 'state_hand' not in data:
            data['state_hand'] = data['cmd_hand'][:]
            data['state_hand_time_stamp'] = data['cmd_hand_time_stamp'][:]
            shift = 3
            
            if len(data['cmd_hand']) > shift:
                data['state_hand'][:shift] = [data['cmd_hand'][0]] * shift
                data['state_hand'][shift:] = data['cmd_hand'][:-shift]
            else:
                data['state_hand'] = [data['cmd_hand'][0]] * len(data['cmd_hand'])

        return data

    def _find_nearest_index(self, time_stamps, target_time) -> int:
        """Find the index of the nearest timestamp to target_time"""
        time_array = np.array([t for t in time_stamps])
        return np.argmin(np.abs(time_array - target_time))
    
    def _integrate_hand_data(self, data: np.ndarray, hand_data: np.ndarray) -> np.ndarray:
        """
        Integrate hand data into data array at specified positions based on array length
        Args:
            data: Joint or EEF pose data
            hand_data: Hand state data
        Returns:
            np.ndarray: Integrated data array
        """
        result = []
        for arr, hand in zip(data, hand_data):
            arr_list = list(arr)
            hand = list(hand)
            half_len = len(arr_list) // 2
            half_hand_len = len(hand) // 2
            left_dex = arr_list[:half_len] + hand[:half_hand_len]
            right_dex = arr_list[half_len:] + hand[half_hand_len:]
            result.append(left_dex + right_dex)
        return np.array(result, dtype=np.float32)
    
    def _align_data(self, data: Dict, config: Dict) -> Dict:
        """Align all data based on image timestamps"""
        jump = config["CAM_HZ"] // config["TRAIN_HZ"]
        img_stamps = data[f'{config["MAIN_TIMELINE"]}_time_stamp'][config["SAMPLE_DROP"]:-config["SAMPLE_DROP"]][::jump]
        main_imgs = np.array(data[config["MAIN_TIMELINE"]][config["SAMPLE_DROP"]:-config["SAMPLE_DROP"]][::jump])
        
        aligned_data = defaultdict(list)
        
        for stamp in img_stamps:
            stamp_sec = stamp
            for key in data:
                if '_time_stamp' in key:
                    continue
                idx = self._find_nearest_index(data[f"{key}_time_stamp"], stamp_sec)
                aligned_data[key].append(data[key][idx])
            aligned_data['timestamp'].append(stamp_sec)

        from kuavo_utils.pcd_util import farthest_point_sampling, preprocess_point_cloud
        for k, v in aligned_data.items():
            if 'pcd' in k:
                final_pcd = []
                for pcd in v:
                    this_frame_pcd = preprocess_point_cloud(pcd)
                    final_pcd.append(this_frame_pcd)
                aligned_data[k] = np.array(final_pcd)

        for key in aligned_data:
            if 'img' in key:
                aligned_data[key] = np.array(aligned_data[key], dtype=np.uint8)
            else:
                aligned_data[key] = np.array(aligned_data[key], dtype=np.float32)
            
        aligned_data['relative_cmd_joint'] = aligned_data['cmd_joint'] - aligned_data['cmd_joint'][0]
        aligned_data['relative_state_joint'] = aligned_data['state_joint'] - aligned_data['state_joint'][0]
        
        assert len(aligned_data['cmd_joint']) == len(aligned_data['cmd_hand']) == len(aligned_data['state_joint']) == len(aligned_data['state_hand']), 'data lack'

        for key in ['cmd', 'state', 'relative_cmd', 'relative_state']:
            hand_key = key.split('_')[-1]
            for k in ['joint', 'eef'] if 'eef' in self.config["DEFAULT_OBS_KEY_MAP"]['low_dim'].keys() else ['joint']:
                aligned_data[f'{key}_{k}_with_hand'] = self._integrate_hand_data(
                    aligned_data[f'{key}_{k}'], 
                    aligned_data[f'{hand_key}_hand'],
                )

        cmd_joint = aligned_data['cmd_joint_with_hand']
        state_joint = aligned_data['state_joint_with_hand']
        relative_cmd_joint = aligned_data['relative_cmd_joint_with_hand']
        relative_state_joint = aligned_data['relative_state_joint_with_hand']
        cmd_hand = aligned_data['cmd_hand']
        half_len = len(cmd_joint[0]) // 2
        half_hand_len = len(cmd_hand[0]) // 2
        
        delta_cmd_joint_with_hand = np.zeros_like(cmd_joint[1:])
        delta_cmd_joint_with_hand = cmd_joint[1:] - state_joint[:-1]
        delta_relative_cmd_joint_with_hand = relative_cmd_joint[1:] - relative_state_joint[:-1]
        
        delta_cmd_joint_with_hand[:, half_len-half_hand_len:half_len] = cmd_joint[1:, half_len-half_hand_len:half_len]
        delta_cmd_joint_with_hand[:, 2* half_len - half_hand_len:] = cmd_joint[1:, 2* half_len - half_hand_len:]
        
        delta_relative_cmd_joint_with_hand[:, half_len-half_hand_len:half_len] = relative_cmd_joint[1:, half_len-half_hand_len:half_len]
        delta_relative_cmd_joint_with_hand[:, 2* half_len - half_hand_len:] = relative_cmd_joint[1:, 2* half_len - half_hand_len:]
        
        if 'eef' in self.config["DEFAULT_OBS_KEY_MAP"]['low_dim'].keys():
            cmd_eef = aligned_data['cmd_eef_with_hand']
            cmd_hand = aligned_data['cmd_hand']
            half_len = len(cmd_eef[0]) // 2
            half_hand_len = len(cmd_hand[0]) // 2
            
            delta_cmd_pose_with_hand = np.zeros_like(cmd_eef[1:])
            delta_cmd_pose_with_hand = cmd_eef[1:] - cmd_eef[:-1]
            
            delta_cmd_pose_with_hand[:, half_len-half_hand_len:half_len] = cmd_eef[1:, half_len-half_hand_len:half_len]   
            delta_cmd_pose_with_hand[:, 2* half_len - half_hand_len:] = cmd_eef[1:, 2* half_len - half_hand_len:] 

        aligned_data[config["MAIN_TIMELINE"]] = main_imgs[:]

        for key in aligned_data:
            aligned_data[key] = aligned_data[key][1:]

        result_data = aligned_data
        if 'eef' in self.config["DEFAULT_OBS_KEY_MAP"]['low_dim'].keys():
            result_data['delta_cmd_eef_pose_with_hand'] = np.array(delta_cmd_pose_with_hand, dtype=np.float32)
        aligned_data['delta_cmd_joint_with_hand'] = np.array(delta_cmd_joint_with_hand, dtype=np.float32)
        aligned_data['delta_relative_cmd_joint_with_hand'] = np.array(delta_relative_cmd_joint_with_hand, dtype=np.float32)
        
        print("Data shapes after alignment:")
        for key, value in result_data.items():
            print(f"{key}: {value.shape}")
        
        return result_data

    def _plot_results(self, result_data: Dict):
        """Plot and save comparison graphs"""
        img_keys = list(self.config["DEFAULT_OBS_KEY_MAP"]["img"].keys())
        img_strips = []

        show_img_num = min(30, len(result_data[img_keys[0]]))
        for img_key in img_keys:
            img_strips.append(np.concatenate(np.array(result_data[img_key][::len(result_data[img_key])//show_img_num]), axis=1))
    
        for i, img_strip in enumerate(img_strips):
            if img_strip.shape[0] != img_strips[0].shape[0]:
                img_strips[i] = cv2.resize(img_strip, (img_strips[0].shape[1], img_strips[0].shape[0]))
      
        img_strip_combined = np.vstack(img_strips)

        ACTION_DIM_LABELS = self.config['ACTION_DIM_LABELS']
        JOINT_DIM_LABELS = self.config['JOINT_DIM_LABELS']
        
        figure_layout = [
            ['image'] * len(JOINT_DIM_LABELS[:7]),
            JOINT_DIM_LABELS[:7],
            JOINT_DIM_LABELS[7:13] + ['extra1'],
            JOINT_DIM_LABELS[13:-6] ,
            JOINT_DIM_LABELS[-6:]+ ['extra2'],
        ]
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([40, 18])
        fig.suptitle(self.base_name, fontsize=16)
        
        for key in self.config['DEFAULT_OBS_KEY_MAP']['low_dim']:
            if 'eef' in key:
                for action_dim, action_label in enumerate(ACTION_DIM_LABELS[:7]):
                    axs[action_label].plot(result_data['cmd_eef_with_hand'][:, action_dim], label='cmd_eef_with_hand', alpha=0.5, zorder=1)
                    axs[action_label].plot(result_data['state_eef_with_hand'][:, action_dim], label='state_eef_with_hand', alpha=0.5, zorder=1)
                    axs[action_label].plot(result_data['delta_cmd_eef_pose_with_hand'][:, action_dim], label='delta_cmd_eef_pose_with_hand', alpha=0.5, zorder=1)
                    axs[action_label].set_title(f"motor {action_dim+1} state")
                    axs[action_label].set_xlabel('Time in one episode')
                    axs[action_label].legend()
            
        for joint_dim, joint_label in enumerate(JOINT_DIM_LABELS):
            axs[joint_label].plot(result_data['cmd_joint_with_hand'][:, joint_dim], label='cmd_joint_with_hand', alpha=0.5, zorder=1)
            axs[joint_label].plot(result_data['state_joint_with_hand'][:, joint_dim], label='state_joint_with_hand', alpha=0.5, zorder=1)
            
            axs[joint_label].plot(result_data['delta_cmd_joint_with_hand'][:, joint_dim], label='delta_cmd_joint', alpha=0.5, zorder=1)
            axs[joint_label].plot(result_data['relative_cmd_joint_with_hand'][:, joint_dim], label='relative_cmd_joint', alpha=0.5, zorder=1)
            axs[joint_label].plot(result_data['relative_state_joint_with_hand'][:, joint_dim], label='relative_state_joint', alpha=0.5, zorder=1)
            
            axs[joint_label].set_title(f"joint {joint_dim+1} state")
            axs[joint_label].set_xlabel('Time in one episode')
            axs[joint_label].legend()
            
        axs['image'].imshow(img_strip_combined)

        plt.tight_layout()
        if self.save_plt_folder:
            os.makedirs(self.save_plt_folder, exist_ok=True)
            plt.savefig(
                f"{self.save_plt_folder}/{self.base_name}.jpg",
                dpi=80,
                bbox_inches='tight'
            )
        
        imgs = [Image.fromarray(result_data[k][-1], 'RGB') for k in img_keys]
        imgs = np.concatenate(imgs, axis=1)
        imgs = Image.fromarray(imgs, 'RGB')
        
        if self.save_lastPic_folder:
            os.makedirs(self.save_lastPic_folder, exist_ok=True)
            imgs.save(f"{self.save_lastPic_folder}/{self.base_name}.jpg", quality=50)  # quality 范围 1-95

    def _save_video(self, output_video, img_array, format='mp4', fps=30, img_size=(384, 384)):
        output_video = output_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(output_video.as_posix(), fourcc, fps, tuple(img_size))

        for img in img_array:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_img)
        video_writer.release()
        
    def process_bag(self, config: Dict) -> Tuple:
        """Main processing function"""
        # Collect data from bag
        raw_data = self._collect_bag_data()
        
        # Validate data
        if not raw_data['cmd_joint'] or not raw_data['state_joint']:
            raise ValueError("ROS bag file contains empty data for at least one topic.")
        if len(raw_data['cmd_joint']) < 100 or len(raw_data['state_joint']) < 100:
            raise ValueError("ROS bag file data count is too small (less than 100 data points).")
        
        # Process and align data
        result_data = self._align_data(raw_data, config)
        
        # Plot results
        if self.save_plt_folder and self.save_lastPic_folder:
            self._plot_results(result_data)
        
        return result_data  # a dict about all data