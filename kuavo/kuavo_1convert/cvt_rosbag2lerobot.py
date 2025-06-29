"""
Script to convert Kuavo rosbag data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
nls目标路径：/home/leju-ali/luyuang/nls/rosbag  repo -id 
kuavo_il路径下：python kuavo/kuavo_1convert/cvt_rosbag2lerobot.py --raw_dir /home/leju-ali/luyuang/nls/rosbag --process_version v0
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import sys
import os

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import torch
import tqdm
import json

from common.kuavo_dataset import (
    KuavoRosbagReader,
    DEFAULT_JOINT_NAMES_LIST,
    DEFAULT_LEG_JOINT_NAMES,
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_HEAD_JOINT_NAMES,
    DEFAULT_CAMERA_NAMES,
    DEFAULT_JOINT_NAMES,
    DEFAULT_LEJUCLAW_JOINT_NAMES,
    DEFAULT_DEXHAND_JOINT_NAMES,
    
    SLICE_ROBOT,
    SLICE_DEX,
    SLICE_CLAW,
    USE_LEJU_CLAW,
    USE_QIANGNAO,
    TRAIN_HZ,
    CONTROL_HAND_SIDE ,
    IS_BINARY,
    DELTA_ACTION,
    RELATIVE_START,
    ONLY_HALF_UP_BODY,
    )


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_cameras(bag_data: dict) -> list[str]:
    """
    /cam_l/color/image_raw/compressed                    : sensor_msgs/CompressedImage                
    /cam_r/color/image_raw/compressed                    : sensor_msgs/CompressedImage                
    /zedm/zed_node/left/image_rect_color/compressed      : sensor_msgs/CompressedImage                
    /zedm/zed_node/right/image_rect_color/compressed     : sensor_msgs/CompressedImage 
    """
    cameras = []

    for k in DEFAULT_CAMERA_NAMES:
        cameras.append(k)
    return cameras

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
) -> LeRobotDataset:
    
    # 根据config的参数决定是否为半身和末端的关节类型
    motors = DEFAULT_JOINT_NAMES_LIST
    # TODO: auto detect cameras
    cameras = DEFAULT_CAMERA_NAMES

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors) + 1,),
            "names": {
                "motors": motors + ["step"]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors) ,),
            "names": {
                "motors": motors
            }
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        if 'depth' in cam:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (480, 640),
                "names": [
                    "height",
                    "width",
                ],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, 480, 640),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )

def load_raw_images_per_camera(bag_data: dict) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in get_cameras(bag_data):
        imgs_per_cam[camera] = np.array([msg['data'] for msg in bag_data[camera]])
    
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    
    bag_reader = KuavoRosbagReader()
    bag_data = bag_reader.process_rosbag(ep_path)
    
    state = np.array([msg['data'] for msg in bag_data['observation.state']], dtype=np.float32)
    action = np.array([msg['data'] for msg in bag_data['action']], dtype=np.float32)
    claw_state = np.array([msg['data'] for msg in bag_data['observation.claw']], dtype=np.float64)
    claw_action= np.array([msg['data'] for msg in bag_data['action.claw']], dtype=np.float64)
    qiangnao_state = np.array([msg['data'] for msg in bag_data['observation.qiangnao']], dtype=np.float64)
    qiangnao_action= np.array([msg['data'] for msg in bag_data['action.qiangnao']], dtype=np.float64)
    step = np.array([msg['data'] for msg in bag_data['observation.step']], dtype=np.float32)

    velocity = None
    effort = None
    
    imgs_per_cam = load_raw_images_per_camera(bag_data)
    
    return imgs_per_cam, state, action, velocity, effort ,claw_state ,claw_action,qiangnao_state,qiangnao_action,step


def diagnose_frame_data(data):
    for k, v in data.items():
        print(f"Field: {k}")
        print(f"  Shape    : {v.shape}")
        print(f"  Dtype    : {v.dtype}")
        print(f"  Type     : {type(v).__name__}")
        print("-" * 40)


def populate_dataset(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(bag_files))
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        from termcolor import colored
        print(colored(f"Processing {ep_path}", "yellow", attrs=["bold"]))
        # 默认读取所有的数据如果话题不存在相应的数值应该是一个空的数据
        imgs_per_cam, state, action, velocity, effort ,claw_state, claw_action,qiangnao_state,qiangnao_action,step = load_raw_episode_data(ep_path) #新增读取数据step
        
        # 对手部进行二值化处理
        if IS_BINARY:
            qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
            qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
            claw_state = np.where(claw_state > 50, 1, 0)
            claw_action = np.where(claw_action > 50, 1, 0)
        else:
            # 进行数据归一化处理
            claw_state = claw_state / 100
            claw_action = claw_action / 100
            qiangnao_state = qiangnao_state / 100
            qiangnao_action = qiangnao_action / 100
            
        ########################
        # delta 处理
        ########################
        # =====================
        # 为了解决零点问题，将每帧与第一帧相减
        if RELATIVE_START:
            # 每个state, action与他们的第一帧相减
            state = state - state[0]
            action = action - action[0]
            
        # ===只处理delta action
        if DELTA_ACTION: #目前false
            delta_action = action[1:] - state[:-1]
            trim = lambda x: x[1:] if (x is not None) and (len(x) > 0) else x
            state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action, step = \
                map(
                    trim, 
                    [state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action, step]
                    )
            for camera, img_array in imgs_per_cam.items():
                imgs_per_cam[camera] = img_array[1:]
            action = delta_action
        
        num_frames = state.shape[0]
        for i in range(num_frames):
            # 将 observation.step 嵌入到 observation.state
            step_value = step[i]  # 当前帧的 step 值
            extended_state = np.concatenate((state[i], [step_value]), axis=0)  # 将 step 添加到 state 的末尾

            print(f"State shape: {state[i].shape}")
            print(f"Step value: {step_value}")
            print(f"Extended state shape: {extended_state.shape}")

            if ONLY_HALF_UP_BODY:   # True
                if USE_LEJU_CLAW:   # False
                    # 使用lejuclaw进行上半身关节数据转换
                    if CONTROL_HAND_SIDE == "left" or CONTROL_HAND_SIDE == "both":
                        output_state = extended_state[SLICE_ROBOT[0][0]:SLICE_ROBOT[0][-1]]
                        output_state = np.concatenate((output_state, claw_state[i, SLICE_CLAW[0][0]:SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                        output_action = action[i, SLICE_ROBOT[0][0]:SLICE_ROBOT[0][-1]]
                        output_action = np.concatenate((output_action, claw_action[i, SLICE_CLAW[0][0]:SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                    if CONTROL_HAND_SIDE == "right" or CONTROL_HAND_SIDE == "both":
                        if CONTROL_HAND_SIDE == "both":
                            output_state = np.concatenate((output_state, extended_state[SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]), axis=0)
                            output_state = np.concatenate((output_state, claw_state[i, SLICE_CLAW[1][0]:SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                            output_action = np.concatenate((output_action, action[i, SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]), axis=0)
                            output_action = np.concatenate((output_action, claw_action[i, SLICE_CLAW[1][0]:SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                        else:
                            output_state = extended_state[SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]
                            output_state = np.concatenate((output_state, claw_state[i, SLICE_CLAW[1][0]:SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                            output_action = action[i, SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]
                            output_action = np.concatenate((output_action, claw_action[i, SLICE_CLAW[1][0]:SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)

                elif USE_QIANGNAO:  #True
                    # 类型: kuavo_sdk/robotHandPosition
                    # left_hand_position (list of float): 左手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                    # right_hand_position (list of float): 右手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                    # 构造qiangnao类型的output_state的数据结构的长度应该为26
                    if CONTROL_HAND_SIDE == "left" or CONTROL_HAND_SIDE == "both":
                        output_state = extended_state[SLICE_ROBOT[0][0]:SLICE_ROBOT[0][-1]]
                        output_state = np.concatenate((output_state, qiangnao_state[i, SLICE_DEX[0][0]:SLICE_DEX[0][-1]].astype(np.float32)), axis=0)
                        
                        output_action = action[i, SLICE_ROBOT[0][0]:SLICE_ROBOT[0][-1]]
                        output_action = np.concatenate((output_action, qiangnao_action[i, SLICE_DEX[0][0]:SLICE_DEX[0][-1]].astype(np.float32)), axis=0)
                    if CONTROL_HAND_SIDE == "right" or CONTROL_HAND_SIDE == "both":
                        if CONTROL_HAND_SIDE == "both":
                            output_state = np.concatenate((output_state, extended_state[SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]), axis=0)
                            output_state = np.concatenate((output_state, qiangnao_state[i, SLICE_DEX[1][0]:SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                            output_action = np.concatenate((output_action, action[i, SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]), axis=0)
                            output_action = np.concatenate((output_action, qiangnao_action[i, SLICE_DEX[1][0]:SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                        else:   # Selected
                            output_state = extended_state[SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]
                            output_state = np.concatenate((output_state, qiangnao_state[i, SLICE_DEX[1][0]:SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                            output_state = np.concatenate((output_state, [step_value]), axis=0) #step接在qiangnao后
                            output_action = action[i, SLICE_ROBOT[1][0]:SLICE_ROBOT[1][-1]]
                            output_action = np.concatenate((output_action, qiangnao_action[i, SLICE_DEX[1][0]:SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                    # output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)
   
            else:   
                if USE_LEJU_CLAW:
                    # 使用lejuclaw进行全身关节数据转换
                    # 原始的数据是28个关节的数据对应原始的state和action数据的长度为28
                    # 数据顺序:
                    # 前 12 个数据为下肢电机数据:
                    #     0~5 为左下肢数据 (l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll)
                    #     6~11 为右下肢数据 (r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll)
                    # 接着 14 个数据为手臂电机数据:
                    #     12~18 左臂电机数据 ("l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll")
                    #     19~25 为右臂电机数据 ("r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll")
                    # 最后 2 个为头部电机数据: head_yaw 和 head_pitch
                    
                    # TODO：构造目标切片
                    output_state = extended_state[0:19]
                    output_state = np.insert(output_state, 19, claw_state[i, 0].astype(np.float32))
                    output_state = np.concatenate((output_state, extended_state[19:26]), axis=0)
                    output_state = np.insert(output_state, 19, claw_state[i, 1].astype(np.float32))
                    output_state = np.concatenate((output_state, extended_state[26:28]), axis=0)

                    output_action = action[i, 0:19]
                    output_action = np.insert(output_action, 19, claw_action[i, 0].astype(np.float32))
                    output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                    output_action = np.insert(output_action, 19, claw_action[i, 1].astype(np.float32))
                    output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)

                elif USE_QIANGNAO:
                    output_state = extended_state[0:19] #提取前19个元素
                    output_state = np.concatenate((output_state, qiangnao_state[i, 0:6].astype(np.float32)), axis=0)
                    output_state = np.concatenate((output_state, extended_state[19:26]), axis=0)
                    output_state = np.concatenate((output_state, qiangnao_state[i, 6:12].astype(np.float32)), axis=0)
                    output_state = np.concatenate((output_state, extended_state[26:28]), axis=0)

                    output_action = action[i, 0:19]
                    output_action = np.concatenate((output_action, qiangnao_action[i, 0:6].astype(np.float32)),axis=0)
                    output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                    output_action = np.concatenate((output_action, qiangnao_action[i, 6:12].astype(np.float32)), axis=0)
                    output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)
            # 打印最终的 state
            print(f"Final processed state for frame {i}: {output_state}")    
            print(f"Output state shape: {output_state.shape}")

            frame = {
                "observation.state": torch.from_numpy(output_state).type(torch.float32),
                "action": torch.from_numpy(output_action).type(torch.float32),
            }
            
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]   
            
            # diagnose_frame_data(frame)
            dataset.add_frame(frame)
        dataset.save_episode(task=task)

    return dataset
            


def port_kuavo_rosbag(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    n: int | None = None,
):
    # Download raw data if not exists
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)
        
    bag_reader = KuavoRosbagReader() 
    bag_files = bag_reader.list_bag_files(raw_dir)
    
    if isinstance(n, int) and n > 0:
        # random sample num_of_bag files
        select_idx = np.random.choice(len(bag_files), n, replace=False)
        bag_files = [bag_files[i] for i in select_idx]
    
    dataset = create_empty_dataset( 
        repo_id,
        robot_type="kuavo4pro",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
        root = root,
    )
    dataset = populate_dataset(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kuavo ROSbag to LERobot Dataset Converter")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw ROS bag directory")
    parser.add_argument(
        "-n", "--num_of_bag", 
        type=int, 
        default=None,  
        help="The number of bag files to process, e.g. 3"
    )
    parser.add_argument(
        '-v', '--process_version',
        default = 'v0',
        type = str,
        help="process version"
    )
    args = parser.parse_args()
    n = args.num_of_bag
    raw_dir=args.raw_dir
    version = args.process_version
    
    task_name = os.path.basename(raw_dir)
    repo_id = f'lerobot/{task_name}'
    lerobot_dir = os.path.join(raw_dir,"../",version,"lerobot")
    if os.path.exists(lerobot_dir):
        shutil.rmtree(lerobot_dir)

    half_arm = len(DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(DEFAULT_DEXHAND_JOINT_NAMES) // 2
    UP_START_INDEX = 12
    if ONLY_HALF_UP_BODY:
        if USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = DEFAULT_ARM_JOINT_NAMES[:half_arm] + DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                    + DEFAULT_ARM_JOINT_NAMES[half_arm:] + DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            arm_slice = [
                (SLICE_ROBOT[0][0] - UP_START_INDEX, SLICE_ROBOT[0][-1] - UP_START_INDEX),(SLICE_CLAW[0][0] + half_arm, SLICE_CLAW[0][-1] + half_arm), 
                (SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw, SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw), (SLICE_CLAW[1][0] + half_arm * 2, SLICE_CLAW[1][-1] + half_arm * 2)
                ]
        elif USE_QIANGNAO:  
            DEFAULT_ARM_JOINT_NAMES = DEFAULT_ARM_JOINT_NAMES[:half_arm] + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                    + DEFAULT_ARM_JOINT_NAMES[half_arm:] + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand:]               
            arm_slice = [
                (SLICE_ROBOT[0][0] - UP_START_INDEX, SLICE_ROBOT[0][-1] - UP_START_INDEX),(SLICE_DEX[0][0] + half_arm, SLICE_DEX[0][-1] + half_arm), 
                (SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand, SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand), (SLICE_DEX[1][0] + half_arm * 2, SLICE_DEX[1][-1] + half_arm * 2)
                ]
        DEFAULT_JOINT_NAMES_LIST = [DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)]  
    else:
        if USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = DEFAULT_ARM_JOINT_NAMES[:half_arm] + DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                    + DEFAULT_ARM_JOINT_NAMES[half_arm:] + DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
        elif USE_QIANGNAO:
            DEFAULT_ARM_JOINT_NAMES = DEFAULT_ARM_JOINT_NAMES[:half_arm] + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                    + DEFAULT_ARM_JOINT_NAMES[half_arm:] + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand:]             
        DEFAULT_JOINT_NAMES_LIST = DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES

    port_kuavo_rosbag(raw_dir, repo_id, root=lerobot_dir,n = n,)