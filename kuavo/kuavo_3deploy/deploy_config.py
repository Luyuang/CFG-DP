from sensor_msgs.msg import CompressedImage

from kuavo_msgs.msg import lejuClawState, sensorsData, robotHandPosition, armTargetPoses    # type: ignore
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest, controlLejuClawResponse
# =============================================================================
################################################
# FOR ENVIRONMENT
################################################
W, H = 640, 480 #224, 224
EEF_TYPE = "dex_hand" # "dex_hand" or "leju_claw"

DEFAULT_OBS_KEY_MAP = {
    "img":{
        "img01": {
            "topic":"/zedm/zed_node/left/image_rect_color/compressed",
            "msg_type":CompressedImage,
            'frequency': 30,
            'handle': {
                "params": {
                    'resize_wh': (W, H),  # (W,H)
                }
            }
        },
        "img02": {
            "topic":"/zedm/zed_node/right/image_rect_color/compressed",
            "msg_type":CompressedImage,
            'frequency': 30,
            'handle': {
                "params": {
                    'resize_wh': (W, H),  # (W,H)
                }
            }
        },
        "img03": {
            "topic":"/cam_l/color/image_raw/compressed",
            "msg_type":CompressedImage,
            'frequency': 30,
            'handle': {
                "params": {
                    'resize_wh': (W, H),  # (W,H)
                }
            }
        },
        "img04": {
            "topic":"/cam_r/color/image_raw/compressed",
            "msg_type":CompressedImage,
            'frequency': 30,
            'handle': {
                "params": {
                    'resize_wh': (W, H),  # (W,H)
                }
            }
        },
    },
    "low_dim":{
        "state_joint": {
            "topic":"/sensors_data_raw",
            "msg_type":sensorsData,
            "frequency": 500,
            'handle': {
                "params": {
                    'slice': [
                                # (12,19), 
                                (19, 26)
                              ]
                }
            },
        },
            
        # "state_dex_hand": {
        #     "topic":"/control_robot_hand_position_state",
        #     "msg_type":robotHandPosition,
        #     "frequency": 100,
        #     'handle': {
        #         "params": {
        #             'slice': [
        #                         # (0,1),
        #                         (6,7)
        #                       ]
        #         }        
        #     },
        # },
        # "state_leju_claw": {
        #     "topic":"/leju_claw_state",
        #     "msg_type":lejuClawState,
        #     "frequency": 500,
        #     'handle': {
        #         "params": {
        #             'slice': [
        #                         # (0,1),
        #                         (1,2)
        #                       ]
        #         }        
        #     },
        # },
    }
}

if EEF_TYPE == "dex_hand":
    DEFAULT_OBS_KEY_MAP["low_dim"]["state_dex_hand"] = {
        "topic":"/control_robot_hand_position_state",
        "msg_type":robotHandPosition,
        "frequency": 100,
        'handle': {
            "params": {
                'slice': [
                            # (0,1),
                            (6,7)
                          ]
            }        
        },
    }
elif EEF_TYPE == "leju_claw":
    DEFAULT_OBS_KEY_MAP["low_dim"]["state_leju_claw"] = {
        "topic":"/leju_claw_state",
        "msg_type":lejuClawState,
        "frequency": 500,
        'handle': {
            "params": {
                'slice': [
                            # (0,1),
                            (1,2)
                          ]
            }        
        },
    }
else:
    raise ValueError("EEF_TYPE must be 'dex_hand' or 'leju_claw'")

DEFAULT_ACT_KEY_MAP = {
    "target_left_eef_pose": "placeholder",
    "taget_gripper": "/control_robot_hand_position",
    "target_joint": "/kuavo_arm_traj",
}

DEX_HAND_OPEN_STATE = [0, 100, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0]
DEX_HAND_CLOSE_STATE = [100] * 12
LEJU_CLAW_OPEN_STATE = [0, 0]
LEJU_CLAW_CLOSE_STATE = [100, 100]

################################################
# FOR INFERENCE
################################################

test_cfg = {
    'model_fr': 'lerobot', # 'oridp' or 'lerobot'
    'policy': 'act', # 'diffusion' or 'act'
    'task': 'Task21_conveyor_pick1_act',
    'ckpt': [
        '/home/leju_kuavo/model/Task21_conveyor_pick1/v0/train_leact/2025-04-29/10-28-17_act/checkpoints/040000/pretrained_model',
        '/home/leju_kuavo/model/Task21_conveyor_pick/v0/train_lerobot/checkpoints/060000/pretrained_model',
        '/home/leju_kuavo/model/Task21_conveyor_pick/v0/train_lerobot/checkpoints/100000/pretrained_model',
        '/home/leju_kuavo/model/Task20_conveyor_pick/v0/outputs/train/2025-04-23/01-45-09_act/checkpoints/100000/pretrained_model',
        '/home/leju_kuavo/model/Task20_conveyor_pick/v0/outputs/train/2025-04-23/01-51-15_diffusion/checkpoints/100000/pretrained_model',
        '/home/leju-ali/hx/kuavo/Task16_nls/V_2pos/train_oridp/data/outputs/2025.04.10/00.48.06_train_diffusion_unet_image_Task16_nls/checkpoints/epoch=0120-train_loss=0.006.ckpt', # oridp ok
        '/home/leju_kuavo/hx/kuavo/Task17_cup/V_1/train_lerobot/outputs/train/2025-04-13/13-42-18_act/checkpoints/120000/pretrained_model', # leact, Task17_cup
        '/home/leju_kuavo/hx/kuavo/Task17_cup/V_1/train_lerobot/outputs/train/2025-04-13/13-42-18_act/checkpoints/020000/pretrained_model', # leact, Task17_cup
        '/home/leju-ali/hx/kuavo/Task16_nls/V_2pos/train_oridp/data/outputs/2025.04.10/00.48.06_train_diffusion_unet_image_Task16_nls/checkpoints/epoch=0300-train_loss=0.001.ckpt',
        '/home/leju-ali/hx/kuavo/Task16_nls/V_2pos/train_lerobot/checkpoints/160000/pretrained_model', # leact, Task16_nls
        '/home/leju-ali/hx/kuavo/Task16_nls/V_1/train_lerobot/checkpoints/060000/pretrained_model', # leact, Task16_nls, bad
        '/home/leju-ali/hx/kuavo/Task16_nls/V_1/train_oridp/epoch=0100-train_loss=0.005.ckpt', 
        '/home/leju-ali/hx/kuavo/Task16_nls/V_1/train_oridp/epoch=0040-train_loss=0.017.ckpt',
        '/home/leju-ali/hx/kuavo/Task14_cup/train_lerobot/lejurobot_leact/home/lejurobot/hx/kuavo/Task14_cup/train_lerobot/checkpoints/120000/pretrained_model', # leact, Task14_cup. OK
        '/home/leju-ali/hx/kuavo/Task14_cup/V_50bags/200000/pretrained_model', # leact, Task14_cup
        '/home/leju-ali/hx/kuavo/Task14_cup/train_oridp/data/outputs/2025.04.01/19.58.17_train_diffusion_unet_image_Task14_cup/checkpoints/epoch=0060-train_loss=0.006.ckpt',
        '/home/leju-ali/hx/kuavo/Task14_cup/train_oridp/data/outputs/2025.04.02/22.05.44_train_diffusion_unet_image_Task14_cup/checkpoints/epoch=0050-train_loss=0.008.ckpt',
        '/home/leju-ali/hx/kuavo_il/outputs/train/2025-04-01/23-28-29_diffusion/checkpoints/200000/pretrained_model', # diffusion
        '/home/leju-ali/hx/kuavo_il/outputs/train/2025-04-01/22-58-45_act/checkpoints/100000/pretrained_model', # act
        '/media/leju-ali/PortableSSD/lejurobot/data/outputs/2025.04.01/19.58.17_train_diffusion_unet_image_Task14_cup/checkpoints/epoch=0080-train_loss=0.003.ckpt',
        '/home/leju-ali/hx/kuavo/Task14_cup/train_lerobot/checkpoints/200000/pretrained_model', # leact, Task14_cup
        '/home/leju-ali/hx/kuavo/Task14_cup/train_oridp/outputs/2025.03.28/19.36.40_train_diffusion_unet_image_Task14_cup/checkpoints/epoch=0090-train_loss=0.003.ckpt', # oridp
        '/home/leju-ali/hx/kuavo_il/outputs/train/2025-03-29/02-23-51_diffusion/checkpoints/140000/pretrained_model', #双GPU训练diffusion, Task15_car
        '/home/leju-ali/hx/kuavo/Task13_zed_dualArm/train_lerobot/outputs/train/2025-03-26/23-01-20_act/checkpoints/080000/pretrained_model', #双GPU act
        '/home/leju-ali/hx/kuavo/Task12_zed_dualArm/train_lerobot/outputs/train/2025-03-21/02-25-36_act/checkpoints/180000/pretrained_model',   #双gpu训练act
        '/home/leju-ali/hx/ckpt/wks/dataset/dataset_wason_20250307/data/outputs/2025.03.07/21.26.03_train_diffusion_unet_image_Task11_Toy/checkpoints/epoch=0140-train_loss=0.004.ckpt',
        '/home/leju-ali/hx/ckpt/wks/dataset/dataset_wason_20250307/data/outputs/2025.03.09/23.31.47_train_diffusion_unet_image_Task11_Toy/checkpoints/epoch=0100-train_loss=0.007.ckpt',   # 单臂抓娃娃ok
            ],
    'debug': False,
    'is_just_img': False,
    'fps': 10,
    'which_arm': 'right', # 'left' or 'right' or 'both'
}

DEBUG = test_cfg['debug']
TASK_NAME = test_cfg['task'] + test_cfg['model_fr'] + test_cfg['policy']
CKPT_PATH = test_cfg['ckpt'][0]
IS_JUST_IMG = test_cfg['is_just_img']
FREQUENCY = test_cfg['fps']
MODEL_FR = test_cfg['model_fr']
POLICY = test_cfg['policy']
WHICH_ARM = test_cfg['which_arm']
CAM_MAP = {
            'head_cam_l':'img01', 
            'head_cam_r':'img02', 
            'wrist_cam_l':'img03', 
            'wrist_cam_r':'img04', 
            }
OBS_LIST = None

ORIDP_MAP = {
    'img02': 'head_cam_r',
    'img04': 'wrist_cam_r',
}

# rosservice call /arm_traj_change_mode "control_mode: 2" 
# python kuavo/kuavo_3deploy/tools/dex_hand_fake_state.py & 
# python kuavo/kuavo_3deploy/tools/arm_joint_deg_interpolator.py -p Task21_r_start1