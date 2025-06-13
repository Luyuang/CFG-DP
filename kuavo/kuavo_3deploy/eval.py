
# %%
import time
import cv2
import torch
import dill

import rospy
import os
import imageio
import numpy as np
import subprocess
import logging

from deploy_config import *

if MODEL_FR == 'oridp':
    ## dp lib
    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.real_world.real_inference_util import (get_real_obs_resolution, get_real_obs_dict)
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from diffusion_policy.policy.base_image_policy import BaseImagePolicy
    OmegaConf.register_new_resolver("eval", eval, replace=True)
elif MODEL_FR == 'lerobot':
    ## le lib
    from pathlib import Path
    import imageio
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    import time
    import rospy
    import threading
    import signal
    import cv2
    import sys

from env import KuavoEnv, log_env, log_model


#######################
### set up logger
#######################

def main():
    #######################
    ### load checkpoint
    #######################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    if MODEL_FR == 'oridp':
        payload = torch.load(open(CKPT_PATH, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        n_obs_steps = cfg.n_obs_steps
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # hacks for method-specific setup.
        if 'diffusion' in cfg.name:
            # diffusion model
            policy: BaseImagePolicy
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            
            device = torch.device('cuda')
            policy.eval().to(device)

            # set inference params
            policy.num_inference_steps = 16 # DDIM inference iterations
            policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        else:
            raise RuntimeError("Unsupported policy type: ", cfg.name)
    elif MODEL_FR == 'lerobot':
        if POLICY == 'diffusion':
            policy = DiffusionPolicy.from_pretrained(Path(CKPT_PATH))
        elif POLICY == 'act':
            policy = ACTPolicy.from_pretrained(Path(CKPT_PATH))
        policy.eval()
        policy.to(device)
        policy.reset()
        n_obs_steps = policy.config.n_obs_steps
    # ====================
    # log model info
    # ====================
    
    log_model.info(f"Model loaded from {CKPT_PATH}")
    # log_model.info(f"Model config: {cfg}")
    log_model.info(f"Model n_obs_steps: {n_obs_steps}")
    # log_model.info(f"Model n_action_steps: {policy.n_action_steps}")
    # log_model.info(f"Model num_inference_steps: {policy.num_inference_steps}")
    log_model.info(f"Model device: {device}")
    
    

    rospy.init_node('test_il', anonymous=True)
    with KuavoEnv(
        frequency=FREQUENCY,
        n_obs_steps=n_obs_steps,
        video_capture_fps=30,
        robot_publish_rate=500,
        video_capture_resolution=(640, 480),
        ) as env:
            is_just_img = IS_JUST_IMG
            
            #####################
            ### prepare obs buffer
            ##################### 
            print("waiting for the obs buffer to be ready ......")
            env.obs_buffer.wait_buffer_ready(is_just_img)

            #####################
            ### prepare recording
            ##################### 
            print("waiting for the video recorder to be ready ......")
            task_name = TASK_NAME
            import threading, signal, sys,  time
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            current_file_path = os.path.abspath(__file__)
            test_rec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), "test_rec")
            os.makedirs(test_rec_path, exist_ok=True)
            
            # TODO: 使用orin推理不存储.mp4
            # realworld video recorder
            output_video_path_x1 = os.path.join(test_rec_path, f'{current_time}_{task_name}_x1.mp4')
            record_title = f"{current_time}: {task_name} - {MODEL_FR} - {POLICY}"
            record_title = f""
            stop_event = threading.Event()  
            video_thread = threading.Thread(target=env.record_video, args=(output_video_path_x1, 640, 480, 30, True, stop_event, record_title))
            video_thread.start()
            
            # model input video recorder
            output_video_path_xn = os.path.join(test_rec_path, f'./{current_time}_{task_name}_xn.mp4')
            writer = imageio.get_writer(output_video_path_xn, fps=10, codec="libx264")
            
            
            def handle_exit_signal(signum, frame, stop_event):
                print(f"Signal received, saving video to {test_rec_path} and cleaning up...")
                stop_event.set()  # 停止视频录制
                cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
                sys.exit(0)  # 退出程序
            
            # 注册信号处理器
            signal.signal(signal.SIGINT, lambda sig, frame: handle_exit_signal(sig, frame, stop_event))
            signal.signal(signal.SIGQUIT, lambda sig, frame: handle_exit_signal(sig, frame, stop_event))
            
            #####################
            ### start predict and control loop
            #####################
            print("waiting for the control to start ......")     
            
            while True:
                # ========= human control loop =========
                print("skip Human in control!")
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    import matplotlib.pyplot as plt

                    while True:
                        # =============get obs================
                        obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs(is_just_img)
                        
                        # =============check time sync================
                        time_stamp_array = []
                        for k, v in camera_obs_timestamps.items():
                            time_stamp_array.append([v[i] for i in range(n_obs_steps)])
                        for k, v in robot_obs_timestamps.items():
                            time_stamp_array.append([v[i] for i in range(n_obs_steps)])
                        
                    
                        log_env.info(f"obs time stamps ({n_obs_steps}):\n" + "\n".join([f'{v}' for v in time_stamp_array]))
                        time_stamp_array = np.array(time_stamp_array)
                        
                        
                        tolerance = 1 / FREQUENCY
                        env_info_diffs = np.abs(time_stamp_array[:, None, :] - time_stamp_array[None, :, :])
                        triu_indices = np.triu_indices(env_info_diffs.shape[0], k=1)
                        filtered_diffs = env_info_diffs[triu_indices]
                        # log_env.info(f"同时刻不同obs之间的时间差约等于0:\n{env_info_diffs}")
                        # assert np.all(np.abs(filtered_diffs) < tolerance), f"同时刻不同类obs之间的时间差超出允许误差范围\n{env_info_diffs}"


                        tolerance = 1 / FREQUENCY
                        adjacent_obs_diffs = np.diff(time_stamp_array, axis=1)
                        # log_env.info(f"同obs不同时刻之间的时间差约等于{1/FREQUENCY}: \n{adjacent_obs_diffs}")
                        # assert np.all(np.abs(adjacent_obs_diffs - 1/FREQUENCY) < tolerance), f"同类obs不同时刻之间的时间差超出允许误差范围\n{adjacent_obs_diffs}"
                        log_env.info(f"diff time stamps check passed.")
                        

                        # # =============show sense img================
                        # imgs = [cv2.cvtColor(obs[k][-1], cv2.COLOR_RGB2BGR) for k in obs.keys() if "img" in k]
                        # concatenated_img = np.hstack(imgs) 
                        
                        # if os.getenv("DISPLAY") is not None:
                        #     try:
                        #         cv2.imshow('Image Stream', concatenated_img)
                        #         cv2.waitKey(1)
                        #         # cv2.destroyAllWindows()
                        #     except Exception as e:
                        #         log_env.warning(f"cv2.imshow() 失败: {e}")
                        #         pass
                        # else:
                        #     log_env.warning("没有找到显示设备，无法显示图像。")

                        # from PIL import Image, ImageDraw, ImageFont
                        # concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)
                        # pil_img = Image.fromarray(concatenated_img)
                        # draw = ImageDraw.Draw(pil_img)
                        # font = ImageFont.load_default()  # 使用默认字体，可以选择其他字体
                        # title = record_title
                        # draw.text((15, 15), title, font=font, fill="green")
                        # concatenated_img_with_text = np.array(pil_img)
                        # writer.append_data(concatenated_img_with_text)

                      
                        # =============run inference================
                        if MODEL_FR == 'oridp':
                            for obs_name in cfg.task.shape_meta.obs.keys():
                                if obs_name in ORIDP_MAP:
                                    obs[obs_name] = obs[CAM_MAP[ORIDP_MAP[obs_name]]]
                            with torch.no_grad():
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                                obs_dict = dict_apply(obs_dict_np, 
                                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                s = time.time()
                                result = policy.predict_action(obs_dict)
                                inference_time = time.time() - s
                                
                                log_model.info(f"Inference latency: {inference_time}")
                                # this action starts from the first obs step
                                action = result['action'][0].detach().to('cpu').numpy()
                        elif MODEL_FR == 'lerobot':
                            # 将所有数据转换为 PyTorch 张量
                            state = torch.from_numpy(obs["agent_pos"]).float()
                            images = {key: torch.from_numpy(obs[key]).float() / 255 for key in obs if "img" in key}

                            # 调整图像通道顺序 (NHWC -> NCHW)
                            for key in images:
                                images[key] = images[key].permute(0, 3, 1, 2)

                            # 发送数据到 GPU
                            device_kwargs = {"device": device, "non_blocking": True}
                            
                            if POLICY == 'diffusion':
                                state = state.to(**device_kwargs).unsqueeze(0)
                                images = {key: img.to(**device_kwargs).unsqueeze(0) for key, img in images.items()}
                            elif POLICY == 'act':
                                state = state.to(**device_kwargs)                   
                                images = {key: img.to(**device_kwargs) for key, img in images.items()}
                            
                            observation = {}
                            for obs_name in policy.config.input_features.keys():
                                if 'observation.images' in obs_name:
                                    observation[obs_name] = images[CAM_MAP[obs_name.split('.')[-1]]]
                                elif 'observation.state' in obs_name:
                                    observation[obs_name] = state
                            
                            # Predict the next action with respect to the current observation
                            with torch.inference_mode():
                                action = policy.select_action(observation)  # act 1.4-1.5s for 30 actions
                            action = action.to("cpu").numpy()
                        else:
                            raise RuntimeError("Unsupported model fr: ", MODEL_FR)
                  
                        # =============execute actions================
                        if DEBUG:
                            env.exec_fake_actions(
                                 actions=action[:,:], current_state = obs['agent_pos'], which_arm = WHICH_ARM,eef_type = EEF_TYPE,
                            )
                        else:
                            env.exec_actions(
                                actions=action[:,:], current_state = obs['agent_pos'], which_arm = WHICH_ARM, eef_type= EEF_TYPE,
                            )
                        log_model.info(f"Submitted {len(action)} steps of actions.")
                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.close()
                    exit(0)
                print("Stopped.")


# %%
if __name__ == '__main__':
    main()
