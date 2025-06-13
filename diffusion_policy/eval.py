
# %%
import time

import cv2
import torch
import dill
import hydra
from omegaconf import OmegaConf
from KuavoEnv import KuavoEnv
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import rospy
input="/home/leju-ali/hx/oridp/ckpt/epoch=0060-train_loss=0.011.ckpt"
# input = '/home/leju-ali/hx/oridp/ckpt/epoch=0100-train_loss=0.006.ckpt'
OmegaConf.register_new_resolver("eval", eval, replace=True)

print("cuda",torch.cuda.is_available())
time.sleep(5)
def main():
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
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

    # setup experiment
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)

    rospy.init_node("test")
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
            import threading, signal, sys
            env.obs_buffer.wait_buffer_ready(is_just_img)
            print("Warming up policy inference")
            latency = 0
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            output_video_path = f'./{cfg.task.name}_{current_time}_speedx1_latency{latency}ms.mp4'
            stop_event = threading.Event()  
            record_title = f"Latency of [Obs_trans, PredictRet_trans, Motor] =[{latency}ms, {latency}ms, {latency}ms]"
            video_thread = threading.Thread(target=env.record_video, args=(output_video_path, 640, 480, 30, True,stop_event,record_title))
            video_thread.start()
            
            def handle_exit_signal(signum, frame, stop_event):
                print("Signal received, saving video and cleaning up...")
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
                    # start episode
                    policy.reset()
                    import matplotlib.pyplot as plt

                    
                    import imageio
                    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    output_video_path = f'./{cfg.task.name}_{current_time}_speedx_latency{latency}ms.mp4'
                    writer = imageio.get_writer(output_video_path, fps=10, codec="libx264")
                    while True:
                        # =============
                        # get obs
                        # =============
                        obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs(is_just_img)
                        print('##################################################################')
                        print(f"{camera_obs_timestamps['img01'][0]:.10f}, {camera_obs_timestamps['img01'][1]:.10f}, \n"
                            f"{camera_obs_timestamps['img02'][0]:.10f}, {camera_obs_timestamps['img02'][1]:.10f}, \n"
                        )
                        if not is_just_img:
                            print(f"{robot_obs_timestamps['ROBOT_state_joint'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_joint'][1]:.10f}, \n"
                                f"{robot_obs_timestamps['ROBOT_state_gripper'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_gripper'][1]:.10f}\n"
                                )
                        print('##################################################################')
                        import numpy as np
                        # create fake obs
                        
                        # =============
                        # show sense img
                        # =============
                        from PIL import Image, ImageDraw, ImageFont
                        # show sense img

                        imgs = []
                        for k, v in obs.items():
                            if "img" in k:
                                img = obs[k][-1]
                                img = cv2.resize(img, (640, 480))
                                img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                imgs.append(img01_bgr)
                                
                        concatenated_img = np.hstack(imgs)  # 横向拼接图像
                        cv2.imshow('Image Stream', concatenated_img)
                        
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
                        
                        # run inference
                        with torch.no_grad():
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            s = time.time()
                            result = policy.predict_action(obs_dict)
                            inference_time = time.time() - s
                            
                            print('Inference latency:', inference_time)
                          
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
           

                        # execute actions
                        
                        env.exec_actions(
                            actions=action[:13,:],
                        )
                        # time.sleep(0.8)
                        print(f"Submitted {len(action)} steps of actions.")

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.close()
                    exit(0)
                print("Stopped.")


# %%
if __name__ == '__main__':
    main()
