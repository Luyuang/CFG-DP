# %% [markdown]
# # Offline Eval

# %%
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import random,os

home_path = os.path.expanduser("~")
root = os.path.join(home_path, "hx/kuavo/Task17_cup/V_1/lerobot") 
parts = root.split("/")[-3:]
repo_id = '_'.join(parts[:2]) + '/' + parts[2]
local_files_only = True


episode_index = 44

le_dataset = LeRobotDataset(repo_id=repo_id, local_files_only=local_files_only, root=root)

from_idx = le_dataset.episode_data_index["from"][episode_index].item()
to_idx = le_dataset.episode_data_index["to"][episode_index].item()
num_episodes = le_dataset.episode_data_index["from"].shape[0]
select_num = 2
selected_episodes = random.sample(range(num_episodes), select_num) if num_episodes >= select_num else list(range(num_episodes))

timestamps = [x / le_dataset.fps for x in range(to_idx - from_idx)]
print(timestamps)
delta_timestamps = {
    "observation.state": timestamps,
    # "observation.images.cam_h": timestamps,
    "observation.images.wrist_cam_l": timestamps,
    "observation.images.wrist_cam_r": timestamps,
    "observation.images.head_cam_l": timestamps,
    # "observation.images.right": timestamps,
    "action": timestamps,
}
val_dataset = LeRobotDataset(repo_id=repo_id, local_files_only=local_files_only, root=root, delta_timestamps=delta_timestamps)
one_eps =next(iter(val_dataset))
observation_keys = [obs_k for obs_k in one_eps.keys() if "observation" in obs_k and "pad" not in obs_k]
for k, v in one_eps.items():
    if k in observation_keys or k == "action":
        print(k, v.shape)
        print(v.min(), v.max())


# %%
from pathlib import Path
import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
import os

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  
'''
act
/wks/dataset/20250309/act/outputs/train/2025-03-11/17-24-11_act/checkpoints/500000

diffusion
/wks/dataset/20250309/500000/outputs/train/2025-03-09/23-43-51_diffusion/checkpoints/500000
'''
# Task12_zed_dualArm
ckpt_path = "/home/lejurobot/hx/kuavo_il/outputs/train/2025-03-20/19-16-45_act/checkpoints/100000/pretrained_model"
ckpt_path = "/home/lejurobot/hx/kuavo_il/outputs/train/2025-03-21/02-25-36_act/checkpoints/120000/pretrained_model"
ckpt_path = "/home/lejurobot/hx/kuavo_il/outputs/train/2025-03-22/14-03-48_diffusion/checkpoints/320000/pretrained_model"
ckpt_path = "/home/lejurobot/hx/kuavo_il/outputs/train/2025-03-24/22-18-55_diffusion/checkpoints/140000/pretrained_model"
ckpt_path ='/home/lejurobot/hx/kuavo_il/outputs/train/2025-03-26/10-10-34_act/checkpoints/060000/pretrained_model'
ckpt_path = '/home/lejurobot/hx/kuavo/Task14_cup/train_lerobot/checkpoints/160000/pretrained_model' 
# task17_cup
ckpt_path = '/home/leju_kuavo/hx/kuavo/Task17_cup/V_1/train_lerobot/outputs/train/2025-04-13/03-07-02_act/checkpoints/040000/pretrained_model', # leact, Task17_cup
ckpt_size = os.path.getsize(ckpt_path + '/model.safetensors') / (1024 ** 3)

pretrained_policy_path = Path(ckpt_path)
policy = ACTPolicy.from_pretrained(pretrained_policy_path)
# policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
# policy.num_inference_steps = 10

policy.eval()
policy.to(device)
policy.reset()

obs_state_dim = policy.config.input_features['observation.state'].shape[0]
n_obs_step = policy.config.n_obs_steps
test_cfg = {
    "slice":[0, 16],
    "fps": 10,
    "low_dim": obs_state_dim,
    "n_obs_step": n_obs_step,
}
print(obs_state_dim)
assert obs_state_dim == test_cfg['slice'][1] - test_cfg['slice'][0]


# %%
test_cfg

# %%
import numpy as np
from collections import defaultdict
import tqdm

all_img: defaultdict[str, list] = defaultdict(list)
upBody_lowDim: defaultdict[str, list] = defaultdict(list)

def add_noise_to_images(images, noise_level=0.2):
    """ 给归一化图像添加噪声 """
    noise = (np.random.rand(*images.shape) * 2 - 1) * noise_level
    return np.clip(images + noise, 0, 1)
#.unsqueeze(0)
def add_noise_to_state(data, noise_scale=0.5):
    """ 给状态或动作数据添加噪声 """
    noise = np.random.uniform(-noise_scale, noise_scale, size=data.shape)
    return data + noise
    
def le_predict(obs_seq, act_seq, policy, device):
    pred_actions, true_actions, true_states = [], [], []
    WINDOW_SIZE = policy.config.n_obs_steps
    for step in tqdm.trange(len(next(iter(obs_seq.values()))) - (WINDOW_SIZE - 1)):
        observation = {
            key: torch.from_numpy(np.stack(values[step:step + WINDOW_SIZE])).float().to(device)
            # .unsqueeze(0)
            for key, values in obs_seq.items()
        }

        with torch.inference_mode():
            action = policy.select_action(observation).cpu().numpy()
        
        final_step = step + WINDOW_SIZE - 1
        pred_actions.append(action)
        true_actions.append(act_seq[final_step])
        true_states.append(obs_seq['observation.state'][final_step])

    return pred_actions, true_actions, true_states

def process_images_from_tensor_to_uint8(all_img):
    for img_eps in all_img.values():
        for i in range(len(img_eps)):
            if isinstance(img_eps[i], torch.Tensor):
                img_eps[i] = img_eps[i].cpu().numpy()  # 先转为 NumPy 数组
                img_eps[i] = np.array([
                    np.transpose((img * 255).clip(0, 255).astype(np.uint8), (1, 2, 0))
                    for img in img_eps[i]
                ])
    return all_img

def prepare_ledata_seq(val_dataset, eps_idx):
    one_eps = val_dataset.get_one_episode(val_dataset.episode_data_index["from"][eps_idx].item())

    observation_keys = [obs_k for obs_k in one_eps.keys() if "observation" in obs_k and "pad" not in obs_k]
    obs_seq = {
        k: add_noise_to_state(one_eps[k], noise_scale=0.01)
        for k in observation_keys if 'images' not in k
    }
    obs_seq.update({
        k: add_noise_to_images(one_eps[k], noise_level=0.02)
        for k in observation_keys if 'images' in
        k
    })
    act_seq = one_eps['action'][:, test_cfg['slice'][0]:test_cfg['slice'][1]]
    return obs_seq, act_seq
    
def main(selected_episodes, dataset, test_cfg, policy, device):
    all_pred_actions, all_true_actions, all_true_states = [], [], []
    all_img = defaultdict(list)
    
    for i in selected_episodes:
        obs_seq, act_seq = prepare_ledata_seq(dataset, eps_idx=i)
        pred_actions, true_actions, true_states = le_predict(obs_seq, act_seq, policy, device)
        
        all_pred_actions.append(np.array(pred_actions))
        all_true_actions.append(np.array(true_actions))
        all_true_states.append(np.array(true_states))
        
        for key in [k for k in obs_seq.keys() if "images" in k]:
            all_img[key].append(obs_seq[key])
    
    all_img = process_images_from_tensor_to_uint8(all_img)
    upBody_lowDim = {
        "pred_actions": all_pred_actions,
        "true_actions": all_true_actions,
        "true_states": all_true_states,
    }
    return all_img, upBody_lowDim
    
all_img, upBody_lowDim = main(selected_episodes, val_dataset, test_cfg, policy, device)


# %%
for k,v in upBody_lowDim.items():
    print(k, len(v), v[0].shape)

for k, v in all_img.items():
    print(k, len(v),)


# %%
def draw_predicted_result(task_name, all_img, upBody_lowDim, eps_idx):
    import numpy as np
    import matplotlib.pyplot as plt
    img_strips = []
    show_img_num = 18
    for img_name, img_eps in all_img.items():
        img_strip = np.concatenate(np.array(img_eps[eps_idx])[::len(img_eps[eps_idx])//show_img_num], axis=1)  # Row for images
        img_strips.append(img_strip)

    img_strip_combined = np.vstack(img_strips)
    
    JOINT_DIM_LABELS= ["zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link", 
                  "dex_hand_l1_link", "dex_hand_l2_link", "dex_hand_l3_link", "dex_hand_l4_link", "dex_hand_l5_link", "dex_hand_l6_link", 
                  "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link", 
                  "dex_hand_r1_link", "dex_hand_r2_link", "dex_hand_r3_link", "dex_hand_r4_link", "dex_hand_r5_link", "dex_hand_r6_link", 
                  ] 
    img_rows = len(all_img)
    figure_layout = [
            JOINT_DIM_LABELS[:7],
            JOINT_DIM_LABELS[7:13] + ['extra1'],
            JOINT_DIM_LABELS[13:-6] ,
            JOINT_DIM_LABELS[-6:]+ ['extra2'],
        ]
    for i in range(img_rows):
        row_images = ['image'] * len(JOINT_DIM_LABELS[:7])
        figure_layout.insert(i, row_images)


    plt.rcParams.update({'font.size': 7})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([30, 15]) 
    # fig.suptitle(task_name, fontsize=15)

    SELECTED_JOINT = JOINT_DIM_LABELS[0:8] + JOINT_DIM_LABELS[13:21] 
    for action_dim, action_label in enumerate(SELECTED_JOINT):
        for low_dim_name, low_dim_values in upBody_lowDim.items():
            if 'pred' in low_dim_name:  
                # selected_range = range(len(low_dim_values[eps_idx]) - 8)  # len(low_dim_values[eps_idx]) - 8
                # draw_pred_num = 12
                # for j in selected_range[::draw_pred_num]:  # select range
                #     x_range = np.arange(j, j + min(draw_pred_num, low_dim_values[eps_idx].shape[1]))  
                #     y_values = low_dim_values[eps_idx][j, :draw_pred_num, action_dim]  
                #     axs[action_label].plot(x_range, y_values,  alpha=0.5, zorder=1)
                axs[action_label].plot(low_dim_values[eps_idx][:, 0, action_dim], label=low_dim_name, alpha=1, zorder=1)

            elif 'true_actions' in low_dim_name:    # (n, 26)
                axs[action_label].plot(low_dim_values[eps_idx][:, action_dim], label=low_dim_name, alpha=0.5, zorder=1)
                
            elif 'true_states' in low_dim_name:
                axs[action_label].plot(low_dim_values[eps_idx][:, action_dim], label=low_dim_name, alpha=0.2, zorder=1)
        axs[action_label].set_xlabel(action_label, labelpad=5)  
        # axs[action_label].set_xlabel('Time in one episode')
        axs[action_label].legend()

    axs['image'].imshow(img_strip_combined)
    # axs['image'].set_xlabel('Time in one episode (subsampled)')
    # axs['image'].set_title('Image Comparison')

    plt.legend()
    plt.show()
    return fig

# %%
# from kuavo_utils.plot import draw_predicted_result
task_name = ckpt_path + '  ' + f"{ckpt_size:.2f} G" + '  ' + f'{()=}'
for eps_idx in range(len(next(iter(all_img.values())))):
    fig = draw_predicted_result(task_name, all_img, upBody_lowDim, eps_idx)
    fig.savefig(f"{task_name}_{eps_idx}.jpeg", dpi=80, bbox_inches='tight')
# %%



