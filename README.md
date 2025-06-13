# Enhancing Humanoid Robot Screwing Tasks with Classifier-Free Guidance in Diffusion Policies

This repository contains the code and resources for our project, which improves humanoid robot performance in screwing tasks by addressing the problem of repetitive actions. We enhance the Diffusion Policy (DP) framework using Classifier-Free Guidance (CFG), achieving more reliable task completion with fewer redundant movements.

## Overview
Screwing tasks, where a humanoid robot must rotate a tool to fasten a workpiece and then retract its arm, are challenging for traditional DP methods. These methods often get stuck in repetitive cycles because they lack awareness of task progression, leading to inefficient or failed task completion. Our solution introduces CFG to guide the robot’s actions, combining a conditional model (aware of task progress via a normalized step count, timestep) with an unconditional model to ensure precise termination of screwing cycles. By modeling temporal context explicitly, we help the robot decide when to stop screwing and move on to the next step.

Our approach uses a transformer-based architecture to process inputs from two cameras (head and wrist), arm motor states, and timestep, predicting smooth action sequences. The result is a policy that completes screwing tasks with high success rates, minimal repetition, and faster execution compared to baselines like DP and ACT.



## Repository Contents

```text
kuavo_il/
├── kuavo/                # Core functional modules
│   ├── kuavo_1convert/   # Data collection and conversion
│   ├── kuavo_2train/     # Training
│   ├── kuavo_3deploy/    # Deployment
│   └── kuavo_utils/      # Tools
├── diffusion_policy/     # Diffusion Policy framework
└── lerobot/              # LeRobot framework
```


## Data collection

Record demonstrations using kuavo/kuavo_1convert/collect_data/record.py, ensuring ROS topics for cameras, joints, and hand states are correctly configured. Convert ROS bags to Zarr with:

```bash

DATASET_DIR=~/kuavo 
TASK_NAME=TASK0_weighting 
mkdir -p $DATASET_DIR/$TASK_NAME/rosbag
cp kuavo/kuavo_1convert/collect_data/record.py $DATASET_DIR/$TASK_NAME/rosbag 

/home/$USER/
├── kuavo
│   └── Task2-RearangeToy 
│       └── rosbag
│           └── record.py
```


## Conversion (ROS Bag to Lerobot)


1. Modify kuavo_dataset configuration

```bash
# bag -> lerobot 
python kuavo/kuavo_1convert/cvt_rosbag2lerobot.py --raw_dir $DATASET_DIR/Task12_zed_dualArm/rosbag --repo_id Task12_zed_dualArm/lerobot
```

- `--raw_dir`: ROS bag file directory
- `--repo_id`: dataset repository ID


![4cam: plt-check](docs/vis_motor_as.png)

#### Visualization

After conversion,  use the tools provided by LeRobot to visualize the data and check the data quality:

```bash
# lerobot dataset：
python lerobot/lerobot/scripts/visualize_dataset.py --repo-id Task12_zed_dualArm/lerobot --root $DATASET_DIR/kuavo/Task12_zed_dualArm/lerobot --episode 55 --local-files-only 1
```
![4cam: plst-check](docs/vis_le.png)

## LeRobot Training

[LeRobot](https://github.com/huggingface/lerobot) is a robot learning framework developed by Hugging Face.

#### Standalone training #kuavo_il directory
```bash 
CUDA_VISIBLE_DEVICES=0
python lerobot/lerobot/scripts/train.py \
      --dataset.repo_id v0/lerobot \
      --policy.type act \
      --dataset.local_files_only true \
      --dataset.root ~/luyuang/nls/v0/lerobot
```
## Contact

For questions, open an issue or email [yuanglu0219@hotmail.com].
