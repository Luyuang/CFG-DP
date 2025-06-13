Enhancing Humanoid Robot Screwing Tasks with Classifier-Free Guidance in Diffusion Policies

This repository contains the code and resources for our project, which improves humanoid robot performance in screwing tasks by addressing the problem of repetitive actions. We enhance the Diffusion Policy (DP) framework using Classifier-Free Guidance (CFG), achieving more reliable task completion with fewer redundant movements.

Overview

Screwing tasks, where a humanoid robot must rotate a tool to fasten a workpiece and then retract its arm, are challenging for traditional DP methods. These methods often get stuck in repetitive cycles because they lack awareness of task progression, leading to inefficient or failed task completion. Our solution introduces CFG to guide the robot’s actions, combining a conditional model (aware of task progress via a normalized step count, timestep) with an unconditional model to ensure precise termination of screwing cycles. By modeling temporal context explicitly, we help the robot decide when to stop screwing and move on to the next step.

Our approach uses a transformer-based architecture to process inputs from two cameras (head and wrist), arm motor states, and timestep, predicting smooth action sequences. The result is a policy that completes screwing tasks with high success rates, minimal repetition, and faster execution compared to baselines like DP and ACT.

Key Features





Classifier-Free Guidance (CFG): Combines conditional and unconditional diffusion models with a dynamic guidance factor to prioritize task termination, reducing repetitive actions.



Temporal Awareness: Incorporates timestep  to track task progress, ensuring the robot knows when to end a screwing cycle.

How It Works

We collect demonstration data using a humanoid robot equipped with a 7-DoF arm and a dexterous hand, capturing 200 trajectories via VR-based teleoperation. Each trajectory includes RGB images from two cameras (Orbbec 335L on the head, Realsense D405 on the wrist), joint angles, hand states, and timestep. Data is stored as ROS bags and converted to Zarr format for efficient training.

The training process enhances DP by integrating CFG. The conditional model predicts actions based on observations and timestep, while the unconditional model provides a baseline action distribution. During inference, CFG blends these predictions with a guidance factor that increases as the task nears completion, ensuring the robot retracts its arm after two screwing cycles. 

We deploy the trained model on a physical robot using ROS, with the policy processing live camera feeds and motor states at 10 Hz. 

Repository Contents

kuavo_il/
├── kuavo/                # Core functional modules
│   ├── kuavo_1convert/   # Data collection and conversion
│   ├── kuavo_2train/     # Training
│   ├── kuavo_3deploy/    # Deployment
│   └── kuavo_utils/      # Tools
├── diffusion_policy/     # Diffusion Policy framework
└── lerobot/              # LeRobot framework


Data Collection

Record demonstrations using kuavo/kuavo_1convert/collect_data/record.py, ensuring ROS topics for cameras, joints, and hand states are correctly configured. Convert ROS bags to Zarr with:

DATASET_DIR=~/kuavo 
TASK_NAME=TASK0_weighting 
mkdir -p $DATASET_DIR/$TASK_NAME/rosbag
cp kuavo/kuavo_1convert/collect_data/record.py $DATASET_DIR/$TASK_NAME/rosbag



ROS Bag to LeRobot

LeRobot format is a data format dedicated to the LeRobot framework





Modify the configuration file in kuavo_dataset

# bag -> lerobot 
python kuavo/kuavo_1convert/cvt_rosbag2lerobot.py --raw_dir $DATASET_DIR/Task12_zed_dualArm/rosbag --repo_id Task12_zed_dualArm/lerobot







--raw_dir: ROS bag file directory



--repo_id: dataset repository ID

![4cam: plt-check](docs/vis_motor_as.png)

Visualization





After conversion,  use the tools provided by LeRobot to visualize the data and check the data quality:

# lerobot dataset：
python lerobot/lerobot/scripts/visualize_dataset.py --repo-id Task12_zed_dualArm/lerobot --root $DATASET_DIR/kuavo/Task12_zed_dualArm/lerobot --episode 55 --local-files-only 1

![4cam: plst-check](docs/vis_le.png)


Training

Train the model with:

CUDA_VISIBLE_DEVICES=0
python lerobot/lerobot/scripts/train.py \
      --dataset.repo_id v0/lerobot \
      --policy.type act \
      --dataset.local_files_only true \
      --dataset.root ~/luyuang/nls/v0/lerobot





Results and Impact

Our experiments on a 7-DoF humanoid robot show significant improvements over baselines, making the policy suitable for precise, repetitive tasks like screwing. The reduced repetition and faster execution enhance efficiency, with potential applications in manufacturing and assembly. The integration of CFG and temporal modeling offers a generalizable approach for other sequential robotic tasks.



Contact

For questions, open an issue or email [yuanglu0219@hotmail.com].
