#!/usr/bin/env python3
"""
Script to convert Kuavo rosbag data to the LeRobot dataset format and then train a model.

This script combines the functionality of cvt_rosbag2lerobot.py and train_distributed.py
to provide a one-click solution for data conversion and model training.

Example usage:
python convert_and_train.py --task_name Task20_conveyor_pick --raw_dir /path/to/rosbag --version v0 --num_processes 2
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert rosbag to lerobot format and train a model")
    
    # General parameters
    parser.add_argument("--task_name", type=str, required=True, 
                        help="Task name (e.g., Task20_conveyor_pick)")
    parser.add_argument("--version", "-v", type=str, default="v0",
                        help="Process version (default: v0)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for the task (default: /home/leju-ali/hx/kuavo/{task_name})")

    # Conversion parameters
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to raw ROS bag directory")
    parser.add_argument("--num_of_bag", "-n", type=int, default=None,
                        help="The number of bag files to process (default: all)")
    parser.add_argument("--skip_conversion", action="store_true",
                        help="Skip the conversion step and only run training")
    
    # Training parameters
    parser.add_argument("--num_processes", type=int, default=2,
                        help="Number of processes for distributed training (default: 2)")
    parser.add_argument("--main_process_port", type=int, default=29503,
                        help="Main process port for distributed training (default: 29503)")
    parser.add_argument("--policy_type", type=str, default="act",
                        help="Policy type for training (default: act)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the training step and only run conversion")
    parser.add_argument("--additional_train_args", type=str, default="",
                        help="Additional arguments to pass to the training script")
    
    return parser.parse_args()

def run_conversion(args):
    """Run the data conversion from rosbag to lerobot format"""
    print(f"=== Starting conversion from rosbag to lerobot format ===")
    
    # Determine paths
    if args.base_dir is None:
        base_dir = f"/home/leju-ali/hx/kuavo/{args.task_name}"
    else:
        base_dir = args.base_dir
    
    raw_dir = args.raw_dir
    version = args.version
    n = args.num_of_bag if args.num_of_bag is not None else None
    lerobot_dir = os.path.join(base_dir, version, "lerobot")
    
    # Build the conversion command
    cmd = [
        "python", 
        "kuavo/kuavo_1convert/cvt_rosbag2lerobot.py",
        "--raw_dir", raw_dir,
        "-v", version
    ]

    if args.num_of_bag is not None:
        cmd.extend(["-n", str(args.num_of_bag)])
    
    # Run the conversion
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print(f"Conversion failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"=== Conversion completed successfully ===")
    print(f"Lerobot dataset created at: {lerobot_dir}")
    
    return lerobot_dir

def run_training(args, lerobot_dir):
    """Run the training using the converted data"""
    print(f"=== Starting training with the converted data ===")
    
    # Determine paths
    if args.base_dir is None:
        base_dir = f"/home/leju-ali/hx/kuavo/{args.task_name}"
    else:
        base_dir = args.base_dir
    
    version = args.version
    output_dir = os.path.join(base_dir, version, "train_lerobot")
    
    # Build the training command
    cmd = [
        "accelerate", "launch",
        f"--num_processes={args.num_processes}",
        f"--main_process_port", str(args.main_process_port),
        f"{os.path.expanduser('~/hx/kuavo_il/lerobot/lerobot/scripts/train_distributed.py')}",
        f"--dataset.repo_id", f"{args.task_name}/lerobot",
        f"--policy.type", args.policy_type,
        f"--dataset.local_files_only", "true",
        f"--dataset.root", lerobot_dir,
        f"--output_dir", output_dir
    ]
    
    # Add any additional training arguments
    if args.additional_train_args:
        cmd.extend(args.additional_train_args.split())
    
    # Run the training
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"=== Training completed successfully ===")
    print(f"Training output saved to: {output_dir}")

def main():
    args = parse_args()
    
    # Determine paths
    if args.base_dir is None:
        base_dir = f"/home/leju-ali/hx/kuavo/{args.task_name}"
    else:
        base_dir = args.base_dir
    
    version = args.version
    lerobot_dir = os.path.join(base_dir, version, "lerobot")
    
    # Run conversion if not skipped
    if not args.skip_conversion:
        lerobot_dir = run_conversion(args)
    else:
        print(f"=== Skipping conversion step ===")
        print(f"Using existing lerobot dataset at: {lerobot_dir}")
    
    # Run training if not skipped
    if not args.skip_training:
        run_training(args, lerobot_dir)
    else:
        print(f"=== Skipping training step ===")

if __name__ == "__main__":
    main()
