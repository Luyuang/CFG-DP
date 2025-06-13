import matplotlib.pyplot as plt
import os
import glob
import time
from tqdm import tqdm
from typing import Tuple
import argparse
import shutil
import zarr
import yaml
from pathlib import Path
import numpy as np

from kuavo_utils.replay_buffer import ReplayBuffer
from kuavo_utils.bag_utils import RosbagReader
from kuavo_utils.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs()

def read_rosbag(bag_path: str, config_dict: dict, folders: dict) -> Tuple:
    """Main entry point for processing rosbag data"""
    
    reader = RosbagReader(bag_path,
                         cfg = config_dict,
                         save_plt_folder=folders['save_plt_folder'],
                         save_lastPic_folder=folders['save_lastPic_folder'],
                         raw_video_folder=folders['raw_video_folder'])
    return reader.process_bag(config_dict)


def process_bag_files(bag_folder_path: str, folders: dict, config_path: dict,  num_of_bag: int=None):
    
    task_name = os.path.basename(os.path.dirname(bag_folder_path))

    bag_paths = glob.glob(f"{bag_folder_path}/*.bag")
    # Find .bag files
    if isinstance(num_of_bag, int) and num_of_bag > 0:
        # random sample num_of_bag files
        select_idx = np.random.choice(len(bag_paths), num_of_bag, replace=False)
        bag_paths = [bag_paths[i] for i in select_idx]
    
    print(f"Select {len(bag_paths)} bag files.")
    
    # Define output zarr path
    output_zarr_path = os.path.join(folders['save_zarr_folder'], f"{task_name}.zarr")
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer.create_from_path(output_zarr_path, mode='a')
    # Process each bag file
    for path in tqdm(bag_paths, desc="Processing bags", unit="bag"):
        start_time = time.time()
        print(f"Processing {path}")
        
        # Get the seed (number of episodes) from replay buffer
        seed = replay_buffer.n_episodes

        data_dict = read_rosbag(path, config_path, folders)
        
        compressor_map = {}
        chunks_map = dict()
        for key, value in data_dict.items():
            if 'img' not in key:
                compressor_map[key] = None
            else:
                compressor_map[key] = Jpeg2k(level=40)
                
            chunks_map[key] = value.shape
            print(value.shape)
        # Add episode to replay buffer
        # replay_buffer.add_episode(data_dict, compressors='disk')
        replay_buffer.add_episode(data_dict, compressors = compressor_map, chunks=chunks_map)
        print(f"Saved seed {seed}")
        
        elapsed_time = time.time() - start_time
        print(f"Time taken for {path}: {elapsed_time:.2f} seconds")
     # shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
    cache_zarr_path = os.path.join(output_zarr_path, '../' + f"{task_name}.zarr.zip")
    print('Saving cache to disk.')
    from filelock import FileLock
    cache_lock_path = cache_zarr_path + '.lock'
    with FileLock(cache_lock_path):
        with zarr.ZipStore(cache_zarr_path) as zip_store:
            replay_buffer.save_to_store(
                store=zip_store
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process rosbag files and save to zarr format")
    
    # 获取当前工作目录
    cwd = Path.cwd()
    print(cwd)
    parser.add_argument(
        "-b", "--bag_folder_path", 
        type=str, 
        default=str(cwd / "kuavo" / "kuavo_1convert" / "dataset" / "Task_0" / "rosbag"),
        help="The rosbag folder under a task folder, e.g. '/app/data-convert/data-example/Task1-RearangeToy/kuavo-rosbag'"
    )

    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default=str(cwd / "kuavo" / "kuavo_1convert" / "config" / "Task_0.yaml"),
        help="The configuration file path, e.g. '/app/data-train-deploy/src/config/Task2-RearangeToy.json'"
    )

    parser.add_argument(
        "-n", "--num_of_bag", 
        type=int, 
        default=None,  
        help="The number of bag files to process, e.g. 3"
    )

    parser.add_argument(
        "-a", "--append", 
        action="store_true", 
        help="Append to existing zarr file"
    )
    
    parser.add_argument(
        '-l', '--jpeg_compress_level',
        default = 40,
        type = int,
        help="image compress level"
    )
    parser.add_argument(
        '-v', '--process_version',
        default = 'v0',
        type = str,
        help="process version"
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        exit(1)

def prepare_folders(arge):
    base_path = arge.bag_folder_path
    append = arge.append
    process_version = arge.process_version
    
    """Prepare necessary folders, clearing them if not in append mode."""
    base_path = Path(base_path)
    
    folders = {
        "save_plt_folder": base_path.parent / process_version / "plt-check" / "motor-plt",
        "save_lastPic_folder": base_path.parent / process_version / "plt-check" / "last-pic",
        "save_zarr_folder": base_path.parent / process_version / "kuavo-zarr",
        "raw_video_folder": base_path.parent / process_version / "raw-video",
        "sample_video_folder": base_path.parent / process_version /"sample-video"
    }
    
    if not append:
        for folder in folders.values():
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
    
    return folders

def main():
    args = parse_args()
    config = load_config(args.config)
    
    folders = prepare_folders(args)

    process_bag_files(args.bag_folder_path, folders, config, num_of_bag=args.num_of_bag)

    # Check the zarr file structure
    try:
        zarr_file = zarr.open(store=zarr.DirectoryStore(folders["save_zarr_folder"]))
        print(zarr_file.tree())
    except Exception as e:
        print(f"Error reading Zarr file: {e}")

if __name__ == "__main__":
    main()