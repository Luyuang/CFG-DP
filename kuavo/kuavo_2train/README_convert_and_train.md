# Convert and Train Tool

This tool provides a one-click solution for converting rosbag data to lerobot format and training a model.

## Features

- Convert rosbag data to lerobot format
- Train a model using the converted data
- Combine both steps in a single command
- Customize parameters for both conversion and training
- Skip either conversion or training if needed

## Usage

### Using the Shell Script

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /path/to/rosbag --version v0
```

### Using the Python Script Directly

```bash
python kuavo/convert_and_train.py --task_name Task20_conveyor_pick --raw_dir /path/to/rosbag --version v0
```

## Parameters

### Required Parameters

- `--task_name`: Task name (e.g., Task20_conveyor_pick)
- `--raw_dir`: Path to raw ROS bag directory

### Optional Parameters

#### General Parameters

- `--version`, `-v`: Process version (default: v0)
- `--base_dir`: Base directory for the task (default: /home/leju-ali/hx/kuavo/{task_name})

#### Conversion Parameters

- `--num_of_bag`, `-n`: Number of bag files to process (default: all)
- `--skip_conversion`: Skip the conversion step and only run training

#### Training Parameters

- `--num_processes`: Number of processes for distributed training (default: 2)
- `--port` (shell script) or `--main_process_port` (Python script): Main process port for distributed training (default: 29503)
- `--policy_type`: Policy type for training (default: act)
- `--skip_training`: Skip the training step and only run conversion
- `--additional_train_args` (Python script only): Additional arguments to pass to the training script

## Examples

### Basic Usage

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag
```

### Specifying Version

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --version v1
```

### Limiting Number of Bag Files

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --num_of_bag 5
```

### Customizing Training Parameters

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --num_processes 4 --port 29504 --policy_type diffusion
```

### Skip Conversion (Use Existing Data)

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --skip_conversion
```

### Skip Training (Only Convert Data)

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --skip_training
```

## Output Directories

- Lerobot dataset: `/home/leju-ali/hx/kuavo/{task_name}/{version}/lerobot`
- Training output: `/home/leju-ali/hx/kuavo/{task_name}/{version}/train_lerobot`

You can customize these paths using the `--base_dir` parameter.
