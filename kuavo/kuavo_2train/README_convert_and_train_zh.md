# 转换与训练工具

该工具提供了一键式解决方案，用于将 rosbag 数据转换为 lerobot 格式并训练模型。

## 功能特性

- 将 rosbag 数据转换为 lerobot 格式
- 使用转换后的数据训练模型
- 通过单个命令组合转换和训练两个步骤
- 自定义转换和训练的参数
- 根据需要跳过转换或训练步骤

## 使用方法

### 使用 Shell 脚本

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /path/to/rosbag --version v0
```

### 直接使用 Python 脚本

```bash
python kuavo/convert_and_train.py --task_name Task20_conveyor_pick --raw_dir /path/to/rosbag --version v0
```

## 参数说明

### 必需参数

- `--task_name`: 任务名称 (例如：Task20_conveyor_pick)
- `--raw_dir`: 原始 ROS bag 文件目录路径

### 可选参数

#### 通用参数

- `--version`, `-v`: 处理版本 (默认: v0)
- `--base_dir`: 任务的基础目录 (默认: /home/leju-ali/hx/kuavo/{task_name})

#### 转换参数

- `--num_of_bag`, `-n`: 要处理的 bag 文件数量 (默认: 全部)
- `--skip_conversion`: 跳过转换步骤，仅运行训练

#### 训练参数

- `--num_processes`: 分布式训练的进程数 (默认: 2)
- `--port` (shell 脚本) 或 `--main_process_port` (Python 脚本): 分布式训练的主进程端口 (默认: 29503)
- `--policy_type`: 训练的策略类型 (默认: act)
- `--skip_training`: 跳过训练步骤，仅运行转换
- `--additional_train_args` (仅限 Python 脚本): 传递给训练脚本的额外参数

## 示例

### 基本用法

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag
```

### 指定版本

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --version v1
```

### 限制 Bag 文件数量

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --num_of_bag 5
```

### 自定义训练参数

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --num_processes 4 --port 29504 --policy_type diffusion
```

### 跳过转换 (使用现有数据)

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --skip_conversion
```

### 跳过训练 (仅转换数据)

```bash
./kuavo/convert_and_train.sh --task_name Task20_conveyor_pick --raw_dir /home/leju-ali/hx/kuavo/Task20_conveyor_pick/rosbag --skip_training
```

## 输出目录

- Lerobot 数据集: `/home/leju-ali/hx/kuavo/{task_name}/{version}/lerobot`
- 训练输出: `/home/leju-ali/hx/kuavo/{task_name}/{version}/train_lerobot`

您可以使用 `--base_dir` 参数自定义这些路径。
