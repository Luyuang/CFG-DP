#!/bin/bash
# Script to convert rosbag data to lerobot format and train a model in one step

# Make the script exit on error
set -e

# Display help information
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --task_name NAME       Task name (e.g., Task20_conveyor_pick) (required)"
    echo "  --raw_dir DIR          Path to raw ROS bag directory (required)"
    echo "  --version, -v VER      Process version (default: v0)"
    echo "  --base_dir DIR         Base directory for the task (default: /home/leju-ali/hx/kuavo/{task_name})"
    echo "  --num_of_bag, -n NUM   Number of bag files to process (default: all)"
    echo "  --num_processes NUM    Number of processes for distributed training (default: 2)"
    echo "  --port PORT            Main process port for distributed training (default: 29503)"
    echo "  --policy_type TYPE     Policy type for training (default: act)"
    echo "  --skip_conversion      Skip the conversion step and only run training"
    echo "  --skip_training        Skip the training step and only run conversion"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --task_name Task20_conveyor_pick --raw_dir /path/to/rosbag --version v0 --num_processes 2"
    exit 0
}

# Parse command line arguments
POSITIONAL_ARGS=()
ADDITIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --task_name|--raw_dir|--version|-v|--base_dir|--num_of_bag|-n|--num_processes|--port|--policy_type|--skip_conversion|--skip_training)
            if [[ "$1" == "--port" ]]; then
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --main_process_port $2"
                shift 2
            else
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1 $2"
                shift 2
            fi
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# Check if required arguments are provided
if [[ "$ADDITIONAL_ARGS" != *"--task_name"* ]] || [[ "$ADDITIONAL_ARGS" != *"--raw_dir"* ]]; then
    echo "Error: --task_name and --raw_dir are required arguments"
    show_help
fi

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Run the Python script with all arguments
echo "Running conversion and training with arguments: $ADDITIONAL_ARGS"
python "$SCRIPT_DIR/convert_and_train.py" $ADDITIONAL_ARGS
