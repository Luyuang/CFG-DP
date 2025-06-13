import time
import subprocess
import os
from pathlib import Path
import argparse
import rospy

# 话题列表（可修改）
TOPIC_LIST = [
    "/cam_l/color/image_raw/compressed",
    "/cam_r/color/image_raw/compressed",
    "/joint_cmd",
    "/sensors_data_raw",
    "/control_robot_hand_position",
    "/control_robot_hand_position_state",
    "/zedm/zed_node/left/image_rect_color/compressed",
    "/zedm/zed_node/right/image_rect_color/compressed",
]

# 预期的频率（单位 Hz）
EXPECTED_FREQUENCY = {
    "/cam_l/color/image_raw/compressed": 30,
    "/cam_r/color/image_raw/compressed": 30,
    "/joint_cmd": 500,
    "/sensors_data_raw": 500,
    "/control_robot_hand_position": 100,
    "/control_robot_hand_position_state": 100,
    "/zedm/zed_node/left/image_rect_color/compressed": 30,
    "/zedm/zed_node/right/image_rect_color/compressed": 30,
}

def check_topics():
    """检查话题是否存在"""
    rospy.init_node("topic_checker", anonymous=True)
    active_topics = rospy.get_published_topics()
    active_topics = [topic[0] for topic in active_topics]

    missing_topics = [topic for topic in TOPIC_LIST if topic not in active_topics]

    if missing_topics:
        print("\n[❌] 以下话题未找到，请检查 ROS 系统是否正常运行：")
        for topic in missing_topics:
            print(f"   - {topic}")
        return False

    print("\n[✅] 所有话题都已发布，检查通过！")
    return True

def check_topic_frequency():
    """检查话题的发布频率"""
    all_passed = True

    for topic, expected_freq in EXPECTED_FREQUENCY.items():
        print(f"检查话题 {topic} 频率（预期 ~{expected_freq} Hz）...")
        try:
            # output = subprocess.check_output(["rostopic", "hz", topic], stderr=subprocess.STDOUT, timeout=5)
            output = subprocess.check_output(["rostopic", "hz", topic], stderr=subprocess.STDOUT, timeout=5)
            print(output)
            print(output.decode())
            lines = output.decode().split("\n")
            for line in lines:
                if "average rate:" in line:
                    freq = float(line.split(":")[-1].strip())
                    if expected_freq * 0.8 <= freq <= expected_freq * 1.2:
                        print(f"   [✅] 话题 {topic} 频率 {freq:.2f} Hz")
                    else:
                        print(f"   [⚠️] 话题 {topic} 频率 {freq:.2f} Hz，偏离预期值！")
                        all_passed = False
                    break
            else:
                print(f"   [❌] 话题 {topic} 没有收到数据！")
                all_passed = False
        except subprocess.CalledProcessError:
            print(f"   [❌] 话题 {topic} 查询失败，可能未发布！")
            all_passed = False
        except subprocess.TimeoutExpired:
            print(f"   [❌] 话题 {topic} 没有响应！")
            all_passed = False

    return all_passed


def check_timestamps():
    """检查话题是否包含时间戳"""
    all_passed = True

    for topic in TOPIC_LIST:
        print(f"检查话题 {topic} 是否包含时间戳...")
        try:
            output = subprocess.check_output(["rostopic", "echo", "-n", "1", topic], stderr=subprocess.STDOUT, timeout=5)
            if any(keyword in output.decode() for keyword in ["stamp", "secs", "nsecs"]):
                print(f"   [✅] 话题 {topic} 包含时间戳")
            else:
                print(f"   [❌] 话题 {topic} **没有** 时间戳！")
                all_passed = False
        except subprocess.CalledProcessError:
            print(f"   [❌] 话题 {topic} 查询失败，可能未发布！")
            all_passed = False
        except subprocess.TimeoutExpired:
            print(f"   [❌] 话题 {topic} 没有响应！")
            all_passed = False

    return all_passed

def prepare_and_record(bag_folder_path: str, cnt: int, duration: int, wait: int):
    """ 采集 rosbag 数据 """
    current_file_directory = Path(bag_folder_path)
    current_directory_name = os.path.basename(current_file_directory)

    for i in range(cnt):
        print(f"准备开始记录数据（{i+1}/{cnt}），请在 {wait} 秒内完成准备工作...")
        
        # 倒计时提示
        for t in range(wait, 0, -1):
            print(f"{t} 秒后开始记录...", end="\r")
            time.sleep(1)

        print(f"\n开始记录数据 {i+1}/{cnt}...")
        
        # 构造 rosbag 录制命令
        command = ["rosbag", "record", "-o", current_directory_name, "--duration", f"{duration}s", "--quiet"] + TOPIC_LIST
        
        # 执行录制
        subprocess.run(command, cwd=current_file_directory)

    print("所有采集任务已完成！")

def main():
    parser = argparse.ArgumentParser(description="Record rosbag data with configurable parameters.")
    
    # 默认 rosbag 目录为当前脚本所在目录
    default_bag_folder = os.path.dirname(__file__)
    
    parser.add_argument(
        "-b", "--bag_folder_path",
        type=str,
        default=default_bag_folder,
        help=f"存储 rosbag 数据的目录，默认为 {default_bag_folder}"
    )
    parser.add_argument(
        "-c", "--cnt",
        type=int,
        default=25,
        help="录制次数（默认 25 次）"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=20,
        help="每次录制的持续时间（秒），默认 20s"
    )
    parser.add_argument(
        "-w", "--wait",
        type=int,
        default=5,
        help="每次录制间的等待时间（秒），默认 5s"
    )
    
    args = parser.parse_args()

    # 运行检查
    # print("\n========== 进行话题检查 ==========")
    # if not check_topics():
    #     print("[❌] 话题检查失败，程序终止！")
    #     return

    print("\n========== 检查话题频率 ==========")
    if not check_topic_frequency():
       print("[⚠️] 话题频率异常，请确认是否继续。")
       input("按 Enter 继续，或 Ctrl+C 或者 Ctrl+\取消...")

    print("\n========== 检查时间戳 ==========")
    if not check_timestamps():
        print("[⚠️] 部分话题缺少时间戳，请确认是否继续。")
        input("按 Enter 继续，或 Ctrl+C 或者 Ctrl+\取消...")

    print("\n✅ 所有检查完成，开始 rosbag 录制！")
    prepare_and_record(args.bag_folder_path, args.cnt, args.duration, args.wait)

if __name__ == "__main__":
    main()
