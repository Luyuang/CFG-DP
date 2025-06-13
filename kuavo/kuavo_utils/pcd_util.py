#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
# import open3d as o3d
import numpy as np
import struct
import time 
import rosbag
# import pytorch3d.ops as torch3d_ops

import torch

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, use_cuda=True):
    
    num_points = 4096

    extrinsics_matrix = np.array([[ 0.5213259,  -0.84716441,  0.10262438,  0.04268034],
                                  [ 0.25161211,  0.26751035,  0.93012341,  0.15598059],
                                  [-0.81542053, -0.45907589,  0.3526169,   0.47807532],
                                  [ 0.,          0.,          0.,          1.        ]])

    extrinsics_matrix = np.array([[ 1, 0,  0,  0],
                                  [ 0,  1,  0,  100],
                                  [0, 0,  1,   0],
                                  [ 0.,          0.,          0.,          1.        ]])
    WORK_SPACE = [
        [0.65, 1.1],
        [0.45, 0.66],
        [-0.7, 0]
    ]

    # scale
    # point_xyz = points[..., :3]*1
    # point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    # point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    # point_xyz = point_homogeneous[..., :-1]
    # points[..., :3] = point_xyz 
    
    # crop
    # points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
    #                             (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
    #                             (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    
    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    return points

def extract_xyzrgb_from_pointcloud2(msg): # should be deprecated, ndarrays are better instead of walking through the data
    """
    从 sensor_msgs/PointCloud2 消息中提取 xyz 和 rgb 数据
    """
    # 获取字段的偏移量
    fields = {field.name: field.offset for field in msg.fields}
    point_step = msg.point_step
    data = msg.data

    is_bigendian = msg.is_bigendian
    unpack_fmt = '>' if is_bigendian else '<'

    xyzrgb = []

    # 遍历点云数据
    for i in range(0, len(data), point_step):
        # 提取 x, y, z 坐标
        x = struct.unpack_from(unpack_fmt + 'f', data, i + fields['x'])[0]
        y = struct.unpack_from(unpack_fmt + 'f', data, i + fields['y'])[0]
        z = struct.unpack_from(unpack_fmt + 'f', data, i + fields['z'])[0]

        # 提取 rgb 数据（浮点型 -> 整数）
        rgb_float = struct.unpack_from(unpack_fmt + 'f', data, i + fields['rgb'])[0]
        rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
        r = (rgb_int >> 16) & 0xFF
        g = (rgb_int >> 8) & 0xFF
        b = rgb_int & 0xFF

        # 保存为一行 xyzrgb
        xyzrgb.append([x, y, z, r, g, b])
    
    return np.array(xyzrgb, dtype=np.float32)


def downsample_pointcloud(xyzrgb, voxel_size):
    """
    对点云数据进行体素下采样
    """
    # 将 numpy 点云转换为 Open3D 的点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])  # xyz 部分
    colors = xyzrgb[:, 3:] / 255.0  # 归一化颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 使用 Open3D 的体素下采样功能
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    # 将下采样后的点云转换为 numpy 格式
    downsampled_xyzrgb = np.hstack((
        np.asarray(pcd_downsampled.points),
        np.asarray(pcd_downsampled.colors) * 255  # 反归一化颜色
    ))

    return downsampled_xyzrgb

def optimized_extract_xyzrgb_from_pointcloud2(msg):
    """
    从 PointCloud2 消息中高效提取 xyz 和 rgb 信息，并返回 (N, 6) 形式的点云。
    处理特定的填充方式：
    - 0-4: x 
    - 4-8: y
    - 8-12: z
    - 12-16: 填充
    - 16-20: rgb
    
    eg.
        import struct
        rgb = 1.8347285e-38
        byte_rep = struct.pack('f', rgb)  # 'f' 表示 float 类
        # 打印字节表示
        print(byte_rep)
        # 以十六进制格式打印字节表示
        print("Hex format:", byte_rep.hex())
        rgb_int = struct.unpack('I', byte_rep)[0]
        print(rgb_int)
        # 提取 RGB 分量
        b = (rgb_int & 0xFF)  
        g = ((rgb_int >> 8) & 0xFF)  
        r = ((rgb_int >> 16) & 0xFF) 
        r ,g , b 
    """
    # 创建自定义的结构化数据类型
    dtype = np.dtype([
        ('x', np.float32),     # 0-4 字节
        ('y', np.float32),     # 4-8 字节
        ('z', np.float32),     # 8-12 字节
        ('_pad', np.uint32),   # 12-16 字节的填充
        ('rgb', np.float32)    # 16-20 字节
    ])

    # 计算点的数量
    num_points = len(msg.data) // msg.point_step

    # 使用自定义数据类型读取数据
    structured_array = np.frombuffer(msg.data, dtype=dtype, count=num_points)

    # 提取 XYZ 坐标
    xyz = np.column_stack((
        structured_array['x'], 
        structured_array['y'], 
        structured_array['z']
    ))

    # 提取 RGB 信息
    rgb = structured_array['rgb'].view(np.uint32)
    r = ((rgb >> 16) & 0xFF).astype(np.float32)
    g = ((rgb >> 8) & 0xFF).astype(np.float32)
    b = (rgb & 0xFF).astype(np.float32)

    # 将 RGB 信息拼接为 (N, 3)
    rgb_combined = np.stack([r, g, b], axis=-1)

    # 拼接 XYZ 和 RGB 为 (N, 6)
    xyzrgb = np.hstack([xyz, rgb_combined])

    num_samples = 20000
    # 采样
    if num_points > num_samples:
        # 随机采样
        indices = np.random.choice(num_points, num_samples, replace=False)
        xyzrgb_sampled = xyzrgb[indices]
    else:
        # 如果点数少于采样数，返回所有点
        xyzrgb_sampled = xyzrgb
    xyzrgb = xyzrgb_sampled
    # 可选：保存 PCD 文件
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # save_to_pcd(xyzrgb, f"pcd_{cur_time}.pcd")
    # display_point_cloud(xyzrgb)
    xyzrgb[:, 3:] = xyzrgb[:, 3:]   # RGB 范围归一化到 [0, 1]
    return xyzrgb

def save_to_pcd(xyzrgb, filename):
    """
    将点云数据保存为 PCD 文件
    :param xyzrgb: 点云数据 (N, 6) 的 numpy 数组，前3列为 XYZ，后3列为 RGB
    :param filename: 保存的文件名
    """
    # 将数据拆分为 XYZ 和 RGB
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:] / 255.0  # RGB 范围归一化到 [0, 1]
    
    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    
    # 设置点云的坐标和颜色
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)
    
    # 保存为 PCD 文件
    o3d.io.write_point_cloud(filename, point_cloud)

def display_point_cloud(xyzrgb):
    """
    可视化点云数据
    :param xyzrgb: 点云数据 (N, 6) 的 numpy 数组，前3列为 XYZ，后3列为 RGB
    """
    # 将数据拆分为 XYZ 和 RGB
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:] / 255.0  # RGB 范围归一化到 [0, 1]

    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 设置点云的坐标和颜色
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)

    # 显示点云
    o3d.visualization.draw_geometries([point_cloud], 
                                      window_name="Point Cloud Viewer",
                                      width=800,
                                      height=600,
                                      point_show_normal=False)
    
    


def display_point_cloud_with_cursor_info(xyzrgb):
    """
    实时显示鼠标指针所指点的 XYZ 和 RGB 值。
    :param xyzrgb: 点云数据 (N, 6) 的 numpy 数组，前3列为 XYZ，后3列为 RGB
    """
    # 拆分点云数据
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:] / 255.0  # RGB 归一化到 [0, 1]

    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)

    # 创建可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer with Cursor Info", width=800, height=600)
    vis.add_geometry(point_cloud)

    # 定义回调函数，实时获取鼠标指针指向的点的信息
    def mouse_callback(vis, action, mods):
        # 获取选中的点索引
        picked_idx = vis.get_picked_points()
        if picked_idx:
            idx = picked_idx[-1]  # 取最后一个被选中的点
            point_xyz = xyz[idx]
            point_rgb = rgb[idx]
            print(f"Point Index: {idx}, XYZ: {point_xyz}, RGB: {point_rgb}")

    # 注册鼠标事件回调
    vis.register_key_callback(ord("P"), mouse_callback)  # 按键 'P' 激活选点模式

    # 启动可视化
    vis.run()
    vis.destroy_window()


def extract_pointcloud_from_rosbag(bag_file, topic_name, output_pcd_file, voxel_size=0.01): # to be mod
    """
    从 rosbag 文件中提取点云数据并保存为 PCD 文件。
    
    参数:
        bag_file: str, rosbag 文件路径
        topic_name: str, 点云数据的 topic 名称
        output_pcd_file: str, 保存的 PCD 文件路径
        voxel_size: float, 体素下采样的尺寸 (单位: 米)
    """
    # 打开 rosbag 文件
    bag = rosbag.Bag(bag_file, 'r')
    
    # 遍历 rosbag 中的指定 topic
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if topic == topic_name:
            print(f"读取消息类型: {type(msg)}")  # 打印消息类型，以确保正确解析
            cur_time = time.time()
            
            # 提取点云数据
            xyzrgb = extract_xyzrgb_from_pointcloud2(msg)
            print(f"提取点云数据耗时: {time.time() - cur_time:.2f} 秒")
            
            # 对点云进行下采样
            cur_time = time.time()
            xyzrgb_downsampled = downsample_pointcloud(xyzrgb, voxel_size)
            print(f"点云下采样耗时: {time.time() - cur_time:.2f} 秒")
            
            # 保存到 PCD 文件
            # save_to_pcd(xyzrgb_downsampled, output_pcd_file)
            # print(f"下采样点云保存到: {output_pcd_file}")
            # break  # 停止读取更多消息（只保存一帧点云）

    # 关闭 rosbag 文件
    bag.close()
