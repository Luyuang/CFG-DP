import rosbag
bag_path = "/home/lejurobot/hx/kuavo/Task14_cup/rosbag/rosbag_2025-03-27-18-22-44.bag"
bag_path = '/home/lejurobot/hx/kuavo/Task13_zed_dualArm/rosbag/rosbag_2025-03-20-17-06-41.bag'
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[
                                                   "/cam_l/color/image_raw/compressed"
                                                #    "/sensors_data_raw"
                                                   ]):
        correct_timestamp = t.to_sec()  
        msg_stamp = msg.header.stamp.to_sec()
        print(msg_stamp, correct_timestamp)