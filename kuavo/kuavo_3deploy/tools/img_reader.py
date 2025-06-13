import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os
def callback(data):
    # Convert the compressed image to a numpy array
    np_arr = np.frombuffer(data.data, np.uint8)

    # Decode the image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # save the image
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_file_dir, 'img.jpg')
    cv2.imwrite(img_path, img)
    # Display the image

    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    # Initialize the ROS node
    rospy.init_node('image_listener', anonymous=True)

    # Create a subscriber to the image topic
    imgsub = rospy.Subscriber('/zedm/zed_node/left/image_rect_color/compressed', CompressedImage, callback)

    # Keep the node running
    rospy.spin()
if __name__ == '__main__':
    main()