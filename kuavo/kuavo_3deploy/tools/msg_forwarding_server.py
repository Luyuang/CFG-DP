import rospy 
from sensor_msgs.msg import JointState
def callback(msg):
    data = msg.position
    # angel to rad
    data = [angle * 3.1415926 / 180 for angle in data]
    # create a new message
    new_msg = JointState()
    # set the header
    new_msg.header.stamp = rospy.Time.now()
    # set the name
    new_msg.name = msg.name
    # set the position
    new_msg.position = data
    # publish the message
    pub.publish(new_msg)

sub = rospy.Subscriber('/kuavo_arm_traj', JointState, callback)
pub = rospy.Publisher('/kuavo_arm_traj_rad', JointState, queue_size=10)

if __name__ == '__main__':
    rospy.init_node('joint_state_converter', anonymous=True)
    rospy.loginfo("Joint state converter is running...")
    rospy.spin()