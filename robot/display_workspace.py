import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

# bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])
bounds = np.array([[0.65, 1.07], [-0.28, 0.28], [-0.1, 0.05]]) # (0.42, 0.56, 0.2)

def publish_rectangle():
    rospy.init_node('rectangle_marker')
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "rectangle"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    # Set the pose of the marker
    marker.pose.position.x = (bounds[0][0] + bounds[0][1])/2.0
    marker.pose.position.y = (bounds[1][0] + bounds[1][1])/2.0
    marker.pose.position.z = (bounds[2][0] + bounds[2][1])/2.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Set the scale of the marker (1x2x0.5 rectangle)
    marker.scale.x = bounds[0][1] - bounds[0][0]
    marker.scale.y = bounds[1][1] - bounds[1][0]
    marker.scale.z = bounds[2][1] - bounds[2][0]

    # Set the color (RGBA)
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.25

    rate = rospy.Rate(10)  # 10 Hz

    counter = 0

    while not rospy.is_shutdown():
        rate.sleep()
        counter += 1
        if counter > 10:
            if marker.action == Marker.ADD:
                marker.action = Marker.DELETE
            else:
                marker.action = Marker.ADD
            counter = 0
            
        marker.header.stamp = rospy.Time.now()
        marker_pub.publish(marker)
        

if __name__ == '__main__':
    try:
        publish_rectangle()
    except rospy.ROSInterruptException:
        pass