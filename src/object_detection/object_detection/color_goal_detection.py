import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import sys

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

class ColorObjDetectionNode(Node):
    def __init__(self):
        super().__init__('color_goal_detection_node')
        self.get_logger().info('Color Goal Detection Node Started')
        
        # Declare the parameters for the color detection
        self.declare_parameter('color_low', [110, 50, 150])
        self.declare_parameter('color_high', [130, 255, 255])
        self.declare_parameter('object_size_min', 1000)
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        
        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the detected object and the bounding box
        self.pub_detected_obj = self.create_publisher(Image, '/detected_color_goal',10)
        self.pub_detected_obj_pose = self.create_publisher(PoseStamped, '/detected_color_goal_pose', 10)
        # Create a subscriber to the RGB and Depth images
        self.sub_rgb = Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = Subscriber(self, PointCloud2, '/camera/depth/points')
        # Create a time synchronizer
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.1)
        # Register the callback to the time synchronizer
        self.ts.registerCallback(self.camera_callback)

    def camera_callback(self, rgb_msg, points_msg):
        #self.get_logger().info('Received RGB and Depth Messages')
        # get ROS parameters
        param_color_low = np.array(self.get_parameter('color_low').value)
        param_color_high = np.array(self.get_parameter('color_high').value)
        param_object_size_min = self.get_parameter('object_size_min').value
        
        # Convert the ROS image message to a numpy array
        rgb_image = self.br.imgmsg_to_cv2(rgb_msg,"bgr8")
        # to hsv
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # color mask
        color_mask = cv2.inRange(hsv_image, param_color_low, param_color_high)
        # find largest contour
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # threshold by size
            if w * h < param_object_size_min:
                return
            # draw rectangle
            rgb_image=cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
        else:
            return
        # get the location of the detected object using point cloud
        pointid = (center_y*points_msg.row_step) + (center_x*points_msg.point_step)
        (X, Y, Z) = struct.unpack_from('fff', points_msg.data, offset=pointid)
        center_points = np.array([X,Y,Z])

        if np.any(np.isnan(center_points)):
            return

        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform('base_footprint',rgb_msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.2))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_robot = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
            # Create a pose message for the detected object
            detected_obj_pose = PoseStamped()
            detected_obj_pose.header.frame_id = 'base_footprint'
            detected_obj_pose.header.stamp = rgb_msg.header.stamp
            detected_obj_pose.pose.position.x = cp_robot[0]
            detected_obj_pose.pose.position.y = cp_robot[1]
            detected_obj_pose.pose.position.z = cp_robot[2]
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Publish the detected object
        self.pub_detected_obj_pose.publish(detected_obj_pose)
        # publush the detected object image
        detect_img_msg = self.br.cv2_to_imgmsg(rgb_image, encoding='bgr8')
        detect_img_msg.header = rgb_msg.header
        self.get_logger().info('image message published')
        self.pub_detected_obj.publish(detect_img_msg)
        
def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    color_obj_detection_node = ColorObjDetectionNode()
    # Spin the node so the callback function is called.
    rclpy.spin(color_obj_detection_node)
    # Destroy the node explicitly
    color_obj_detection_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
