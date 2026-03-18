import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

class FakeObjPose(Node):
    def __init__(self):
        super().__init__('fake_object_node')
        self.get_logger().info('Fake object Node Started')
        
        # Create publisher for the control command
        self.pub_pose = self.create_publisher(PoseStamped, '/detected_color_object_pose', 10)
    
        # Create timer
        self.timer = self.create_timer(0.3, self.timer_update)
    
    def timer_update(self):

        detected_obj_pose_world = PoseStamped()
        detected_obj_pose_world.header.frame_id = 'base_footprint'
        detected_obj_pose_world.header.stamp = self.get_clock().now().to_msg()
        detected_obj_pose_world.pose.position.x = 2.
        detected_obj_pose_world.pose.position.y = 0.
        detected_obj_pose_world.pose.position.z = 0.
        self.pub_pose.publish(detected_obj_pose_world)
        
def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    fake_obj_node = FakeObjPose()
    rclpy.spin(fake_obj_node)
    # Destroy the node explicitly
    fake_obj_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
