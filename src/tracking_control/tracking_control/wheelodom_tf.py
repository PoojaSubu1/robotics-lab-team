import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf2_ros

import geometry_msgs.msg

class WheelOdomTfNode(Node):
    def __init__(self):
        super().__init__('wheel_odom_tf_node')
        self.subscription = self.create_subscription(
            Odometry,
            'odom_raw',
            self.odom_callback,
            10
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def odom_callback(self, msg):
        # Create a transform message
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = msg.header.frame_id
        transform.child_frame_id = msg.child_frame_id
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    wheel_odom_tf_node = WheelOdomTfNode()
    rclpy.spin(wheel_odom_tf_node)
    wheel_odom_tf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()