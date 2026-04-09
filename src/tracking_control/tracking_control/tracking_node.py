import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math


def hat(k):
    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat


def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[0] * qhat + 2 * qhat2


def euler_from_quaternion(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]


def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class TrackingNode(Node):
    STATE_IDLE = 0
    STATE_GO_TO_GOAL = 1
    STATE_RETURN_HOME = 2
    STATE_ROTATE_TO_START_ORIENTATION = 3
    STATE_DONE = 4

    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')

        # Detected poses are stored in world frame
        self.obs_pose = None
        self.goal_pose = None
        self.last_obs_time = None
        self.last_goal_time = None

        # Start pose
        self.start_position = None
        self.start_yaw = None

        # State
        self.state = self.STATE_IDLE
        self.goal_reached_counter = 0

        # ROS params
        self.declare_parameter('world_frame_id', 'odom')
        self.odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Pub/Sub
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        self.sub_detected_obs_pose = self.create_subscription(
            PoseStamped,
            'detected_color_object_pose',
            self.detected_obs_pose_callback,
            10
        )
        self.sub_detected_goal_pose = self.create_subscription(
            PoseStamped,
            'detected_color_goal_pose',
            self.detected_goal_pose_callback,
            10
        )

        self.timer = self.create_timer(0.01, self.timer_update)

        # General control params
        self.goal_stop_distance = 0.30
        self.home_stop_distance = 0.25
        self.orientation_tolerance = 0.10

        self.max_linear_vel = 0.28
        self.max_angular_vel = 0.60

        self.k_heading = 1.8
        self.k_return_heading = 2.2

        self.slow_distance = 1.5

        # Obstacle handling
        self.obs_timeout = 0.40
        self.obs_filter_max_dist = 4.0
        self.obs_filter_max_height = 0.7

        self.front_obs_distance = 0.75
        self.front_obs_angle = 1.0
        self.emergency_obs_distance = 0.50

        self.repulsion_dist_go = 1.3
        self.repulsion_gain_go = 0.55

        self.repulsion_dist_return = 1.6
        self.repulsion_gain_return = 0.95

        self.get_logger().info('Tracking Node initialized.')

    def stop_cmd(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        return cmd

    def clamp(self, val, low, high):
        return max(low, min(high, val))

    def detected_obs_pose_callback(self, msg):
        center_points = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # Simple filtering
        if np.linalg.norm(center_points) > self.obs_filter_max_dist or abs(center_points[2]) > self.obs_filter_max_height:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_id,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            t_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))
            cp_world = t_R @ center_points + np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException:
            return

        self.obs_pose = cp_world
        self.last_obs_time = self.get_clock().now()

    def detected_goal_pose_callback(self, msg):
        center_points = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_id,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            t_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))
            cp_world = t_R @ center_points + np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException:
            return

        self.goal_pose = cp_world
        self.last_goal_time = self.get_clock().now()

    def get_robot_pose_in_world(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_id,
                'base_footprint',
                rclpy.time.Time()
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.w, q.x, q.y, q.z])
            return np.array([x, y]), yaw
        except TransformException:
            return None, None

    def world_point_to_robot(self, point_world):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint',
                self.odom_id,
                rclpy.time.Time()
            )
            R = q2R([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ])
            t = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            return R @ point_world + t
        except TransformException:
            return None

    def get_current_poses(self):
        obs_in_robot = None
        goal_in_robot = None

        # Drop stale obstacle detection
        if self.obs_pose is not None and self.last_obs_time is not None:
            age = (self.get_clock().now() - self.last_obs_time).nanoseconds / 1e9
            if age > self.obs_timeout:
                self.obs_pose = None
                self.last_obs_time = None

        if self.obs_pose is not None:
            obs_in_robot = self.world_point_to_robot(self.obs_pose)

        if self.goal_pose is not None:
            goal_in_robot = self.world_point_to_robot(self.goal_pose)

        return obs_in_robot, goal_in_robot

    def build_home_target_in_robot(self):
        current_pos, current_yaw = self.get_robot_pose_in_world()
        if current_pos is None or current_yaw is None or self.start_position is None:
            return None, None

        dx = self.start_position[0] - current_pos[0]
        dy = self.start_position[1] - current_pos[1]

        cos_yaw = math.cos(-current_yaw)
        sin_yaw = math.sin(-current_yaw)

        home_x = dx * cos_yaw - dy * sin_yaw
        home_y = dx * sin_yaw + dy * cos_yaw
        home_target = np.array([home_x, home_y, 0.0])

        dist_to_home = np.linalg.norm([dx, dy])
        return home_target, dist_to_home

    def obstacle_avoidance_vector(self, obs_in_robot, repulsion_dist, repulsion_gain):
        if obs_in_robot is None:
            return 0.0, 0.0

        obs_x = obs_in_robot[0]
        obs_y = obs_in_robot[1]
        obs_dist = np.linalg.norm(obs_in_robot[:2])

        if obs_dist < 0.05 or obs_dist > repulsion_dist:
            return 0.0, 0.0

        repulsion_strength = repulsion_gain * (1.0 / obs_dist - 1.0 / repulsion_dist) / (obs_dist ** 2)

        # More push sideways than straight backward
        lateral_scale = 1.8
        f_rep_x = -repulsion_strength * obs_x / obs_dist
        f_rep_y = -repulsion_strength * obs_y / obs_dist * lateral_scale

        return f_rep_x, f_rep_y

    def controller(self, target_in_robot, obs_in_robot, returning_home=False):
        cmd = Twist()

        if target_in_robot is None:
            return cmd

        target_x = target_in_robot[0]
        target_y = target_in_robot[1]
        target_dist = np.linalg.norm(target_in_robot[:2])

        if target_dist < 0.01:
            return cmd

        target_angle = math.atan2(target_y, target_x)

        # Stronger "rotate first" behavior on return
        rotate_first_angle = 0.90 if returning_home else 1.05
        if abs(target_angle) > rotate_first_angle:
            cmd.linear.x = 0.0
            gain = self.k_return_heading if returning_home else self.k_heading
            cmd.angular.z = self.clamp(gain * target_angle, -self.max_angular_vel, self.max_angular_vel)
            return cmd

        # Emergency obstacle behavior:
        # pivot away in place instead of creeping forward
        if obs_in_robot is not None:
            obs_dist = np.linalg.norm(obs_in_robot[:2])
            obs_angle = math.atan2(obs_in_robot[1], obs_in_robot[0])

            if obs_dist < self.emergency_obs_distance and abs(obs_angle) < self.front_obs_angle:
                turn_dir = -1.0 if obs_angle >= 0.0 else 1.0
                cmd.linear.x = 0.0
                cmd.angular.z = turn_dir * self.max_angular_vel
                return cmd

        # Attractive vector
        f_att_x = target_x
        f_att_y = target_y

        # Repulsive vector
        if returning_home:
            repulsion_dist = self.repulsion_dist_return
            repulsion_gain = self.repulsion_gain_return
            heading_gain = self.k_return_heading
        else:
            repulsion_dist = self.repulsion_dist_go
            repulsion_gain = self.repulsion_gain_go
            heading_gain = self.k_heading

        f_rep_x, f_rep_y = self.obstacle_avoidance_vector(obs_in_robot, repulsion_dist, repulsion_gain)

        f_total_x = f_att_x + f_rep_x
        f_total_y = f_att_y + f_rep_y

        desired_angle = math.atan2(f_total_y, f_total_x)
        angular_cmd = heading_gain * desired_angle

        forward_component = math.cos(desired_angle)

        if target_dist < self.slow_distance:
            speed_scale = target_dist / self.slow_distance
        else:
            speed_scale = 1.0

        linear_cmd = self.max_linear_vel * speed_scale * max(0.0, forward_component)

        # Additional slowdown near front obstacle
        if obs_in_robot is not None:
            obs_dist = np.linalg.norm(obs_in_robot[:2])
            obs_angle = math.atan2(obs_in_robot[1], obs_in_robot[0])

            if obs_dist < self.front_obs_distance and abs(obs_angle) < self.front_obs_angle:
                if returning_home:
                    linear_cmd *= 0.20
                else:
                    linear_cmd *= 0.40

        cmd.linear.x = self.clamp(linear_cmd, 0.0, self.max_linear_vel)
        cmd.angular.z = self.clamp(angular_cmd, -self.max_angular_vel, self.max_angular_vel)
        return cmd

    def rotate_to_start_orientation(self):
        cmd = Twist()

        _, current_yaw = self.get_robot_pose_in_world()
        if current_yaw is None or self.start_yaw is None:
            return cmd

        angle_error = wrap_to_pi(self.start_yaw - current_yaw)

        if abs(angle_error) < self.orientation_tolerance:
            self.state = self.STATE_DONE
            self.get_logger().info('Returned home and matched start orientation.')
            return cmd

        cmd.angular.z = self.clamp(1.5 * angle_error, -self.max_angular_vel, self.max_angular_vel)
        return cmd

    def timer_update(self):
        obs_in_robot, goal_in_robot = self.get_current_poses()

        if self.state == self.STATE_IDLE:
            if self.goal_pose is not None:
                current_pos, current_yaw = self.get_robot_pose_in_world()
                if current_pos is not None and current_yaw is not None:
                    self.start_position = current_pos.copy()
                    self.start_yaw = current_yaw
                    self.state = self.STATE_GO_TO_GOAL
                    self.goal_reached_counter = 0
                    self.get_logger().info(
                        f'Starting run from x={self.start_position[0]:.2f}, y={self.start_position[1]:.2f}'
                    )

            self.pub_control_cmd.publish(self.stop_cmd())
            return

        if self.state == self.STATE_GO_TO_GOAL:
            if goal_in_robot is None:
                self.pub_control_cmd.publish(self.stop_cmd())
                return

            goal_dist = np.linalg.norm(goal_in_robot[:2])

            if goal_dist <= self.goal_stop_distance:
                self.goal_reached_counter += 1
                if self.goal_reached_counter >= 10:
                    self.state = self.STATE_RETURN_HOME
                    self.goal_reached_counter = 0
                    self.get_logger().info('Goal reached. Returning home.')
                    self.pub_control_cmd.publish(self.stop_cmd())
                    return
            else:
                self.goal_reached_counter = 0

            cmd = self.controller(goal_in_robot, obs_in_robot, returning_home=False)
            self.pub_control_cmd.publish(cmd)
            return

        if self.state == self.STATE_RETURN_HOME:
            home_target_in_robot, dist_to_home = self.build_home_target_in_robot()

            if home_target_in_robot is None or dist_to_home is None:
                self.pub_control_cmd.publish(self.stop_cmd())
                return

            if dist_to_home <= self.home_stop_distance:
                self.state = self.STATE_ROTATE_TO_START_ORIENTATION
                self.get_logger().info('Reached home position. Rotating to start orientation.')
                self.pub_control_cmd.publish(self.stop_cmd())
                return

            cmd = self.controller(home_target_in_robot, obs_in_robot, returning_home=True)
            self.pub_control_cmd.publish(cmd)
            return

        if self.state == self.STATE_ROTATE_TO_START_ORIENTATION:
            cmd = self.rotate_to_start_orientation()
            self.pub_control_cmd.publish(cmd)
            return

        if self.state == self.STATE_DONE:
            self.pub_control_cmd.publish(self.stop_cmd())
            return


def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
