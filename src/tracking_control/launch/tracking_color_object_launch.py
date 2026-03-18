from launch import LaunchDescription
from launch_ros.actions import Node 
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    object_detection_pkg = 'object_detection'
    tracking_pkg = 'tracking_control'
    obj_detection_package_path = get_package_share_directory(object_detection_pkg)
    tracking_package_path = get_package_share_directory(tracking_pkg)
    
    obj_detection_node = Node(
        package=object_detection_pkg,
        executable='color_obj_detection',
        name='color_obj_detection_node',
        parameters=[
            {'color_low': [120, 40, 0]},{'color_high': [255, 255, 70]}, {'object_size_min':200}
        ],
        output="screen"
    )

    goal_detection_node = Node(
        package=object_detection_pkg,
        executable='color_goal_detection',
        name='color_goal_detection_node',
        parameters=[
            {'color_low': [0, 0, 120]},{'color_high': [70, 70, 255]}, {'object_size_min':200}
        ],
        output="screen"
    )

    tracking_control_node = Node(
        package=tracking_pkg,
        executable='tracking_node',
        name='tracking_node',
        output="screen"
    )
    
    return LaunchDescription([
        obj_detection_node,
        goal_detection_node,
        tracking_control_node
    ])