from launch import LaunchDescription
from launch_ros.actions import Node 
# 参数声明与获取
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration
# 文件包含相关
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    cam_package_path = get_package_share_directory('astra_camera')
    yahboomcar_package_path = get_package_share_directory('yahboomcar_bringup')
    
    astra_camera_launch = IncludeLaunchDescription(XMLLaunchDescriptionSource(
        [os.path.join(cam_package_path, 'launch'),
         '/astro_pro_plus.launch.xml'])
    )
    yahboomcar_brinup_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        [os.path.join(yahboomcar_package_path, 'launch'),
         '/yahboomcar_bringup_X3_launch.py'])
    )
    
    return LaunchDescription([
        astra_camera_launch,
        yahboomcar_brinup_launch
    ])
