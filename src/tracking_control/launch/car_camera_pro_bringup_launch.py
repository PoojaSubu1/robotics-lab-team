from launch import LaunchDescription
from launch_ros.actions import Node 
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
         '/astra_pro.launch.xml'])
    )
    yahboomcar_brinup_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        [os.path.join(yahboomcar_package_path, 'launch'),
         '/yahboomcar_bringup_X3_launch.py'])
    )
    
    return LaunchDescription([
        astra_camera_launch,
        yahboomcar_brinup_launch
    ])
