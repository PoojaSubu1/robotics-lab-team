from setuptools import setup
import os
from glob import glob

package_name = 'tracking_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yahboom',
    maintainer_email='eric565648@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = tracking_control.tracking_node:main',
            'joy_safety_ctrl = tracking_control.joy_safety_ctrl:main',
            'wheelodom_tf = tracking_control.wheelodom_tf:main',
        ],
    },
)
