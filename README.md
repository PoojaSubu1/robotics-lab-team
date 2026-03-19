# S26_roboticsII
ROS2 Workspace for S26 RPI RoboticsII course.

## Fork the repository

Press the `Fork` on the page and fork the repo to your account. This will create a repository with exact same content in your github account.

## Cloning the repository 
On your machine, open a new terminal.
```
cd ~/codes
git clone https://github.com/YOUR_ACCOUNT/S26_roboticsII_ws.git [team_name]_ws
```

**Please see the instructions in the lab assignment document for full instructions on how to run this code and how to keep the files on the robot updated.** 

## Building the ROS2 workspace
The step will build and ROS2 workspace and compile the packages after successfully moving this workspace to the robot.

**Docker**: Open/access a docker container via a terminal (make sure to use ssh -Y)
Run a docker container
```
./run_docker.sh
```
or access (execute) a running docker container
```
./exec_docker.sh
```

Then run this command to build your work space.
```
cd ~/codes/[team_name]_ws
colcon build
```

**Important**: You have to compile the package **EVERYTIME** you modify the code, even the code is written in **PYTHON**. You can simply build one specific package using the following command. (So you don't have to compile ALL packages.)
```
colcon build --packages-select [package_name]
```
The [package_name] are for example, `object_detection`, `tracking_control`.

## Activate ROS2 environment
Activate ROS2 environment to run ROS software

**Important**: You have to activate ROS2 environment **EVERYTIME** you open a new terminal and run/access the docker container.

The reason you don't need to do this while doing previous assignments is that there's an ROS activation command in `/root/.bashrc` in the docker container. Whenever a terminal is launched, it will exected this bash script once.

**Docker**: Open/access a docker container via a terminal (SSH)
```
cd ~/codes/[team_name]_ws
source install/setup.bash
```

## Launch tracking nodes!

### Color Detection and Tracking Node
**Docker**: Open a terminal and access docker (via SSH). Remeber to **Activate ROS2 environment**.
```
ros2 launch tracking_control tracking_color_object_launch.py
```

### Teleoperation node
**Docker**: Open another terminal and access docker (via SSH). Remeber to **Activate ROS2 environment**. In this node, you can control the robot and activate/deactive tracking.
```
ros2 run tracking_control joy_safety_ctrl
```
### Launch the robot and camera
**Docker**: Open another terminal and access docker (via SSH).
For the old camera model (astra pro, Robot 1~6)
```
ros2 launch tracking_control car_camera_pro_bringup_launch.py
```
For the new camera model (astra pro plus, Robot 7~). **Unplug and plug** the camera cable if you haven't do so after booting up the robot. The image of the camera cable location is in Lab Assignmnet 1 description.
```
ros2 launch tracking_control car_camera_proplus_bringup_launch.py
```

## Robot Teleoeration
At the terminal that run the teleoperation node, the terminal should show this.
```
Control Your Robot!
Press and hold the keys to move around.
Press space key to turn on/off tracking.
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

t/y : turn counterclockwise/clockwise
k : force stop
space key : turn on/off tracking
anything else : stop smoothly

CTRL-C to quit
```

As the hint suggested, the following function can be done by press and/or hold the key.

- Activate/Deactivate autonomous tracking: press space key
- Moving linear in a direction: press and hold `u    i    o` (left forward, forward, right forward), `j    l` (left right), `m    ,    .` (left backward, backward, right backward)
- Turning counterclockwise/clockwise: `t`/`y`
- Stop the robot: press `k` or anything else

Note that the keyboard teleoperation has higher priority than the autonomous tracking command. In addition, deactivate (press space) the robot if you think the robot is (almost) out of control to protect the robot for potential damages.

## Acknowledgment and Summary of Modifications
This assignment is updated over the years by the course instructor Esen Yel, and course TAs Jainik Mehta, Alex Elias, and Jonathan Fried, based on the [original assignment](https://github.com/eric565648/S24_roboticsII_ws/tree/main/src/tracking_control) written by Chen Lung (Eric) Lu. 

The main files that are changed from the original:
- Addition of src/object_detection/object_detection/color_goal_detection.py
- update of src/object_detection/object_detection/color_obj_detection.py
- update of ‎src/tracking_control/tracking_control/tracking_node.py to include obstacle callback.
- update of src/tracking_control/launch/tracking_color_object_launch.py and setup.py to include goal detection node
