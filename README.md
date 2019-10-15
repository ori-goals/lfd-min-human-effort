# learn_to_manipulate

### Prerequisites

Gazebo 7 or higher, ROS Kinetic.

### Installation

1. Copy the learn_to_manipulate package into your catkin workspace.

2. In learn_to_manipulate/models there is a folder called hokuyo_mod which is modified laser model with a lower number of beams and limited arc. Gazebo needs to be able to find and load this model. The easiest way to achieve this is to copy the hokuyo_mod folder in this directory onto your computer: /home/username/.gazebo/models

3. Install packages for controlling the robot arm:

sudo apt-get install  ros-kinetic-ur5-moveit-config ros-kinetic-ur-kinematics

sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control

4. Copy the universal_robot folder from the kinetic-devel branch of my fork of it (https://github.com/marcrigter96/universal_robot) into your catkin workspace. This fork modifies the description of the robot to include a rectangle implement for pushing the block.

5. Build your catkin workspace.


### Run the simulation

roslaunch ur_gazebo ur5_joint_limited.launch  (launch gazebo)

roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true limited:=true (launch planner for moving robot arm joints)

rosrun learn_to_manipulate run_sim.py  (run a script which simulates episodes) 

roslaunch scitos_teleop teleop_joystick.launch js:=/dev/input/js0  (optional: use this to use input from a joystick)
