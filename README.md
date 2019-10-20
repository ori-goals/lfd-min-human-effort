# learn_to_manipulate

### Prerequisites

Gazebo 7 or higher, ROS Kinetic, pytorch.

### Installation

1. Clone the learn_to_manipulate package into your catkin workspace.

2. Install packages for controlling the robot arm:

sudo apt-get install  ros-kinetic-ur5-moveit-config ros-kinetic-ur-kinematics

sudo apt-get install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control

3. Clone the universal_robot folder from the kinetic-devel branch of my fork of it (https://github.com/marcrigter96/universal_robot) into your catkin workspace. This fork modifies the description of the robot to include a rectangle implement for pushing the block.

4. Add to your .bashrc: 

export GAZEBO_MODEL_PATH=/path_to_this_package/models:$GAZEBO_MODEL_PATH


5. Build your catkin workspace.


### Run the simulation

roslaunch ur_gazebo ur5_joint_limited.launch  (launch gazebo)

roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true limited:=true (launch planner for moving robot arm joints)

rosrun learn_to_manipulate run_sim.py  (run a script which simulates episodes) 

roslaunch scitos_teleop teleop_joystick.launch js:=/dev/input/js0  (optional: use this to use input from a joystick)
