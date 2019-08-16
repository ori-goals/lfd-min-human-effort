# learn_to_manipulate

TODO: Explain ur_gazebo arm install, and the mod to add implement
TODO: Explain modified Hokuyo laser and where to put this.
TODO: Explain how to reconfigure a different script to run
TODO: Should I recommend Gazebo 9? Had block spawn problem in 7.

roslaunch ur_gazebo ur5_joint_limited.launch  

roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true limited:=true 

rosrun learn_to_manipulate script_to_run 

roslaunch scitos_teleop teleop_joystick.launch js:=/dev/input/js0 
