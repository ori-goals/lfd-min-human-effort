#!/usr/bin/env python
# Copyright (C) 2016 Toyota Motor Corporation
import controller_manager_msgs.srv
import rospy
import trajectory_msgs.msg
import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *


rospy.init_node('test')

# initialize ROS publisher
pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command',
                      trajectory_msgs.msg.JointTrajectory, queue_size=10)

# wait to establish connection between the controller
while pub.get_num_connections() == 0:
    rospy.sleep(0.1)

# make sure the controller is running
rospy.wait_for_service('/hsrb/controller_manager/list_controllers')
list_controllers = (
    rospy.ServiceProxy('/hsrb/controller_manager/list_controllers',
                       controller_manager_msgs.srv.ListControllers))
running = False
while running is False:
    rospy.sleep(0.1)
    for c in list_controllers().controller:
        if c.name == 'arm_trajectory_controller' and c.state == 'running':
            running = True

# fill ROS message
traj = trajectory_msgs.msg.JointTrajectory()
traj.joint_names = ["arm_lift_joint", "arm_flex_joint",
                    "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
p = trajectory_msgs.msg.JointTrajectoryPoint()
p.positions = [0.2, -0.5, 0, 0, 0]
p.velocities = [0, 0, 0, 0, 0]
p.time_from_start = rospy.Time(3)
traj.points = [p]

# publish ROS message
#pub.publish(traj)
rospy.sleep(1)



##### spawn in urdf

initial_pose = Pose()
initial_pose.position.x = 0.5
initial_pose.position.y = 0.0
initial_pose.position.z = 0.2

f = open('/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/models/base_box_1/model.sdf','r')
sdff = f.read()

rospy.wait_for_service('gazebo/spawn_sdf_model')
spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
spawn_model_prox("another_name", sdff, "robotos_name_space", initial_pose, "world")



### get the location of the hsrb
rospy.wait_for_service('/gazebo/get_model_state')
get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
resp1 = get_model_state_prox('hsrb','')
print(resp1.pose)


### set the location of the hsrb
rospy.wait_for_service('/gazebo/set_model_state')
set_model_state_prox = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
state = ModelState(model_name='hsrb', reference_frame='map')
set_model_state_prox(state)
resp1 = get_model_state_prox('hsrb','')
print(resp1.pose)
