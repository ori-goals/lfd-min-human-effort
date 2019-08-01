#!/usr/bin/env python
import hsrb_interface
import rospy
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
from hsrb_interface import geometry
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *

class SimulateManipulation(object):
    def __init__(self):
        rospy.init_node('learn_to_manipulate')
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')

    def run_new_episode(self):
        self.spawn_objects()

    def control_arm(self):
        pass

    def spawn_objects(self):

        # reset the location of the hsrb
        rospy.wait_for_service('/gazebo/set_model_state')
        set_model_state_prox = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState(model_name='hsrb', reference_frame='map')
        set_model_state_prox(state)
        self.whole_body.move_to_neutral()

        self.spawn_table()
        self.spawn_block()

    def spawn_table(self):
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_properties_prox = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world = get_world_properties_prox()

        if 'table' not in world.model_names:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "../models/table1/model.sdf")
            f = open(path,'r')
            sdf = f.read()

            initial_pose = Pose()
            initial_pose.position.x = 0.5
            initial_pose.position.y = 0.0
            initial_pose.position.z = 0.2

            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox("table", sdff, "simulation", initial_pose, "world")


    def spawn_block(self):
        pass
