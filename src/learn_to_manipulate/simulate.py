#!/usr/bin/env python
import rospy
import hsrb_interface
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
from hsrb_interface import geometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *

class Simulation(object):
    def __init__(self):
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.initial_pose_pub = rospy.Publisher('laser_2d_correct_pose', PoseWithCovarianceStamped, queue_size=10)
        self.block_width = 0.1
        self.goal_width_x = 0.2
        self.goal_centre_x = 0.9

    def run_new_episode(self, case_number):
        self.reset_hsrb()
        self.spawn_table()
        controller = self.choose_controller()
        controller.set_arm_initial()
        self.spawn_block()
        controller.run_episode(case_number)

    def choose_controller(self):
        return self.controllers[0]

    def reset_hsrb(self):
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_properties_prox = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world = get_world_properties_prox()

        if 'block' in world.model_names:
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model_prox('block')

        # reset the location of the hsrb
        self.whole_body.move_to_neutral()
        rospy.wait_for_service('/gazebo/set_model_state')
        set_model_state_prox = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState(model_name='hsrb', reference_frame='map')
        set_model_state_prox(state)
        self.whole_body.move_to_neutral()

        # reset estimated location
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'map'
        pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        pose.pose.pose.orientation.w = 1.0
        self.initial_pose_pub.publish(pose)

    def spawn_table(self):
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_properties_prox = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world = get_world_properties_prox()

        if 'table' not in world.model_names:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "../../models/table1/model.sdf")
            f = open(path,'r')
            table_sdf = f.read()

            table_initial_pose = Pose()
            table_initial_pose.position.x = 0.7
            table_initial_pose.position.y = 0.0
            table_initial_pose.position.z = 0.2

            path = os.path.join(my_path, "../../models/goal/model.sdf")
            f = open(path,'r')
            goal_sdf = f.read()

            self.goal_pose = Pose()
            self.goal_pose.position.x = self.goal_centre_x
            self.goal_pose.position.y = 0.0
            self.goal_pose.position.z = 0.40

            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox("table", table_sdf, "simulation", table_initial_pose, "world")
            spawn_model_prox("goal", goal_sdf, "simulation", self.goal_pose, "world")

    def spawn_block(self):
        block_name = 'block_30'
        self.block_length = float("0." + block_name.split('_')[1])
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "../../models/" + block_name + "/model.sdf")
        f = open(path,'r')
        sdf = f.read()

        initial_pose = Pose()
        initial_pose.position.x = 0.55
        initial_pose.position.y = 0.0
        initial_pose.position.z = 0.6

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox("block", sdf, "simulation", initial_pose, "world")
