#!/usr/bin/env python
import hsrb_interface
import rospy
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
from hsrb_interface import geometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *

class Simulate(object):
    def __init__(self):
        rospy.init_node('learn_to_manipulate')
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.initial_pose_pub = rospy.Publisher('laser_2d_correct_pose', PoseWithCovarianceStamped, queue_size=10)

    def run_new_episode(self):
        self.reset_hsrb()
        self.spawn_table()
        self.set_arm_initial()
        self.spawn_block()
        self.control_arm()

    def set_arm_initial(self):
        self.whole_body.move_to_neutral()
        self.whole_body.linear_weight = 500
        self.whole_body.angular_weight = 500
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.4,y=0.1,z=0.9,ei=0.0, ej=0.0, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.4,y=0.0,z=0.46,ei=0.0, ej=-1.7, ek=3.14)], ref_frame_id='map')


    def control_arm(self):
        self.whole_body.linear_weight = 50
        self.whole_body.angular_weight = 50
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.45,y=0.0,z=0.46,ei=0.0, ej=-1.7, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.5,y=0.0,z=0.46,ei=0.0, ej=-1.7, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.6,y=0.1,z=0.46,ei=0.0, ej=-1.7, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.65,y=0.0,z=0.46,ei=0.0, ej=-1.7, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.75,y=-0.05,z=0.46,ei=0.0, ej=-1.7, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.85,y=-0.05,z=0.46,ei=0.0, ej=-1.7, ek=3.14)], ref_frame_id='map')

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
            sdf = f.read()

            initial_pose = Pose()
            initial_pose.position.x = 0.7
            initial_pose.position.y = 0.0
            initial_pose.position.z = 0.2

            rospy.wait_for_service('gazebo/spawn_sdf_model')
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox("table", sdf, "simulation", initial_pose, "world")


    def spawn_block(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "../../models/block_30/model.sdf")
        f = open(path,'r')
        sdf = f.read()

        initial_pose = Pose()
        initial_pose.position.x = 0.55
        initial_pose.position.y = 0.0
        initial_pose.position.z = 0.6

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox("block", sdf, "simulation", initial_pose, "world")
