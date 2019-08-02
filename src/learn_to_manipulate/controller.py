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

class Controller(object):
    def __init__(self):
        self.init_pose = {'x':0.4, 'y':0.0, 'z':0.46}
        self.init_ori = {'i':0.0, 'j':-1.7, 'k':3.14}
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')

    def set_arm_initial(self):
        self.whole_body.move_to_neutral()
        self.whole_body.linear_weight = 500
        self.whole_body.angular_weight = 500
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.4,y=0.1,z=0.9,ei=0.0, ej=0.0, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=self.init_pose['x'],y=self.init_pose['y'],z=self.init_pose['z'],
                                                ei=self.init_ori['i'], ej=self.init_ori['j'], ek=self.init_ori['k'])], ref_frame_id='map')

    def control_episode(self):
        episode_running = True
        step = 0
        previous_pose = self.init_pose
        while episode_running:
            next_pose = self.get_next_pose(step, previous_pose)
            self.execute_pose(next_pose)
            episode_running = self.check_episode_status(step)
            step += 1
            previous_pose = next_pose

    def check_episode_status(self, step):
        if step > 6:
            return False
        else:
            return True

    def execute_pose(self, pose):
        self.whole_body.linear_weight = 50
        self.whole_body.angular_weight = 50
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=pose['x'],y=pose['y'],z=pose['z'],
                                                ei=self.init_ori['i'], ej=self.init_ori['j'], ek=self.init_ori['k'])], ref_frame_id='map')

    def get_next_pose(self):
        pass


class HandCodedController(Controller):
    def __init__(self):
        Controller.__init__(self)
        self.controller_type = 'hand_coded'

    def get_next_pose(self, step, previous_pose):
        delta = self.get_delta(step)
        pose = previous_pose
        pose['x'] += delta['x']
        pose['y'] += delta['y']
        return pose

    def get_delta(self, step):
        dx = [0.04, 0.04, 0.04, 0.03, 0.05, 0.03, 0.05, 0.05]
        dy = [0.0, 0.0, -0.10, 0.05, 0.05, 0.05, 0.04, 0.04]
        return {'x':dx[step], 'y':dy[step]}
