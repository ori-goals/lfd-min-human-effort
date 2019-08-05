#!/usr/bin/env python
import hsrb_interface
import rospy
import tf
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
import numpy as np
from hsrb_interface import geometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *


def qv_rotate(q1, v1):
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2),
        tf.transformations.quaternion_conjugate(q1))[0:3]

class Controller(object):
    def __init__(self, sim):
        self.init_pose = {'x':0.4, 'y':0.0, 'z':0.46}
        self.init_ori = {'i':0.0, 'j':-1.7, 'k':3.14}
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.steps_max = 30
        self.sim = sim

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
            episode_success, episode_running = self.check_episode_status(step)
            step += 1
            previous_pose = next_pose

    def check_episode_status(self, step):

        episode_running = True

        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        block_pose = get_model_state_prox('block','').pose

        block_corners = [[self.sim.block_width/2.0, self.sim.block_length/2.0, 0.0],
                        [self.sim.block_width/2.0, -self.sim.block_length/2.0, 0.0],
                        [-self.sim.block_width/2.0, self.sim.block_length/2.0, 0.0],
                        [-self.sim.block_width/2.0, -self.sim.block_length/2.0, 0.0]]
        block_q = [block_pose.orientation.x, block_pose.orientation.y,
                    block_pose.orientation.z, block_pose.orientation.w]
        block_centre = [block_pose.position.x, block_pose.position.y,
                        block_pose.position.z]

        # if the block has fallen off the table
        if block_pose.position.z < 0.3:
            episode_success = False
            episode_running = False
            return episode_success, episode_running

        # if all of the block corners are in the goal region
        block_in_goal = True
        for corner in block_corners:
            corner = qv_rotate(block_q, corner)
            corner = np.array(corner) + np.array(block_centre)
            if corner[0] < self.sim.goal_centre_x - self.sim.goal_width_x/2.0:
                block_in_goal = False

        if block_in_goal:
            print('block in goal!')
            episode_success = True
            episode_running = False
            return episode_success, episode_running

        # if the maximum step count is exceeded
        if self.max_steps(step):
            episode_success = False
            episode_running = False
            return episode_success, episode_running
        else:
            episode_success = False
            episode_running = True
            return episode_success, episode_running


    def max_steps(self, step):
        if step >= self.steps_max:
            return True
        else:
            return False

    def execute_pose(self, pose):
        self.whole_body.linear_weight = 50
        self.whole_body.angular_weight = 50
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=pose['x'],y=pose['y'],z=pose['z'],
                                                ei=self.init_ori['i'], ej=self.init_ori['j'], ek=self.init_ori['k'])], ref_frame_id='map')

    def get_next_pose(self):
        pass


class HandCodedController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.controller_type = 'hand_coded'

    def get_next_pose(self, step, previous_pose):
        delta = self.get_delta(step)
        pose = previous_pose
        pose['x'] += delta['x']
        pose['y'] += delta['y']
        return pose

    def get_delta(self, step):
        dx = [0.04, 0.04, 0.04, 0.03, 0.05, 0.03, 0.05, 0.05, 0.05]
        dy = [0.0, 0.0, -0.10, 0.05, 0.05, 0.02, 0.0, -0.05, 0.0]
        return {'x':dx[step], 'y':dy[step]}

    def max_steps(self, step):
        if step > 7:
            return True
        else:
            return False