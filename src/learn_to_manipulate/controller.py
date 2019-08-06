#!/usr/bin/env python
import hsrb_interface
import rospy
import tf
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
import numpy as np
from hsrb_interface import geometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from learn_to_manipulate.actor_nn import ActorNN
from learn_to_manipulate.experience import Experience


def qv_rotate(q1, v1):
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2),
        tf.transformations.quaternion_conjugate(q1))[0:3]

class Controller(object):
    def __init__(self, sim):
        self.init_pose = {'x':0.4, 'y':0.0, 'z':0.453}
        self.init_ori = {'i':0.0, 'j':-1.7, 'k':3.14}
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.steps_max = 50
        self.time_step = 0.1
        self.sim = sim
        rospy.Subscriber("fixed_laser/scan", LaserScan, self.store_laser)


    def set_arm_initial(self):
        self.whole_body.move_to_neutral()
        self.whole_body.linear_weight = 500
        self.whole_body.angular_weight = 500
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.4,y=0.1,z=0.9,ei=0.0, ej=0.0, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=self.init_pose['x'],y=self.init_pose['y'],z=self.init_pose['z'],
                                                ei=self.init_ori['i'], ej=self.init_ori['j'], ek=self.init_ori['k'])], ref_frame_id='map')
        self.pose = {'x':self.init_pose['x'], 'y':self.init_pose['y'], 'z':self.init_pose['z']}

    def begin_new_episode(self):
        pass

    def run_episode(self, case_number):
        self.begin_new_episode(case_number)
        episode_running = True
        step = 0
        while episode_running:
            next_pose, moved = self.get_next_pose(step, self.pose)

            # if there is no movement sleep and try again
            if not moved:
                rospy.sleep(0.1)
                continue
            self.execute_pose(next_pose)
            self.pose = next_pose
            result, episode_running = self.check_episode_status(step)
            step += 1
        self.end_episode(result)
        return result

    def get_next_pose(self, step, previous_pose):

        # if there is no movement return false
        delta = self.get_delta(step)
        if abs(delta['x']) < 0.0001 and abs(delta['y']) < 0.0001:
            return previous_pose, False

        pose = previous_pose
        pose['x'] += delta['x']
        pose['y'] += delta['y']
        return pose, True

    def end_episode(self):
        pass

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

    def store_laser(self, data):
        scan = np.array(data.ranges)
        range_max = 5.0
        indexes = scan > range_max
        scan[indexes] = range_max
        self.current_laser_scan = scan

    def get_state(self):
        return self.current_laser_scan.tolist() + [self.pose['x'], self.pose['y']]

    def execute_pose(self, pose):
        self.whole_body.linear_weight = 50
        self.whole_body.angular_weight = 50
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=pose['x'],y=pose['y'],z=pose['z'],
                                                ei=self.init_ori['i'], ej=self.init_ori['j'], ek=self.init_ori['k'])], ref_frame_id='map')


class HandCodedController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.controller_type = 'hand_coded'

    def get_delta(self, step):
        dx = [0.04, 0.04, 0.04, 0.03, 0.05, 0.03, 0.05, 0.05, 0.05]
        dy = [0.0, 0.0, -0.10, 0.05, 0.05, 0.02, 0.0, -0.05, 0.0]
        return {'x':dx[step], 'y':dy[step]}

    def max_steps(self, step):
        if step > 7:
            return True
        else:
            return False

class KeypadController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.controller_type = 'key_teleop'
        rospy.Subscriber("key_vel", Twist, self.store_key_vel)
        self.key_vel = Twist()

    def store_key_vel(self, data):
        self.key_vel = data

    def get_delta(self, step):
        dx = self.time_step*self.key_vel.linear.x
        dy = self.time_step*self.key_vel.angular.z
        return {'x':dx, 'y':dy}


class LearntController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.controller_type = 'learnt'
        nominal_means = np.array([0.02, 0.0])
        nominal_sigma_exps = np.array([-5.5, -5.5])
        self.policy = ActorNN(nominal_means, nominal_sigma_exps)
        self.exp = Experience(window_size = 50)

    def get_confidence(self):
        confidence = {'mean':0.5, 'sigma':0.2}
        return confidence

    def begin_new_episode(self, case_number):
        confidence = self.get_confidence()
        self.exp.new_episode(confidence, case_number)

    def end_episode(self, result):
        self.exp.end_episode(result)
        self.update_controller()

    def get_next_pose(self, step, pose):
        state = self.get_state()
        delta = self.get_delta(step, state)
        self.exp.add_step(state, delta)

        pose['x'] += delta['x']
        pose['y'] += delta['y']
        return pose, True

    def get_delta(self, step, state):
        action = self.policy.get_action(state)
        return {'x':action[0], 'y':action[1]}

    def update_controller(self):
        pass
