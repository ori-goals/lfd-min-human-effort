#!/usr/bin/env python
import hsrb_interface
import rospy
import tf
import pickle
import copy
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
from learn_to_manipulate.utils import qv_rotate


class Controller(object):
    def __init__(self, sim):
        self.init_pose = {'x':0.3, 'y':0.0, 'z':0.461}
        self.init_ori = {'i':0.0, 'j':-1.7, 'k':3.14}
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.gripper = self.robot.get('gripper')
        self.steps_max = 50
        self.time_step = 0.1
        self.sim = sim
        self.max_pose_x = 0.78
        self.episode_count = 0
        rospy.Subscriber("fixed_laser/scan", LaserScan, self.store_laser)

    @classmethod
    def from_save_info(cls, sim, save_info):
        controller = cls(sim)
        controller.config = save_info['config']
        controller.exp = save_info['exp']
        return controller


    def get_save_info(self):
        return {'type':self.type, 'exp':self.exp, 'config':self.config}

    def set_arm_initial(self):
        self.whole_body.move_to_neutral()
        self.whole_body.linear_weight = 500
        self.whole_body.angular_weight = 500
        self.whole_body.end_effector_frame = 'hand_palm_link'
        self.whole_body.move_end_effector_pose([geometry.pose(x=0.35,y=0.1,z=0.9,ei=0.0, ej=0.0, ek=3.14)], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=self.init_pose['x'],y=self.init_pose['y'],z=self.init_pose['z'],
                                                ei=self.init_ori['i'], ej=self.init_ori['j'], ek=self.init_ori['k'])], ref_frame_id='map')
        self.whole_body.move_end_effector_pose([geometry.pose(x=self.init_pose['x'],y=self.init_pose['y'],z=self.init_pose['z'],
                                                ei=self.init_ori['i'], ej=self.init_ori['j'], ek=self.init_ori['k'])], ref_frame_id='map')
        self.pose = {'x':self.init_pose['x'], 'y':self.init_pose['y'], 'z':self.init_pose['z']}
        self.gripper.command(0.0)
        rospy.sleep(1.0)

    def begin_new_episode(self):
        pass

    def run_episode(self, case_name, case_number):
        print("Starting new episode with controller type: %s" % (self.type))
        self.begin_new_episode(case_name, case_number)
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
            result, episode_running = self.check_episode_status(step, next_pose)
            step += 1

        if result['success']:
            print('Episode succeeded.')
        else:
            print('Episode failed by %s\n' % (result['failure_mode']))
        episode = self.end_episode(result)
        return episode

    def get_next_pose(self, step, previous_pose):
        # if there is no movement return false
        state = self.get_state()
        delta = self.get_delta(step, state)
        if abs(delta['x']) < 0.0001 and abs(delta['y']) < 0.0001:
            return previous_pose, False

        self.exp.add_step(state, delta)
        pose = previous_pose
        pose['x'] += delta['x']
        pose['y'] += delta['y']
        return pose, True

    def end_episode(self, result):
        self.episode_count += 1
        episode = self.exp.end_episode(result)
        episode_length = len(self.exp.episode_df)
        self.update_learnt_controller(episode.result, episode_length)
        return episode

    def check_episode_status(self, step, pose):

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
            result = {'success':False, 'failure_mode':'falling'}
            episode_running = False
            return result, episode_running

        # if all of the block corners are in the goal region
        block_in_goal = True
        for corner in block_corners:
            corner = qv_rotate(block_q, corner)
            corner = np.array(corner) + np.array(block_centre)
            if corner[0] < self.sim.goal_centre_x - self.sim.goal_width_x/2.0:
                block_in_goal = False

        if block_in_goal:
            result = {'success':True, 'failure_mode':''}
            episode_running = False
            return result, episode_running

        # if the maximum x is exceeded
        if pose['x'] > self.max_pose_x:
            result = {'success':False, 'failure_mode':'max_x_exceeded'}
            episode_running = False
            return result, episode_running

        # if the maximum step count is exceeded
        if self.max_steps(step):
            result = {'success':False, 'failure_mode':'timeout'}
            episode_running = False
            return result, episode_running
        else:
            result = {'success':False, 'failure_mode':''}
            episode_running = True
            return result, episode_running


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
        self.type = 'hand_coded'

    def get_delta(self, step, state):
        dx = [0.04, 0.04, 0.04, 0.03, 0.05, 0.03, 0.05, 0.05, 0.05]
        dy = [0.0, 0.0, -0.10, 0.05, 0.05, 0.02, 0.0, -0.05, 0.0]
        return {'x':dx[step], 'y':dy[step]}

    def max_steps(self, step):
        if step > 7:
            return True
        else:
            return False

class TeleopController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.config = DemoConfig(bc_learning_rates = [0.001, 0.001],
                                bc_steps_per_frame = 10, td_max = 0.5)
        self.exp = Experience(window_size = float('inf'), prior_alpha = 0.3, prior_beta = 0.2, length_scale = 1.0)

    def begin_new_episode(self, case_name, case_number):
        confidence = 1.0
        sigma = 0.0
        self.exp.new_episode(confidence, sigma, case_name, case_number)

    def update_learnt_controller(self, result, episode_length):

        # we don't do a behaviour cloning update on unsuccessful episodes
        if not result:
            return

        learn_controller_exists = False
        for controller in self.sim.controllers:
            if controller.type == 'learnt':
                learnt_controller = controller
                learnt_controller_exists =  True
                break

        if learnt_controller_exists:
            learnt_controller.policy.bc_update(self.exp, self.config, episode_length)

class SavedTeleopController(TeleopController):
    def __init__(self, sim, file, type):
        TeleopController.__init__(self, sim)
        self.type = type
        self.load_saved_experience(file, type)

    def load_saved_experience(self, file, type):
        file = open(file,"rb")
        all_runs, controller_save_info = pickle.load(file)
        controller_found = False
        for save_info in controller_save_info:
            if save_info['type'] == type:
                experience = save_info['exp']
                controller_found = True

        if not controller_found:
            sys.exit('Controller type %s not found in file %s.' % (type, file))
        self.previous_experience = experience

    def run_episode(self, case_name, case_number):
        print("Starting new episode with saved controller type: %s" % (self.type))

        # find the appropriate episode in saved experience
        episode_found = False
        for episode in self.previous_experience.episode_list:
            if episode.case_name == case_name and episode.case_number == case_number:
                episode_found = True
                break

        if not episode_found:
            sys.exit('Saved episode number %s and name %s not found.' % (str(case_number), case_name))

        if episode.result:
            print('Episode succeeded.')
        else:
            print('Episode failed by %s\n' % (episode.failure_mode))
        self.end_episode(episode)
        return episode

    def end_episode(self, episode):
        self.episode_count += 1
        self.exp.add_saved_episode(episode)
        episode_length = len(episode.episode_df)
        self.update_learnt_controller(episode.result, episode_length)

class KeypadController(TeleopController):
    def __init__(self, sim):
        TeleopController.__init__(self, sim)
        self.type = 'keypad_teleop'
        rospy.Subscriber("key_vel", Twist, self.store_key_vel)
        self.key_vel = Twist()

    def store_key_vel(self, data):
        self.key_vel = data

    def get_delta(self, step, state):
        dx = self.time_step*self.key_vel.linear.x
        dy = self.time_step*self.key_vel.angular.z
        return {'x':dx, 'y':dy}

class LearntController(Controller):

    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.type = 'learnt'
        nominal_means = np.array([0.02, 0.0])
        nominal_sigma_exps = np.array([-5.0, -5.0])
        self.policy = ActorNN(nominal_means, nominal_sigma_exps)
        self.exp = Experience(window_size = 50, prior_alpha = 0.2, prior_beta = 0.3, length_scale = 1.0)
        self.config = LearntConfig(rl_buffer_frames_min = 200, ac_learning_rates = [0.00001, 0.00001],
                                rl_steps_per_frame = 5, td_max = 0.5)

    def begin_new_episode(self, case_name, case_number):
        confidence, sigma, _, _ = self.exp.get_state_value(self.get_state())
        self.exp.new_episode(confidence, sigma, case_name, case_number)

    def get_delta(self, step, state):
        action = self.policy.get_action(state)
        return {'x':action[0], 'y':action[1]}

    def update_learnt_controller(self, result, episode_length):
        if len(self.exp.replay_buffer) > self.config.rl_buffer_frames_min:
            self.policy.ac_update(self.exp, self.config, episode_length)

    def get_save_info(self):
        return {'type':self.type, 'exp':self.exp, 'config':self.config, 'policy':self.policy}

    @classmethod
    def from_save_info(cls, sim, save_info):
        controller = cls(sim)
        controller.config = save_info['config']
        controller.exp = save_info['exp']
        controller.policy = save_info['policy']
        return controller

class LearntConfig(object):
    def __init__(self, rl_buffer_frames_min, ac_learning_rates, rl_steps_per_frame, td_max):
        self.rl_buffer_frames_min = rl_buffer_frames_min
        self.ac_learning_rates = ac_learning_rates
        self.rl_steps_per_frame = rl_steps_per_frame
        self.td_max = td_max

class DemoConfig:
    def __init__(self, bc_learning_rates, bc_steps_per_frame, td_max):
        self.bc_learning_rates = bc_learning_rates
        self.bc_steps_per_frame = bc_steps_per_frame
        self.td_max = td_max
