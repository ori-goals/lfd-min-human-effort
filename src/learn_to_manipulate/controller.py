#!/usr/bin/env python
import hsrb_interface
import rospy
import tf
import pickle
import copy
import torch
import controller_manager_msgs.srv
import trajectory_msgs.msg
import os.path
import numpy as np
import moveit_commander
import torch.optim as opt
import moveit_msgs.msg
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from sensor_msgs.msg import Joy
from torch import nn
from learn_to_manipulate.actor_nn import ActorNN
from learn_to_manipulate.experience import Experience
from learn_to_manipulate.utils import qv_rotate
from learn_to_manipulate.ddpg import DDPGAgent
from utils import *
from matplotlib import pyplot as plt
from learn_to_manipulate.ddpg_models import Critic, Actor
from roscpp.srv import SetLoggerLevel, GetLoggers
import rosnode


def shut_up_commander():
    print('\n')
    node_names = rosnode.get_node_names()
    for name in node_names:
        if 'move_group_commander_wrappers' in name:
            try:
                get_logger_prox = rospy.ServiceProxy(name+'/get_loggers', GetLoggers)
                resp = get_logger_prox()
                set_logger_prox = rospy.ServiceProxy(name+'/set_logger_level', SetLoggerLevel)
                for logger in resp.loggers:
                    set_logger_prox(logger.name, 'error')
            except:
                pass


class Controller(object):
    def __init__(self, sim):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        self.steps_max = 50
        self.time_step = 0.05
        self.sim = sim
        self.num_states = 32
        self.num_actions = 2
        self.pose_limits = {'min_x':0.25, 'max_x':0.75, 'min_y':-0.25, 'max_y':0.25}
        self.episode_number = 0
        self.init_pose = {'x':0.29, 'y':0.0, 'z':0.37, 'qx':0.0, 'qy':0.6697, 'qz':0.0, 'qw':0.7426}
        rospy.Subscriber("fixed_laser/scan", LaserScan, self.store_laser)
        self.actions_high = np.array([0.04, 0.03])
        self.actions_low = np.array([-0.02, -0.03])
        laser_low = np.zeros(30)
        laser_high = np.ones(30)
        self.states_high = np.concatenate((laser_high, np.array([1.0, 0.3])))
        self.states_low = np.concatenate((laser_low, np.array([0.0, -0.3])))
        shut_up_commander()

    @classmethod
    def from_save_info(cls, sim, save_info):
        controller = cls(sim)
        controller.config = save_info['config']
        controller.experience = save_info['experience']
        return controller

    def get_save_info(self):
        return {'type':self.type, 'experience':self.experience, 'config':self.config}

    def set_arm_initial(self):
        move_arm_initial(self)

    def run_episode(self, case_name, case_number):
        print("Starting episode %d with controller type: %s" % (self.sim.episode_number, self.type))
        self.current_pose = copy.copy(self.init_pose)
        self.begin_new_episode(case_name, case_number)
        episode_running = True
        dense_reward = 0.0
        step = 0
        if self.type == 'ddpg':
            self.agent.noise.reset()

        while episode_running:
            state = self.get_state()
            action = self.get_action(state, step)
            if abs(action[0]) < 0.001 and abs(action[1]) < 0.001:
                rospy.sleep(0.1)
                continue
            new_state, reward, result, episode_running = self.execute_action(action, step)
            self.add_to_memory(state, action, reward, new_state, not episode_running)
            self.update_agent()
            dense_reward += reward
            step += 1
        episode = self.end_episode(result, dense_reward, step)
        return episode, dense_reward

    def add_to_memory(self, state, action, reward, new_state, terminal):
        new_state_norm = self.to_normalised_state(new_state)
        state_norm = self.to_normalised_state(state)
        action_norm = self.to_normalised_action(action)
        self.replay_buffer.add(state_norm, action_norm, reward, terminal, new_state_norm)
        self.experience.add_step(state, action, reward, terminal, new_state)

    def begin_new_episode(self, case_name, case_number):
        confidence, sigma = self.get_controller_confidence()
        self.experience.new_episode(confidence, sigma, case_name, case_number)

    def end_episode(self, result, dense_reward, step):
        episode = self.experience.end_episode(result, self.type, dense_reward)
        self.print_result(result, dense_reward)
        if self.type == 'ddpg':
            self.agent.add_plotting_data(dense_reward, step, self.episode_number)
        self.episode_number += 1
        return episode

    def print_result(self, result, dense_reward):
        if result['success']:
            print('Episode succeeded. The dense reward was %.4f\n' % (dense_reward))
        else:
            print('Episode failed by %s. The dense reward was %.4f\n' % (result['failure_mode'], dense_reward))

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
        range_max = 1.0
        indexes = scan > range_max
        scan[indexes] = range_max
        self.current_laser_scan = scan

    def get_state(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        block_pose = get_model_state_prox('block','').pose
        blockx = block_pose.position.x
        blocky = block_pose.position.y
        xdiff = blockx - self.current_pose['x']
        ydiff = blocky - self.current_pose['y']
        block_angle = np.arccos(block_pose.orientation.w)*2
        if block_pose.orientation.z < 0:
            block_angle *= -1.
        return np.array(self.current_laser_scan.tolist() + [self.current_pose['x'], self.current_pose['y']])

    def execute_action(self, action, step):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        old_block_pose = get_model_state_prox('block','').pose

        old_arm_pose = copy.copy(self.current_pose)
        self.current_pose['x'] += action[0]
        self.current_pose['y'] += action[1]
        if self.current_pose['x'] > self.pose_limits['max_x']:
            self.current_pose['x'] = self.pose_limits['max_x']
        if self.current_pose['x'] < self.pose_limits['min_x']:
            self.current_pose['x'] = self.pose_limits['min_x']
        if self.current_pose['y'] > self.pose_limits['max_y']:
            self.current_pose['y'] = self.pose_limits['max_y']
        if self.current_pose['y'] < self.pose_limits['min_y']:
            self.current_pose['y'] = self.pose_limits['min_y']
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = self.current_pose['qx']
        pose_goal.orientation.y = self.current_pose['qy']
        pose_goal.orientation.z = self.current_pose['qz']
        pose_goal.orientation.w = self.current_pose['qw']
        pose_goal.position.x = self.current_pose['x']
        pose_goal.position.y = self.current_pose['y']
        pose_goal.position.z = self.current_pose['z']
        waypoints = [pose_goal]
        plan, fraction = self.group.compute_cartesian_path(waypoints,  0.005, 0.0)
        self.group.execute(plan, wait=True)

        new_block_pose = get_model_state_prox('block','').pose
        new_state = self.get_state()
        result, episode_running = self.check_episode_status(step)
        reward = self.get_dense_reward(old_block_pose, new_block_pose, result)
        return new_state, reward, result, episode_running


    def get_dense_reward(self, old_block_pose, new_block_pose, result):
        targ = np.array([self.sim.goal_centre_x + 10.0, 0.0])
        old = np.array([old_block_pose.position.x, old_block_pose.position.y])
        new = np.array([new_block_pose.position.x, new_block_pose.position.y])
        old_dist = np.linalg.norm(targ - old)
        new_dist = np.linalg.norm(targ - new)
        reward = (old_dist - new_dist)*5.0

        if result['success']:
            reward += 1.0
        return reward

    def go_to_pose(self, pose):
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = pose['qx']
        pose_goal.orientation.y = pose['qy']
        pose_goal.orientation.z = pose['qz']
        pose_goal.orientation.w = pose['qw']
        pose_goal.position.x = pose['x']
        pose_goal.position.y = pose['y']
        pose_goal.position.z = pose['z']
        waypoints = [pose_goal]
        plan, fraction = self.group.compute_cartesian_path(waypoints,  0.005, 0.0)
        self.group.execute(plan, wait=True)

    def get_controller_confidence(self):
        confidence, sigma, _, _ = self.experience.get_state_value(self.get_state())
        return confidence, sigma

    def to_state(self, state):
        state_k = (self.states_high - self.states_low)/ 2.
        state_b = (self.states_high + self.states_low)/ 2.
        return state_k * state + state_b

    def to_action(self, action):
        act_k = (self.actions_high - self.actions_low)/ 2.
        act_b = (self.actions_high + self.actions_low)/ 2.
        return act_k * action + act_b

    def to_normalised_state(self, state):
        state_k_inv = 2./(self.states_high - self.states_low)
        state_b = (self.states_high + self.states_low)/ 2.
        return state_k_inv * (state - state_b)

    def to_normalised_action(self, action):
        act_k_inv = 2./(self.actions_high - self.actions_low)
        act_b = (self.actions_high + self.actions_low)/ 2.
        return act_k_inv * (action - act_b)


class TeleopController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.config = []
        self.experience = Experience(window_size = float('inf'), prior_alpha = 0.3, prior_beta = 0.2, length_scale = 2.0)
        self.replay_buffer = ReplayBuffer(10000)
        self.checked_for_rl = False
        self.rl_controller = None

    def get_controller_confidence(self):
        return 1.0, 0.0

    def update_agent(self):
        if not self.checked_for_rl:
            self.check_for_rl()

        # if we have teleop controller with a large enough buffer we can add these updates
        rl_buffer = None
        if (self.rl_controller is not None):
            if self.rl_controller.replay_buffer.count( ) > self.rl_controller.config.min_buffer_size:
                rl_buffer = self.rl_controller.replay_buffer

            if self.replay_buffer.count() > self.rl_controller.config.demo_min_buffer_size:
                self.rl_controller.agent.update(rl_buffer, self.replay_buffer)

    def check_for_rl(self):
        for contr_type in self.sim.controllers.keys():
            if 'ddpg' == contr_type:
                self.rl_controller = self.sim.controllers[contr_type]


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
                experience = save_info['experience']
                controller_found = True

        if not controller_found:
            sys.exit('Controller type %s not found in file %s.' % (type, file))
        self.previous_experience = experience

    def run_episode(self, case_name, case_number):
        print("Starting episode %d with saved controller type: %s" % (self.sim.episode_number, self.type))
        self.begin_new_episode(case_name, case_number)

        # find the appropriate episode in saved experience
        episode_found = False
        for episode in self.previous_experience.episode_list:
            if episode.case_name == case_name and episode.case_number == case_number:
                episode_found = True
                break
        if not episode_found:
            sys.exit('Saved episode number %s and name %s not found.' % (str(case_number), case_name))

        # loop through episode steps
        dense_reward = 0.0
        step = 0
        for index, row in episode.episode_df.iterrows():
            self.add_to_memory(row['state'], row['action'], row['dense_reward'], row['next_state'], row['terminal'])
            dense_reward += row['dense_reward']
            step += 1
        result = {'success':episode.result, 'failure_mode':episode.failure_mode}
        episode = self.end_episode(result, dense_reward, step)
        return episode, dense_reward


class KeypadController(TeleopController):
    def __init__(self, sim):
        TeleopController.__init__(self, sim)
        self.type = 'keypad_teleop'
        rospy.Subscriber("key_vel", Twist, self.store_key_vel)
        self.key_vel = Twist()

    def store_key_vel(self, data):
        self.key_vel = data

    def get_action(self, state, step):
        dx = self.time_step*self.key_vel.linear.x
        dy = self.time_step*self.key_vel.angular.z
        return [dx, dy]

class JoystickController(TeleopController):
    def __init__(self, sim):
        TeleopController.__init__(self, sim)
        self.type = 'joystick_teleop'
        rospy.Subscriber("/teleop_joystick/joy", Joy, self.store_gamepad_vel)
        self.velx = 0.0
        self.vely = 0.0
        self.max_vel = 0.5
        self.time_received = rospy.get_rostime().to_sec()

    def store_gamepad_vel(self, data):
        self.velx = data.axes[4]*self.max_vel
        self.vely = data.axes[3]*self.max_vel
        self.time_received = rospy.get_rostime().to_sec()

    def get_action(self, state, step):
        dx = self.time_step*self.velx
        dy = self.time_step*self.vely
        time_now = rospy.get_rostime().to_sec()
        if abs(time_now - self.time_received) < 0.1:
            return  [dx, dy]
        else:
            return [0.0, 0.0]

class DDPGController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.type = 'ddpg'
        self.experience = Experience(window_size = 50, prior_alpha = 0.2, prior_beta = 0.3, length_scale = 2.0)
        self.config = DDPGConfig(lr_critic=0.001, lr_actor=0.0001, lr_bc=0.001, rl_batch_size=128, demo_batch_size=64,
                                min_buffer_size=256, tau=0.001, gamma=0.99, noise_factor=0.4, buffer_size=50000,
                                demo_min_buffer_size=128, q_filter_epsilon=0.02)
        self.agent = DDPGAgent(self.config, self.num_states, self.num_actions)
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.checked_for_teleop = False
        self.teleop_controller = None

    def get_controller_confidence(self):
        return 1.0, 0.0

    def get_action(self, state, step):
        state_norm = self.to_normalised_state(state)
        action_norm = self.agent.actor.get_action(state_norm)
        action_norm += self.agent.noise()*self.agent.noise_factor
        action_norm = np.clip(action_norm, -1., 1.)
        action = self.to_action(action_norm)
        return action

    def check_for_teleop(self):
        for contr_type in self.sim.controllers.keys():
            if 'teleop' in contr_type:
                self.teleop_controller = self.sim.controllers[contr_type]

    def update_agent(self):
        if not self.checked_for_teleop:
            self.check_for_teleop()

        # if we have teleop controller with a large enough buffer we can add these updates
        teleop_buffer = None
        if (self.teleop_controller is not None):
            if self.teleop_controller.replay_buffer.count( ) > self.config.demo_min_buffer_size:
                teleop_buffer = self.teleop_controller.replay_buffer

        if self.replay_buffer.count() > self.config.min_buffer_size:
            self.agent.update(self.replay_buffer, teleop_buffer)

    def get_save_info(self):
        return {'type':self.type, 'experience':self.experience, 'config':self.config, 'agent':self.agent}

class SavedDDPGAgent(Controller):
    def __init__(self, sim, file):
        Controller.__init__(self, sim)
        self.type = 'baseline'
        self.experience = Experience(window_size = float('inf'), prior_alpha = 0.2, prior_beta = 0.3, length_scale = 2.0)
        self.config = []
        self.baseline_agent = self.load_saved_agent(file)
        self.checked_for_controllers = False
        self.rl_controller = None
        self.teleop_controller = None

    def run_episode(self, case_name, case_number):
        print("Starting episode %d with controller type: %s" % (self.sim.episode_number, self.type))
        self.current_pose = copy.copy(self.init_pose)
        self.begin_new_episode(case_name, case_number)
        episode_running = True
        dense_reward = 0.0
        step = 0

        while episode_running:
            state = self.get_state()
            action = self.get_action(state, step)
            new_state, reward, result, episode_running = self.execute_action(action, step)
            self.add_to_experience(state, action, reward, new_state, not episode_running)
            dense_reward += reward
            step += 1
        episode = self.end_episode(result, dense_reward, step)


        if not self.checked_for_controllers:
            self.check_for_controllers()
            self.checked_for_controllers = True

        # if the episode was successful add each step to the demonstration replay buffer
        # only if there is a ddpg agent to train and a teleop controller providing
        # other demonstrations
        if episode.result and (self.rl_controller is not None) and (self.teleop_controller is not None):
            for index, row in episode.episode_df.iterrows():
                self.add_to_replay_buffer(row['state'], row['action'], row['dense_reward'], row['next_state'], row['terminal'])
                self.update_agent()
        return episode, dense_reward

    def add_to_experience(self, state, action, reward, new_state, terminal):
        self.experience.add_step(state, action, reward, terminal, new_state)

    def update_agent(self):

        # if we have teleop controller with a large enough buffer we can add these updates
        rl_buffer = None
        if self.rl_controller.replay_buffer.count() > self.rl_controller.config.min_buffer_size:
            rl_buffer = self.rl_controller.replay_buffer

        if self.teleop_controller.replay_buffer.count() > self.rl_controller.config.demo_min_buffer_size:
            self.rl_controller.agent.update(rl_buffer, self.telelop_controller.replay_buffer)

    def add_to_replay_buffer(self, state, action, reward, new_state, terminal):
        '''
        Adds these steps to the replay buffer of the teleoop controller.
        '''
        new_state_norm = self.to_normalised_state(new_state)
        state_norm = self.to_normalised_state(state)
        action_norm = self.to_normalised_action(action)
        self.telelop_controller.replay_buffer.add(state_norm, action_norm, reward, terminal, new_state_norm)


    def load_saved_agent(self, file):
        file = open(file,"rb")
        all_runs, controller_save_info = pickle.load(file)
        controller_found = False
        for save_info in controller_save_info:
            if save_info['type'] == 'ddpg':
                agent  = save_info['agent']
                controller_found = True

        if not controller_found:
            sys.exit('Controller type %s not found in file %s.' % (type, file))
        return agent

    def get_action(self, state, step):
        state_norm = self.to_normalised_state(state)
        action_norm = self.baseline_agent.actor.get_action(state_norm)
        action_norm = np.clip(action_norm, -1., 1.)
        action = self.to_action(action_norm)
        return action

    def get_save_info(self):
        return {'type':self.type, 'experience':self.experience, 'agent':self.baseline_agent}

    def check_for_controllers(self):
        for contr_type in self.sim.controllers.keys():
            if 'teleop' in contr_type:
                self.teleop_controller = self.sim.controllers[contr_type]
            elif 'ddpg' == contr_type:
                self.rl_controller = self.sim.controllers[contr_type]

class DDPGConfig:
    def __init__(self, lr_critic, lr_actor, lr_bc, rl_batch_size, demo_batch_size, min_buffer_size, tau, gamma, noise_factor, buffer_size, demo_min_buffer_size, q_filter_epsilon):
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.lr_bc = lr_bc
        self.rl_batch_size = rl_batch_size
        self.demo_batch_size = demo_batch_size
        self.min_buffer_size = min_buffer_size
        self.tau = tau
        self.gamma = gamma
        self.noise_factor = noise_factor
        self.buffer_size = buffer_size
        self.demo_min_buffer_size = demo_min_buffer_size
        self.q_filter_epsilon = q_filter_epsilon
