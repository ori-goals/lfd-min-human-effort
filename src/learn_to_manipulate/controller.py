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
from torch import nn
from learn_to_manipulate.actor_nn import ActorNN
from learn_to_manipulate.experience import Experience
from learn_to_manipulate.utils import qv_rotate
from learn_to_manipulate.ddpg import DDPGagent
from utils import *
from matplotlib import pyplot as plt
from learn_to_manipulate.ddpg_models import Critic, Actor

class Controller(object):
    def __init__(self, sim):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        self.steps_max = 60
        self.time_step = 0.05
        self.sim = sim
        self.num_states = 52
        self.num_actions = 2
        self.pose_limits = {'min_x':0.25, 'max_x':0.75, 'min_y':-0.25, 'max_y':0.25}
        self.episode_count = 0
        self.init_pose = {'x':0.29, 'y':0.0, 'z':0.37, 'qx':0.0, 'qy':0.6697, 'qz':0.0, 'qw':0.7426}
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
        qxs = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, self.init_pose['qx']]
        qys = [0.0, 0.0, 0.0, 0.0, 0.0, 0.6697, 0.6697, self.init_pose['qy']]
        qzs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.init_pose['qz']]
        qws = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7426, 0.7426, self.init_pose['qw']]
        xs = [0.816, 0.70, 0.7, 0.7, 0.7, 0.77, 0.77, self.init_pose['x']]
        ys = [0.191, 0.20, 0.199, 0.2, 0.2, 0.2, 0.0, self.init_pose['y']]
        zs = [0.0, 0.12, 0.253, 0.35, 0.35, 0.35, 0.37, self.init_pose['z']]
        current_pose = self.group.get_current_pose().pose
        if current_pose.position.z > 0.3:
            start_ind = len(qxs) - 1
        else:
            start_ind = 0

        for ind in range(start_ind, len(qxs)):
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.orientation.x = qxs[ind]
            pose_goal.orientation.y = qys[ind]
            pose_goal.orientation.z = qzs[ind]
            pose_goal.orientation.w = qws[ind]
            pose_goal.position.x = xs[ind]
            pose_goal.position.y = ys[ind]
            pose_goal.position.z = zs[ind]
            if ind < len(qxs) - 1:
                self.group.set_pose_target(pose_goal)
                plan = self.group.go(wait=True)
                self.group.stop()
                self.group.clear_pose_targets()
            else:
                self.go_to_pose(self.init_pose)
        rospy.sleep(1.0)

    def run_episode(self, case_name, case_number):
        print("Starting new episode with controller type: %s" % (self.type))
        self.current_pose = copy.copy(self.init_pose)
        self.begin_new_episode(case_name, case_number)

        episode_running = True
        episode_reward = 0.0
        step = 0
        while episode_running:
            state = self.get_state()
            action = self.get_action(state, step)
            if abs(action['x']) < 0.0001 and abs(action['y']) < 0.0001:
                rospy.sleep(0.1)

            new_state, reward, result, episode_running = self.execute_action(action, step)
            self.add_to_memory(state, action, reward, new_state, result)

            episode_reward += reward
            step += 1

        if result['success']:
            print('Episode succeeded.')
        else:
            print('Episode failed by %s\n' % (result['failure_mode']))
        episode = self.end_episode(result)
        print(episode_reward)
        return episode, episode_reward

    def add_to_memory(self, state, action, reward, new_state, result):
        self.exp.add_step(state, action)

    def begin_new_episode(self, case_name, case_number):

        if self.type == 'ddpg':
            pass
        else:
            confidence, sigma = self.get_controller_confidence()
            self.exp.new_episode(confidence, sigma, case_name, case_number)

    def end_episode(self, result):
        self.episode_count += 1
        if self.type == 'ddpg':
            episode = 5
        else:
            episode = self.exp.end_episode(result)
            episode_length = len(self.exp.episode_df)
        #self.update_learnt_controller(episode.result, episode_length)
        return episode

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
        range_max = 2.0
        indexes = scan > range_max
        scan[indexes] = range_max
        self.current_laser_scan = scan

    def get_state(self):
        return np.array(self.current_laser_scan.tolist() + [self.current_pose['x'], self.current_pose['y']])

    def execute_action(self, action, step):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        old_block_pose = get_model_state_prox('block','').pose

        old_arm_pose = copy.copy(self.current_pose)
        self.current_pose['x'] += action['x']
        self.current_pose['y'] += action['y']
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
        reward = (old_dist - new_dist)*500.0

        if result['success']:
            reward += 100.0
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


class TeleopController(Controller):
    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.config = DemoConfig(bc_learning_rates = [0.0001, 0.0001],
                                bc_steps_per_frame = 10, td_max = 0.5)
        self.exp = Experience(window_size = float('inf'), prior_alpha = 0.3, prior_beta = 0.2, length_scale = 2.0)

    def get_controller_confidence(self):
        return 1.0, 0.0

    def update_learnt_controller(self, result, episode_length):

        # we don't do a behaviour cloning update on unsuccessful episodes
        if not result:
            return

        learnt_controller_exists = False
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

    def get_action(self, state, step):
        dx = self.time_step*self.key_vel.linear.x
        dy = self.time_step*self.key_vel.angular.z
        return {'x':dx, 'y':dy}

class LearntController(Controller):

    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.type = 'learnt'
        nominal_means = np.array([0.02, 0.0])
        nominal_sigma_exps = np.array([-5.5, -5.5])
        self.policy = ActorNN(nominal_means, nominal_sigma_exps)
        self.exp = Experience(window_size = 50, prior_alpha = 0.2, prior_beta = 0.3, length_scale = 2.0)
        self.config = LearntConfig(rl_buffer_frames_min = 500000, ac_learning_rates = [0.00001, 0.00001],
                                rl_steps_per_frame = 5, td_max = 0.5)



    def get_controller_confidence(self):
        confidence, sigma, _, _ = self.exp.get_state_value(self.get_state())
        return confidence, sigma

    def get_action(self, state, step):
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

class DDPGController(Controller):

    def __init__(self, sim):
        Controller.__init__(self, sim)
        self.type = 'ddpg'
        self.exp = Experience(window_size = 50, prior_alpha = 0.2, prior_beta = 0.3, length_scale = 2.0)
        self.action_space_high = np.array([0.05, 0.03])
        self.action_space_low = np.array([-0.03, -0.03])

        laser_low = np.zeros(50)
        arm_low = np.array([0.0, -0.3])
        laser_high = np.ones(50)*2
        arm_high = ([0.8, 0.3])
        self.state_high = np.concatenate((laser_high, arm_high))
        self.state_low = np.concatenate((laser_low, arm_low))

        cuda = torch.cuda.is_available() #check for CUDA
        self.device   = torch.device("cuda" if cuda else "cpu")
        state_dim = 52
        action_dim = 2
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        self.critic  = Critic(state_dim, action_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic  = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.q_optimizer  = opt.SGD(self.critic.parameters(),  lr=0.001)#, weight_decay=0.01)
        self.policy_optimizer = opt.SGD(self.actor.parameters(), lr=0.00005)


        self.memory = ReplayBuffer(50000)
        self.plot_reward = []
        self.plot_policy = []
        self.plot_q = []
        self.plot_steps = []
        self.epsilon = 1
        self.epsilon_decay = 1e-6
        self.buffer_start = 200
        self.batch_size = 128
        self.tau = 0.02
        self.gamma = 0.99
        self.episode_number = 0

    def get_controller_confidence(self):
        return 1.0, 0.0


    def run_episode(self, case_name, case_number):
        print("Starting new episode with controller type: %s" % (self.type))
        self.current_pose = copy.copy(self.init_pose)
        self.begin_new_episode(case_name, case_number)
        episode_running = True
        ep_q_value = 0.
        episode_reward = 0.0
        step = 0
        self.noise.reset()
        while episode_running:
            self.epsilon -= self.epsilon_decay
            state = self.get_state()
            state_norm = self.to_normalised_state(state)
            action_norm = self.actor.get_action(state)
            action_norm += self.noise()*max(0, self.epsilon)*0.5
            action_norm = np.clip(action_norm, -1., 1.)

            action = self.to_action(action_norm)

            new_state, reward, result, episode_running = self.execute_action({'x':action[0], 'y':action[1]}, step)
            new_state_norm = self.to_normalised_state(new_state)
            terminal = not episode_running
            if step == 0:
                print(action)
            self.memory.add(state_norm, action_norm, reward, terminal, new_state_norm)
            if self.memory.count() > self.buffer_start:
                policy_loss, q_loss = self.update()
            episode_reward += reward
            step += 1


        self.plot_reward.append([episode_reward, self.episode_number+1])
        self.plot_steps.append([step+1, self.episode_number+1])
        try:
            self.plot_policy.append([policy_loss.data, self.episode_number+1])
            self.plot_q.append([q_loss.data, self.episode_number+1])
        except:
            pass
        self.episode_number += 1


        if result['success']:
            print('Episode succeeded.')
        else:
            print('Episode failed by %s\n' % (result['failure_mode']))
        episode = self.end_episode(result)
        print(episode_reward)
        return episode, episode_reward



    def update(self):
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.memory.sample(self.batch_size)
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        a_batch = torch.FloatTensor(a_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(self.device)
        s2_batch = torch.FloatTensor(s2_batch).to(self.device)


        #compute loss for critic
        a2_batch = self.target_actor(s2_batch)
        target_q = self.target_critic(s2_batch, a2_batch) #detach to avoid updating target
        y = r_batch + (1.0 - t_batch) * self.gamma * target_q.detach()
        q = self.critic(s_batch, a_batch)

        self.q_optimizer.zero_grad()
        MSE = nn.MSELoss()
        q_loss = MSE(q, y) #detach to avoid updating target
        q_loss.backward()
        self.q_optimizer.step()

        #compute loss for actor
        self.policy_optimizer.zero_grad()
        policy_loss = -self.critic(s_batch, self.actor(s_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        #soft update of the frozen target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        return policy_loss, q_loss

    def get_action(self, state, step):
        action_normalised = self.agent.get_action(np.array(state))
        action = self.to_action(action_normalised)
        if step == 0:
            print(action)
        action = self.noise.get_action(action, step)
        return {'x':action[0], 'y':action[1]}

    def get_controller_confidence(self):
        confidence, sigma, _, _ = self.exp.get_state_value(self.get_state())
        return confidence, sigma

    def to_state(self, state):
        state_k = (self.state_high - self.state_low)/ 2.
        state_b = (self.state_high + self.state_low)/ 2.
        return state_k * state + state_b

    def to_action(self, action):
        act_k = (self.action_space_high - self.action_space_low)/ 2.
        act_b = (self.action_space_high + self.action_space_low)/ 2.
        return act_k * action + act_b

    def to_normalised_state(self, state):
        state_k_inv = 2./(self.state_high - self.state_low)
        state_b = (self.state_high + self.state_low)/ 2.
        return state_k_inv * (state - state_b)

    def to_normalised_action(self, action):
        act_k_inv = 2./(self.action_space_high - self.action_space_low)
        act_b = (self.action_space_high + self.action_space_low)/ 2.
        return act_k_inv * (action - act_b)

    def add_to_memory(self, state, action, reward, new_state, result):
        self.exp.add_step(state, action)
        action_normalised = self.to_normalised_action(np.array([action['x'], action['y']]))
        self.agent.memory.push(state, action_normalised, reward, new_state, result['success'])
        if len(self.agent.memory) > self.batch_size:
            self.agent.update(self.batch_size)

class DDPGConfig:
    def __init__(self):
        pass

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
