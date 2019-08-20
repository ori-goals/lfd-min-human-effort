#!/usr/bin/env python
import tf
import numpy as np
from collections import deque
import random
import geometry_msgs
import rospy
import pickle
import time
from learn_to_manipulate.experience import Experience

def qv_rotate(q1, v1):
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2),
        tf.transformations.quaternion_conjugate(q1))[0:3]

def join_demo_files(file1, file2, controller_type, save_folder):
    file1 = open(file1,"rb")
    file2 = open(file2,"rb")
    all_runs1, controller_save_info1 = pickle.load(file1)
    all_runs2, controller_save_info2 = pickle.load(file2)
    for save_info in controller_save_info1:
        if save_info['type'] == controller_type:
            experience1 = save_info['experience']
    for save_info in controller_save_info2:
        if save_info['type'] == controller_type:
            experience2 = save_info['experience']

    new_experience = Experience(float('inf'), 1, 1, 1)
    all_runs = experience1.episode_list + experience2.episode_list
    new_experience.episode_list = all_runs
    for episode in new_experience.episode_list:
        print(episode.case_number)
    save_info = {'type':controller_type, 'experience':new_experience, 'config':[]}
    fname = time.strftime("%Y-%m-%d-%H-%M") + '_merged'
    new_file_path = save_folder + '/' + fname + '.pkl'

    with open(new_file_path, 'w') as f:
        pickle.dump([all_runs, [save_info]], f)
        f.close

def move_arm_initial(contr):
    joint_goal = contr.group.get_current_joint_values()
    joint_goal[0] = -0.3981706
    joint_goal[1] = -2.1341021
    joint_goal[2] = 2.1484844
    joint_goal[3] = -1.6805012
    joint_goal[4] = -1.530783
    joint_goal[5] = -0.3962608
    contr.group.go(joint_goal, wait=True)
    contr.group.stop()

    contr.go_to_pose(contr.init_pose)
    rospy.sleep(1.0)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp=0
        self.buffer=deque()

    def add(self, s, a, r, t, s2):
        experience=(s, a, r, t, s2)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)

        s, a, r, t, s2 = map(np.stack, zip(*batch))

        return s, a, r, t, s2

    def clear(self):
        self.buffer = deque()
        self.num_exp=0
