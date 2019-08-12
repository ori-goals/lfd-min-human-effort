#!/usr/bin/env python
import tf
import numpy as np
from collections import deque
import random
import geometry_msgs
import rospy

def qv_rotate(q1, v1):
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2),
        tf.transformations.quaternion_conjugate(q1))[0:3]

def move_arm_initial(contr):
    qxs = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, contr.init_pose['qx']]
    qys = [0.0, 0.0, 0.0, 0.0, 0.0, 0.6697, 0.6697, contr.init_pose['qy']]
    qzs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, contr.init_pose['qz']]
    qws = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7426, 0.7426, contr.init_pose['qw']]
    xs = [0.816, 0.70, 0.7, 0.7, 0.7, 0.77, 0.77, contr.init_pose['x']]
    ys = [0.191, 0.20, 0.199, 0.2, 0.2, 0.2, 0.0, contr.init_pose['y']]
    zs = [0.0, 0.12, 0.253, 0.35, 0.35, 0.35, 0.37, contr.init_pose['z']]
    current_pose = contr.group.get_current_pose().pose
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
            contr.group.set_pose_target(pose_goal)
            plan = contr.group.go(wait=True)
            contr.group.stop()
            contr.group.clear_pose_targets()
        else:
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
