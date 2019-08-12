#!/usr/bin/env python
import numpy as np
import pandas as pd
import rospy, os, time, pickle, math, copy, random
from geometry_msgs.msg import Point, Pose, Twist
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Int16, String


class Episode():
    def __init__(self, df, confidence, sigma, result, failure_mode, case_number, case_name):
        self.episode_df = df
        self.confidence = confidence
        self.sigma = sigma
        self.result = result
        self.failure_mode = failure_mode
        self.case_number = case_number
        self.case_name = case_name

class Experience(object):
    def __init__(self, window_size, prior_alpha, prior_beta, length_scale):
        self.window_size = window_size
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.length_scale = length_scale
        self.col_names = ['state', 'dx', 'dy', 'reward', 'return']
        self.replay_buffer = pd.DataFrame(columns = self.col_names)
        self.episode_list = []
        self.replay_buffer_episodes = []

    def new_episode(self, episode_confidence, sigma, case_name, case_number):
        self.episode_df = pd.DataFrame(columns = self.col_names)
        self.episode_confidence = episode_confidence
        self.episode_confidence_sigma = sigma
        self.episode_case_number = case_number
        self.episode_case_name = case_name

    def add_step(self, state, action):
        self.episode_df.loc[len(self.episode_df)] = [state, action[0], action[1], -1.0, -1.0]

    def end_episode(self, result):
        self.store_episode_result(result['success'])
        episode = Episode(df = self.episode_df, confidence = self.episode_confidence,
            sigma = self.episode_confidence_sigma, result = result['success'], failure_mode = result['failure_mode'],
            case_number = self.episode_case_number, case_name = self.episode_case_name)
        self.episode_list.append(episode)
        self.add_to_replay_buffer(episode)
        return episode

    def add_saved_episode(self, episode):
        self.episode_list.append(episode)
        self.add_to_replay_buffer(episode)

    def add_to_replay_buffer(self, episode):

        # insert new episode into the window
        self.replay_buffer_episodes.insert(0, episode)
        if self.window_size != float('inf'):
            self.replay_buffer_episodes = self.replay_buffer_episodes[0:self.window_size]

        # construct the replay buffer for learner from the window of experience
        self.replay_buffer = pd.DataFrame(columns = self.col_names)
        for episode in self.replay_buffer_episodes:
            df_reverse = episode.episode_df.reindex(index=episode.episode_df.index[::-1])
            self.replay_buffer = pd.concat([self.replay_buffer, df_reverse], ignore_index=True)


    def store_episode_result(self, result):
        if result:
            episode_return = 1.0
        else:
            episode_return = 0.0
        self.episode_df['return'] = episode_return
        self.episode_df.at[(len(self.episode_df)-1), 'reward'] = episode_return

    def get_state_value(self, state):
        alpha = copy.copy(self.prior_alpha)
        beta = copy.copy(self.prior_beta)
        length_scale = self.length_scale

        for index, row in self.replay_buffer.iterrows():
            old_state = np.array(row['state'])
            state_delta = old_state - state

            # calculate the difference between the two states
            weight = np.exp(-1.0*(np.linalg.norm(state_delta/length_scale))**2)
            if weight < 1e-7:
                weight = 1e-7

            if np.isnan(round(row["return"])):
                print(row["return"])
                print(self.replay_buffer)

            # increment alpha for success and beta for failure
            if int(round(row["return"])) == 1:
                alpha = alpha + weight
            else:
                beta = beta + weight

        value = alpha/(alpha + beta)
        variance = alpha*beta/((alpha+beta)**2*(alpha+beta+1.0))
        sigma = np.sqrt(variance)
        return value, sigma, alpha, beta
