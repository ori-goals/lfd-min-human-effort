#!/usr/bin/env python
import numpy as np
import pandas as pd
import rospy, os, time, pickle, math, copy, random
from geometry_msgs.msg import Point, Pose, Twist
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Int16, String


class Episode():
    def __init__(self, df, confidence, result, failure_mode, case_number):
        self.df = df
        self.confidence = confidence
        self.result = result
        self.failure_mode = failure_mode
        self.case_number = case_number

class Experience(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.col_names = ['state', 'dx', 'dy', 'reward', 'return']
        self.episode_list = []
        self.replay_buffer_episodes = []

    def new_episode(self, episode_confidence, case_number):
        self.episode_df = pd.DataFrame(columns = self.col_names)
        self.episode_confidence = episode_confidence
        self.episode_case_number = case_number

    def add_step(self, state, action):
        self.episode_df.loc[len(self.episode_df)] = [state, action['x'], action['y'], -1.0, -1.0]

    def end_episode(self, result):
        self.store_episode_result(result)
        print(self.episode_df)
        episode = Episode(df = self.episode_df, confidence = self.episode_confidence,
            result = result, failure_mode = '', case_number = self.episode_case_number)
        self.episode_list.append(episode)
        self.add_to_replay_buffer(episode)

    def add_to_replay_buffer(self, episode):

        # insert new episode into the window
        self.replay_buffer_episodes.insert(0, episode)
        self.replay_buffer_episodes = self.replay_buffer_episodes[0:self.window_size]

        # construct the replay buffer for learner from the window of experience
        self.replay_buffer_tuples = pd.DataFrame(columns = self.col_names)
        for episode in self.replay_buffer_episodes:
            df_reverse = episode.df.reindex(index=episode.df.index[::-1])
            self.replay_buffer_tuples = pd.concat([self.replay_buffer_tuples, df_reverse], ignore_index=True)


    def store_episode_result(self, result):
        if result:
            episode_return = 1.0
        else:
            episode_return = 0.0
        self.episode_df['return'] = episode_return
        self.episode_df.at[(len(self.episode_df) - 1), 'reward'] = episode_return
