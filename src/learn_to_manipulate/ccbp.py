#!/usr/bin/env python
import numpy as np
import pandas as pd
import rospy, os, time, pickle, math, copy, random
from geometry_msgs.msg import Point, Pose, Twist
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Int16, String


def get_planner_state_value(state, replay_buffer, length_scale, prior):
    alpha = prior.planner_alpha
    beta = prior.planner_beta

    for index, row in replay_buffer.iterrows():
        experience_state = np.array(row["laser_state"])
        state_delta = experience_state - state

        # calculate the difference between the two states
        weight = np.exp(-1.0*(np.linalg.norm(state_delta/length_scale))**2)
        if weight < 1e-7:
            weight = 1e-7

        # increment alpha for success and beta for failure
        if int(round(row["return"])) == 1:
            alpha = alpha + weight
        else:
            beta = beta + weight

    value = alpha/(alpha + beta)
    variance = alpha*beta/((alpha+beta)**2*(alpha+beta+1.0))
    sigma = np.sqrt(variance)
    return value, sigma, alpha, beta

def get_learner_state_value(state, replay_buffer, length_scale, prior):
    alpha = prior.agent_alpha
    beta = prior.agent_beta

    for index, row in replay_buffer.iterrows():
        experience_state = np.array(row["laser_state"])
        state_delta = experience_state - state

        # calculate the difference between the two states
        weight = np.exp(-1.0*(np.linalg.norm(state_delta/length_scale))**2)
        if weight < 1e-7:
            weight = 1e-7

        # increment alpha for success and beta for failure
        if int(round(row["return"])) == 1:
            alpha = alpha + weight
        else:
            beta = beta + weight

    value = alpha/(alpha + beta)
    variance = alpha*beta/((alpha+beta)**2*(alpha+beta+1.0))
    sigma = np.sqrt(variance)
    return value, sigma, alpha, beta
