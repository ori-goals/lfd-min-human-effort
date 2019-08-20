#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import matplotlib.pyplot as plt
import rospy
import logging
import numpy as np
import os

def  episode_complete():
    my_path = os.path.abspath(os.path.dirname(__file__))
    my_path = os.path.join(my_path, "../scripts/.run_completed.txt")
    file = open(my_path,"w")
    file.close()

def human_learner_mab():
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    repeats = 5
    episodes = 1200
    alphas = [2.0]
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_2_0']
    for alpha_ind in range(len(alphas)):
        alpha = alphas[alpha_ind]
        save_folder = folders[alpha_ind]
        for i in range(repeats):
            sim = Simulation(alpha=alpha)
            sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
            case_name = 'final_cases'
            case_count = 0
            for case_number in np.random.choice(episodes, episodes, replace=False):
                sim.run_new_episode(case_name, case_number, switching_method = 'contextual_bandit')
                episode_complete()
                if (case_count + 1) % 100 == 0:
                    sim.save_simulation(save_folder)
                case_count += 1
            sim.save_simulation(save_folder)


def human_then_learner():
    human_episodes = [100, 200, 300]
    episodes = 1200
    repeats = 3
    save_folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/100demos',
                    '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/200demos',
                    '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/300demos']
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    for ind in range(len(human_episodes)):
        num_human_episodes = human_episodes[ind]
        save_folder = save_folders[ind]
        for i in range(repeats):
            sim = Simulation(alpha=0.0) # alpha doesn't matter
            sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
            case_name = 'final_cases'
            case_count = 0
            for case_number in np.random.choice(episodes, episodes, replace=False):
                if case_count < num_human_episodes:
                    sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
                else:
                    sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')
                if (case_count + 1) % 100 == 0:
                    sim.save_simulation(save_folder)
                case_count += 1
                episode_complete()
            sim.save_simulation(save_folder)

if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    #human_learner_mab()
    human_then_learner()
