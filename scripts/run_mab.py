#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
import matplotlib.pyplot as plt
import rospy
import logging
import numpy as np

def human_learner_mab():
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_999.pkl'
    repeats = 5
    episodes = 1000
    for i in range(repeats):
        alpha = 0.3
        sim = Simulation(alpha=alpha)
        save_folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_3'
        sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
        case_name = 'final_cases'

        for case_number in np.random.choice(episodes, episodes, replace=False):
            sim.run_new_episode(case_name, case_number, switching_method = 'contextual_bandit')
            if (case_number + 1) % 100 == 0:
                sim.save_simulation(save_folder)
        sim.save_simulation(save_folder)

    for i in range(repeats):
        alpha = 0.1
        sim = Simulation(alpha=alpha)
        save_folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_1'
        sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
        case_name = 'final_cases'

        for case_number in np.random.choice(episodes, episodes, replace=False):
            sim.run_new_episode(case_name, case_number, switching_method = 'contextual_bandit')
            if (case_number + 1) % 100 == 0:
                sim.save_simulation(save_folder)
        sim.save_simulation(save_folder)

    for i in range(repeats):
        alpha = 1.0
        sim = Simulation(alpha=alpha)
        save_folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0'
        sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
        case_name = 'final_cases'

        for case_number in np.random.choice(episodes, episodes, replace=False):
            sim.run_new_episode(case_name, case_number, switching_method = 'contextual_bandit')
            if (case_number + 1) % 100 == 0:
                sim.save_simulation(save_folder)
        sim.save_simulation(save_folder)

def human_then_learner():
    human_episodes = 250
    for i in range(0, 1):
        sim = Simulation(alpha=0.3)
        saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_999.pkl'
        sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
        case_name = 'final_cases'

        for case_number in range(0, 810):
            if case_number < human_episodes:
                sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
            else:
                sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')
            if (case_number - 1) % 100 == 0:
                sim.save_simulation('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_then_learner/lr_bc_0_01_alpha_0_3')

if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    human_learner_mab()
    human_then_learner()
