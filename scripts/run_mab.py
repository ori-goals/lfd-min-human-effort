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


def baseline_only():
    repeats = 5
    episodes = 1200
    save_folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/baseline_only/baseline2'
    for i in range(repeats):
        sim = Simulation(alpha = 0.0)
        baseline_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/baseline_controllers/baseline2.pkl'
        #baseline_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/400demos/2019-08-29-10-54_joystick_teleop400_ddpg800.pkl'
        sim.add_controllers({'baseline':{'file':baseline_file}})
        case_name = 'final_cases'
        case_count = 0
        for case_number in np.random.choice(episodes, episodes, replace=False):
            sim.run_new_episode(case_name, case_number, controller_type = 'baseline')
            episode_complete()
            if (case_count + 1) % 100 == 0:
                sim.save_simulation(save_folder)
            case_count += 1

def train_baseline():
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    repeats = 1
    episodes = 1200
    alphas = [0.5, 1.0, 2.0]
    folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/baselines'
    sim = Simulation(alpha=0.0) # alpha doesn't matter
    sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
    case_name = 'final_cases'
    case_count = 0
    for case_number in np.random.choice(episodes, episodes, replace=False):
        if case_count < 80:
            sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
        else:
            sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')
        if (case_count + 1) % 50 == 0:
            sim.save_simulation(folder)
        case_count += 1
        episode_complete()
    sim.save_simulation(save_folder)

def human_learner_mab():
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    repeats = 1
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

def human_learner_baseline_mab():
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    baseline_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/baseline_controllers/baseline2.pkl'
    repeats = 1
    episodes = 1200
    alphas = [1.0]
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0']
    for alpha_ind in range(len(alphas)):
        alpha = alphas[alpha_ind]
        save_folder = folders[alpha_ind]
        for i in range(repeats):
            sim = Simulation(alpha=alpha)
            sim.add_controllers({'ddpg':{}, 'baseline':{'file':baseline_file}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
            case_name = 'final_cases'
            case_count = 0
            for case_number in np.random.choice(episodes, episodes, replace=False):
                sim.run_new_episode(case_name, case_number, switching_method = 'contextual_bandit')
                episode_complete()
                if (case_count + 1) % 100 == 0:
                    sim.save_simulation(save_folder)
                case_count += 1
            sim.save_simulation(save_folder)

def human_learner_baseline_ncb():
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    baseline_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/baseline_controllers/baseline2.pkl'
    repeats = 5
    episodes = 1200
    alphas = [1.0]
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_ncb/alpha_1_0']
    for alpha_ind in range(len(alphas)):
        alpha = alphas[alpha_ind]
        save_folder = folders[alpha_ind]
        for i in range(repeats):
            sim = Simulation(alpha=alpha)
            sim.add_controllers({'ddpg':{}, 'baseline':{'file':baseline_file}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
            case_name = 'final_cases'
            case_count = 0
            for case_number in np.random.choice(episodes, episodes, replace=False):
                sim.run_new_episode(case_name, case_number, switching_method = 'non_contextual_bandit')
                episode_complete()
                if (case_count + 1) % 100 == 0:
                    sim.save_simulation(save_folder)
                case_count += 1
            sim.save_simulation(save_folder)

def human_learner_baseline_softmax():
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    baseline_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/baseline_controllers/baseline2.pkl'
    repeats = 2
    episodes = 1200
    taus = [0.01]
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_01']
    for tau_ind in range(len(taus)):
        tau = taus[tau_ind]
        save_folder = folders[tau_ind]
        for i in range(repeats):
            sim = Simulation(delta_tau = tau)
            sim.add_controllers({'ddpg':{}, 'baseline':{'file':baseline_file}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
            case_name = 'final_cases'
            case_count = 0
            for case_number in np.random.choice(episodes, episodes, replace=False):
                sim.run_new_episode(case_name, case_number, switching_method = 'softmax')
                episode_complete()
                if (case_count + 1) % 100 == 0:
                    sim.save_simulation(save_folder)
                case_count += 1
            sim.save_simulation(save_folder)

def human_then_learner():
    human_episodes = [200]
    episodes = 1200
    repeats = 1
    save_folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/200demos',
                    '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/300demos',
                    '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/400demos']
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

def limited_demos_human_then_learner():
    num_human_episodes = 150
    demos_avail = 1200
    num_total_episodes = 2405
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    save_folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/limited_demos/human150_then_learner'
    sim = Simulation(alpha=0.0) # alpha doesn't matter
    sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
    case_name = 'final_cases'
    case_count = 0
    for case_number in np.random.choice(demos_avail, num_total_episodes, replace=True):
        if case_count < num_human_episodes:
            sim.run_new_episode(case_name, case_number, controller_type = 'joystick_teleop')
        else:
            sim.run_new_episode(case_name, case_number, controller_type = 'ddpg')
        if (case_count + 1) % 200 == 0:
            sim.save_simulation(save_folder)
        case_count += 1
        episode_complete()
    sim.save_simulation(save_folder)

def human_learner_mab_limited_demos():
    num_human_episodes = 150
    demos_avail = 1200
    num_total_episodes = 2405
    saved_controller_file = '/home/marcrigter/ros/learn_to_manipulate_ws/src/learn_to_manipulate/config/demos/demo_final_cases_0_1399.pkl'
    save_folder = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/limited_demos/mab_alpha_2_0'
    sim = Simulation(alpha=2.0, max_demos=150)
    sim.add_controllers({'ddpg':{}, 'saved_teleop':{'file':saved_controller_file, 'type':'joystick_teleop'}})
    case_name = 'final_cases'
    case_count = 0
    for case_number in np.random.choice(demos_avail, num_total_episodes, replace=True):
        sim.run_new_episode(case_name, case_number, switching_method = 'contextual_bandit')
        if (case_count + 1) % 200 == 0:
            sim.save_simulation(save_folder)
        case_count += 1
        episode_complete()
    sim.save_simulation(save_folder)

if __name__ == "__main__" :
    rospy.init_node('learn_to_manipulate')
    #limited_demos_human_then_learner()
    human_learner_mab_limited_demos()
    #human_learner_mab()
    #human_then_learner()
    #human_learner_baseline_mab()
    #human_learner_baseline_softmax()
    #human_learner_baseline_softmax()
    #train_baseline()
    #baseline_only()

    #human_learner_baseline_softmax()
