#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *

def plot_length_param():
    known_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_with_baseline.pkl'
    unknown_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_different_with_baseline.pkl'
    plot_length_param_estimation(known_episodes_file, unknown_episodes_file)

def plot_rel_use():
    plot_relative_usage('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_ddpg_mab/bc_0_001_alpha_0_3', 800)

def plot_success_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_ddpg_mab/bc_0_001_alpha_0_3', '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_ddpg_mab/bc_0_01_alpha_0_03']
    methods = ['bc = 0.001', 'bc = 0.01']
    plot_ddpg_success_rate(folders, methods, 410)

def plot_human_cost():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_ddpg_mab/bc_0_001_alpha_0_3',
    '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_ddpg_mab/bc_0_01_alpha_0_03',
    '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_then_learner/lr_bc_0_01_alpha_0_3',
    '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/initial_tests/tuning/human_then_learner/lr_bc_0_001_alpha_0_3']
    methods = ['mab, bc = 0.001', 'mab, bc = 0.01', 'human then learner, bc = 0.01', 'human then learner, bc = 0.001']
    plot_human_cost_sliding_window(folders, methods, num_episodes = 800, cost_failure = 5.0)

if __name__ == "__main__" :
    #plot_rel_use()
    plot_human_cost()
    #plot_success_rate()
