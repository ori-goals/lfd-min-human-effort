#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *

def plot_length_param():
    known_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_with_baseline.pkl'
    unknown_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_different_with_baseline.pkl'
    plot_length_param_estimation(known_episodes_file, unknown_episodes_file)

def plot_rel_use():
    plot_relative_usage('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_5', 1400)

def plot_success_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_5']
    methods = ['alpha = 1.0', 'alpha = 0.5', 'alpha = 2.0']
    plot_ddpg_success_rate(folders, methods, 1000)

def plot_human_cost():
    folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_5',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_2_0']

    methods = ['alpha = 0.5', 'alpha = 1.0', 'alpha = 2.0']
    plot_human_cost_sliding_window(folders, methods, num_episodes = 1400, cost_failure = 5.0)

if __name__ == "__main__" :
    #plot_rel_use()
    plot_human_cost()
    #plot_success_rate()
