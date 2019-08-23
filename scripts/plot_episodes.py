#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *

def plot_length_param():
    known_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_with_baseline.pkl'
    unknown_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_different_with_baseline.pkl'
    plot_length_param_estimation(known_episodes_file, unknown_episodes_file)

def plot_rel_use():
    plot_relative_usage('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0', 600)

def plot_success_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/baselines']
    methods = ['alpha = 1.0', 'alpha = 0.5', 'alpha = 2.0']
    plot_ddpg_success_rate(folders, methods, 420)


def plot_baseline_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0']
    methods = ['alpha = 1.0', 'alpha = 0.5', 'alpha = 2.0']
    plot_baseline_success_rate(folders, methods, 230)

def plot_human_cost():
    folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_5',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_2_0',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/200demos',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/300demos',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/400demos']
    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/baseline_only/baseline2']
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0']


    methods = ['alpha = 0.5', 'alpha = 1.0', 'alpha = 2.0', '200 demos', '300 demos', '400 demos']
    plot_human_cost_sliding_window(folders, methods, num_episodes = 1200, cost_failure = 5.0)

if __name__ == "__main__" :
    #plot_rel_use()
    #plot_baseline_rate()
    plot_human_cost()
#    plot_success_rate()
