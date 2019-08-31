#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *

def plot_length_param():
    known_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/setup/hyperparam_tuning/length_scale/experience_with_baseline/100_with_baseline.pkl'
    unknown_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/setup/hyperparam_tuning/length_scale/experience_with_baseline/100_different_with_baseline.pkl'
    plot_length_param_estimation(known_episodes_file, unknown_episodes_file, n_buffer=50)

def plot_rel_use():
    plot_relative_usage('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_02/short_files', 1200)

def plot_success_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/limited_demos/human150_then_learner']
    methods = ['alpha = 1.0', 'alpha = 0.5', 'alpha = 2.0']
    plot_ddpg_success_rate(folders, methods, 2200)


def plot_baseline_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0']
    methods = ['alpha = 1.0', 'alpha = 0.5', 'alpha = 2.0']
    plot_baseline_success_rate(folders, methods, 230)

def plot_human_cost():
    folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_5/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_2_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/400demos/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_01/short_files']

    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/200demos/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/300demos/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/400demos/short_files']
    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0']

    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_ncb/alpha_1_0',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0']

    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_005/short_files',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_01/short_files',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_02/short_files',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_04']

    #folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_3',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_2_0']
    methods = [r'$\it{contextual\ bandit\ (\alpha = 0.5)}$', r'$\it{contextual\ bandit\ (\alpha = 1)}$', r'$\it{contextual\ bandit\ (\alpha = 2)}$',
                r'$\it{contextual\ bandit\ w.\ baseline\ (\alpha = 1)}$', r'$\it{human\ then\ learner\ (n_h = 400)}$', r'$\it{Boltzmann\ (\tau=0.01)}$', '', '', '']
    plot_human_cost_sliding_window(folders, methods, num_episodes = 1200, cost_failure = 5.0)

def shorten_files():
    folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_01']
    create_short_files(folders)

if __name__ == "__main__" :
    #shorten_files()
    #plot_length_param()
    #plot_rel_use()
    #plot_baseline_rate()
    #plot_human_cost()
    plot_success_rate()
