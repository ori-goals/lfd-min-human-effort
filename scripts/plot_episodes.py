#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *

def plot_length_param():
    known_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/setup/hyperparam_tuning/length_scale/experience_with_baseline/100_with_baseline.pkl'
    unknown_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/setup/hyperparam_tuning/length_scale/experience_with_baseline/100_different_with_baseline.pkl'
    plot_length_param_estimation(known_episodes_file, unknown_episodes_file, n_buffer=50)

def plot_rel_use():
    plot_relative_usage('/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/demo_cost_0_143/alpha_1_0/short_files', 1200)

def plot_success_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/limited_demos/mab_alpha_2_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/limited_demos/human150_then_learner/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/limited_demos/human150_then_learner_eps_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/rl_only/noise_factor_1_0/short_files']
    methods = ['alpha 2', 'human 150 then learner', 'human 150 then learner eps 0', 'rl only']
    plot_ddpg_success_rate(folders, methods, 1200)


def plot_baseline_rate():
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0']
    methods = ['alpha = 1.0', 'alpha = 0.5', 'alpha = 2.0']
    plot_baseline_success_rate(folders, methods, 230)

def plot_human_cost():
    folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/demo_cost_0_2/alpha_0_5/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/demo_cost_0_2/alpha_1_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/demo_cost_0_2/alpha_2_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_mab/alpha_1_0/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/300demos/short_files',
                '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_01/short_files']


    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/400demos_repeat/short_files']

    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_ncb/alpha_1_0',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0']

    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_005/short_files',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_01/short_files',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_02/short_files',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_baseline_softmax/dtau_0_04']

    #folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_0_3',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_1_0',
    #            '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_learner_mab/alpha_2_0']
    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/rl_only/single_file']
    methods = [r'$\it{contextual\ bandit\ (\alpha = 0.5)}$', r'$\it{contextual\ bandit\ (\alpha = 1)}$', r'$\it{contextual\ bandit\ (\alpha = 2)}$',
                r'$\it{contextual\ bandit\ w.\ baseline\ (\alpha = 1)}$', r'$\it{human\ then\ learner\ (n_h = 300)}$', r'$\it{Boltzmann\ (\Delta \tau=0.002)}$', 'rl only', '', '']
    #folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/rl_only']
    plot_human_cost_sliding_window(folders, methods, num_episodes = 1200, cost_failure = 5.0)

def shorten_files():
    folders =  ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/main_experiment/human_then_learner/300demos_repeat']
    create_short_files(folders)

if __name__ == "__main__" :
    #shorten_files()
    #shorten_files()
    #plot_length_param()
    #plot_rel_use()
    #plot_baseline_rate()
    plot_human_cost()
    #plot_success_rate()
