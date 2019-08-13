#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *


if __name__ == "__main__" :
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/ddpg_bc_testing/add_q_filter/q_filter_mods/demos_then_rl', '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/ddpg_bc_testing/no_q_filter/rl_only']
    methods = ['40_demos_then_rl', 'rl_only']
    plot_ddpg_success_rate(folders, methods, max_episodes = 150)
