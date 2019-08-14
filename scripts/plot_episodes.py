#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *


if __name__ == "__main__" :
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/bc_loss/150demos_then_rl/0_001', '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/bc_loss/150demos_then_rl/0_01']
    methods = ['150_demos_then_rl, bc = 0.001', '150_demos_then_rl, bc = 0.01']
    plot_ddpg_success_rate(folders, methods, max_episodes = 600)
