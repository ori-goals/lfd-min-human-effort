#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *


if __name__ == "__main__" :
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/rl_only']
    methods = ['rl_only']
    plot_human_cost_sliding_window(folders, methods, max_episodes = 60)
