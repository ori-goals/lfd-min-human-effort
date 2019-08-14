#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *


if __name__ == "__main__" :
    folders = ['/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/tuning_problem_domain/final_cases/demos_then_rl']
    methods = ['170_demos_then_rl']
    plot_ddpg_success_rate(folders, methods, max_episodes = 700)
