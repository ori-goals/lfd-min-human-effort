#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation
from learn_to_manipulate.plot import *


if __name__ == "__main__" :
    known_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_with_baseline.pkl'
    unknown_episodes_file = '/home/marcrigter/pCloudDrive/Development/LearnToManipulate/data/initial_tests/hyperparam_tuning/length_scale/experience_with_baseline/100_different_with_baseline.pkl'
    plot_length_param_estimation(known_episodes_file, unknown_episodes_file)
