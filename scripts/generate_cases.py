#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation


if __name__ == "__main__" :
    sim = Simulation.generate_cases('harder_rl_attempt_aug11', 2000,
    {'min_y':0.05, 'max_y':-0.05, 'min_x':0.37, 'max_x':0.4, 'min_angle_deg':-20, 'max_angle_deg':20, 'block_names':['block_30']})
