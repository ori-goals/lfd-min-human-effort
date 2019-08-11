#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation


if __name__ == "__main__" :
    sim = Simulation.generate_cases('rl_attempt_aug11', 2000,
    {'min_y':0.02, 'max_y':-0.02, 'min_x':0.37, 'max_x':0.4, 'min_angle_deg':-5, 'max_angle_deg':5, 'block_names':['block_30']})
