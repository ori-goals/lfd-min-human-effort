#!/usr/bin/env python
from learn_to_manipulate.simulate import Simulation


if __name__ == "__main__" :
    sim = Simulation.generate_cases('lfd_rl_aug10', 2000,
    {'min_y':0.02, 'max_y':-0.02, 'min_x':0.35, 'max_x':0.38, 'min_angle_deg':-5, 'max_angle_deg':5, 'block_names':['block_30']})
